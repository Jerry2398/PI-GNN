import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from collections import Iterable

from torch_geometric.data import NeighborSampler
from .config import MODEL_CONFIG
from .sage import SAGEConv
import time
from threading import Thread


class MyThread(Thread):
    def __init__(self, func, args):
        super(MyThread, self).__init__()
        self.func = func
        self.args = args

    def run(self):
        self.result = self.func(*self.args)

    def get_result(self):
        try:
            return self.result
        except Exception:
            return None


class GraphSage(nn.Module):
    def __init__(self, in_dim, out_dim, multi_threading=True, normalize=False):
        super(GraphSage, self).__init__()
        self.out_dim = out_dim

        self.num_layers = MODEL_CONFIG['num_layer']
        self.normalize = normalize
        self.multi_threading = multi_threading
        self.expand_size = MODEL_CONFIG['expand_size']
        self.hidden_dim = MODEL_CONFIG['hidden_dim']

        self.batch_num = 4

        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(in_dim, self.hidden_dim[0], normalize=self.normalize))
        for i in range(self.num_layers-1):
            self.convs.append(SAGEConv(self.hidden_dim[i], self.hidden_dim[i+1], normalize=self.normalize))
        self.linear_w = nn.Parameter(torch.Tensor(self.hidden_dim[-1], out_dim))
        self.expand_w = None

        self.loss_fc = torch.nn.CrossEntropyLoss(reduction='mean')
        nn.init.xavier_uniform_(self.linear_w)

    def init_expanded_parameters(self):
        for i in range(self.num_layers):
            self.convs[i].init_expanded_parameters()
        if self.expand_w is not None:
            nn.init.xavier_uniform_(self.expand_w)

    def forward(self, x, adjs, phase='retrain'):
        for i, (edge_index, _, size) in enumerate(adjs):
            x_target = x[:size[1]]  # Target nodes are always placed first.
            x, old_out, new_out = self.convs[i](phase=phase, x=(x, x_target), edge_index=edge_index)
            x = F.leaky_relu(x)
            x = F.dropout(x, training=self.training)
        if self.expand_w is None:
            out = torch.matmul(x, self.linear_w)
        else:
            out = torch.matmul(x, torch.cat([self.linear_w, self.expand_w], dim=0))
        return out

    def inference(self, data, phase='retrain'):
        x, edge_index = data.x, data.edge_index
        x_list = []
        for i in range(self.num_layers):
            x, old_out, new_out = self.convs[i].forward(phase=phase, x=x, edge_index=edge_index)
            x = F.leaky_relu(x)
            x = F.dropout(x, training=self.training)
            x_list.append(x)
        if self.expand_w is None:
            out = torch.matmul(x, self.linear_w)
        else:
            out = torch.matmul(x, torch.cat([self.linear_w, self.expand_w], dim=0))
        return out

    def expand(self):
        self.convs[0].expand(0, self.expand_size[0])
        for i in range(1, self.num_layers):
            self.convs[i].expand(self.expand_size[i-1], self.expand_size[i])
        self.expand_w = nn.Parameter(torch.randn(size=(self.expand_size[-1], self.out_dim)), requires_grad=True)

    def isolate_parameters(self):
        for i in range(self.num_layers):
            self.convs[i].isolate_parameters()
        self.linear_w.requires_grad = False

    def open_parameters(self):
        for i in range(self.num_layers):
            self.convs[i].open_parameters()
        self.linear_w.requires_grad = True

    def combine(self):
        for i in range(self.num_layers):
            self.convs[i].combine()
        self.linear_w = nn.Parameter(torch.cat([self.linear_w.detach(), self.expand_w], dim=0))
        self.expand_w = None

    def print_params(self):
        print("parameters are:")
        print("parameters list:")
        for i in range(self.num_layers):
            print(self.convs[i].l_weights_list[0])
        print("expanded parameters:")
        for i in range(self.num_layers):
            if self.convs[i].expand_l_weights is not None:
                print(self.convs[i].expand_l_weights)

    def print_grad(self):
        print("grad is:")
        if self.expand_w is not None:
            print(self.expand_w.grad)
        for i in range(self.num_layers):
            if self.convs[i].expand_l_weights is not None:
                print(self.convs[i].expand_l_weights.grad)

    def print_params_name(self):
        for name, item in self.named_parameters():
            print(name)

    def batch_forward(self, x, y, data, phase):
        batch_size, n_id, adjs = data
        out = self.forward(x[n_id], adjs, phase)
        loss = self.loss_fc(out, y[n_id[:batch_size]])
        return loss

    def retrain(self, retrain_graph, phase, epoch, args):
        self.train()
        affected_train_length = len(retrain_graph.affected_train_mask)
        batch_s = affected_train_length // self.batch_num
        affected_train_loader = NeighborSampler(edge_index=retrain_graph.edge_index, node_idx=retrain_graph.affected_train_mask, sizes=[10, 10], shuffle=True, batch_size=batch_s)

        memory_train_length = len(retrain_graph.memory_train_mask)
        batch_s = memory_train_length // self.batch_num
        memory_train_loader = NeighborSampler(edge_index=retrain_graph.edge_index, node_idx=retrain_graph.memory_train_mask, sizes=[10, 10], shuffle=True, batch_size=batch_s)

        retrain_opt = optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=args.retrain_lr,
                                 weight_decay=args.retrain_weight_decay)

        x = retrain_graph.x
        y = retrain_graph.y

        loss_list = []
        loss1_list = []
        loss2_list = []

        total_time = 0
        for i, (memory_data, affected_data) in enumerate(zip(memory_train_loader, affected_train_loader)):
            if self.multi_threading:
                t1 = MyThread(self.batch_forward, args=(x, y, memory_data, phase))
                t2 = MyThread(self.batch_forward, args=(x, y, affected_data, phase))
                start_time = time.time()
                t1.start()
                t2.start()
                t1.join()
                t2.join()
                loss1 = t1.get_result()
                loss2 = t2.get_result()
            else:
                start_time = time.time()
                loss1 = self.batch_forward(x, y, memory_data, phase)
                loss2 = self.batch_forward(x, y, affected_data, phase)

            loss = args.beta2 * loss1 + loss2

            retrain_opt.zero_grad()
            loss.backward()
            retrain_opt.step()
            end_time = time.time()
            total_time = total_time + end_time - start_time

            loss_list.append(loss.item())
            loss1_list.append(loss1.item())
            loss2_list.append(loss2.item())

        average_loss = sum(loss_list) / len(loss_list)
        average_loss1 = sum(loss1_list) / len(loss1_list)
        average_loss2 = sum(loss2_list) / len(loss2_list)
        if epoch % 10 == 0:
            print(
                f"retrain epoch:\t{epoch} and loss:\t{average_loss}, loss1:{average_loss1}, loss2:{average_loss2}, beta1:{args.beta1}, samples:{len(retrain_graph.affected_train_mask) + len(retrain_graph.memory_train_mask)}")

        return average_loss, total_time

    def rectify(self, rectify_graph, phase, epoch, args):
        self.train()

        last_affected_train_length = len(rectify_graph.last_affected_train_mask)
        batch_s = last_affected_train_length // self.batch_num
        last_affected_train_loader = NeighborSampler(edge_index=rectify_graph.edge_index, node_idx=rectify_graph.last_affected_train_mask, sizes=[10, 10], shuffle=True, batch_size=batch_s)

        memory_train_length = len(rectify_graph.memory_train_mask)
        batch_s = memory_train_length // self.batch_num
        memory_train_loader = NeighborSampler(edge_index=rectify_graph.edge_index, node_idx=rectify_graph.memory_train_mask, sizes=[10, 10], shuffle=True, batch_size=batch_s)

        rectify_opt = optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=args.rectify_lr,
                                 weight_decay=args.rectify_weight_decay)
        x = rectify_graph.x
        y = rectify_graph.y

        loss_list = []
        loss1_list = []
        loss2_list = []

        total_time = 0
        for i, (memory_data, last_affected_data) in enumerate(zip(memory_train_loader, last_affected_train_loader)):
            if self.multi_threading:
                t1 = MyThread(self.batch_forward, args=(x, y, memory_data, phase))
                t2 = MyThread(self.batch_forward, args=(x, y, last_affected_data, phase))
                start_time = time.time()
                t1.start()
                t2.start()
                t1.join()
                t2.join()
                loss1 = t1.get_result()
                loss2 = t2.get_result()
            else:
                start_time = time.time()
                loss1 = self.batch_forward(x, y, memory_data, phase)
                loss2 = self.batch_forward(x, y, last_affected_data, phase)

            loss = loss1 - args.beta1 * loss2

            rectify_opt.zero_grad()
            loss.backward()
            rectify_opt.step()
            end_time = time.time()
            total_time = total_time + end_time - start_time

            loss_list.append(loss.item())
            loss1_list.append(loss1.item())
            loss2_list.append(loss2.item())

        average_loss = sum(loss_list)/len(loss_list)
        average_loss1 = sum(loss1_list)/len(loss1_list)
        average_loss2 = sum(loss2_list)/len(loss2_list)
        if epoch % 10 == 0:
            print(
                f"retrain epoch:\t{epoch} and loss:\t{average_loss}, loss1:{average_loss1}, loss2:{average_loss2}, beta1:{args.beta1}, samples:{len(rectify_graph.last_affected_train_mask) + len(rectify_graph.memory_train_mask)}")

        return average_loss, total_time

    def init_train(self, graph, phase, args):
        self.train()
        trainlength = len(graph.train_mask)
        batch_s = trainlength // self.batch_num
        train_loader = NeighborSampler(edge_index=graph.edge_index, node_idx=graph.train_mask, sizes=[10, 10], shuffle=True, batch_size=batch_s)
        opt = optim.Adam(self.parameters(), lr=args.init_lr, weight_decay=args.init_weight_decay)

        y = graph.y
        x = graph.x

        loss_list = []
        start_time = time.time()
        for batch_size, n_id, adjs in train_loader:
            opt.zero_grad()
            out = self.forward(x[n_id], adjs, phase)
            loss = self.loss_fc(out, y[n_id[:batch_size]])
            loss.backward()
            opt.step()
            loss_list.append(loss.item())
        end_time = time.time()

        average_loss = sum(loss_list)/len(loss_list)
        return average_loss, end_time - start_time
