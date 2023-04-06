import os
import torch
import argparse
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch_geometric.nn import SAGEConv
from torch.nn import Parameter

from data.dataset import DyDataset


MODEL_CONFIG = {
    'hidden_dim': [32, 32]
}


class Sage(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Sage, self).__init__()
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(in_channels=in_dim, out_channels=MODEL_CONFIG['hidden_dim'][0]))
        self.convs.append(SAGEConv(in_channels=MODEL_CONFIG['hidden_dim'][0], out_channels=MODEL_CONFIG['hidden_dim'][1]))
        self.w = Parameter(torch.Tensor(MODEL_CONFIG['hidden_dim'][1], out_dim))
        nn.init.xavier_uniform_(self.w)

    def inference(self, graph):
        x, edge_index = graph.x, graph.edge_index
        x = self.convs[0](x, edge_index)
        x = F.relu(x)
        x = self.convs[1](x, edge_index)
        x = F.relu(x)
        out = torch.matmul(x, self.w)
        return out

    def forward(self, x, adjs):
        for i, (edge_index, _, size) in enumerate(adjs):
            x_target = x[:size[1]]  # Target nodes are always placed first.
            x = self.convs[i](x=(x, x_target), edge_index=edge_index)
            x = F.relu(x)
            x = F.dropout(x, training=self.training)
        out = torch.matmul(x, self.w)
        return out


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda_enable', action='store', default=True, type=bool)
    parser.add_argument('--dataset_name', action='store', default='cora', type=str)
    parser.add_argument('--save_dir', action='store', default='save_models', type=str)
    parser.add_argument('--edge_type', action='store', default='stream_edges', type=str)
    parser.add_argument('--seed', action='store', default=2021, type=int)
    parser.add_argument('--m_size', action='store', default=64, type=int)  # 64, 256
    parser.add_argument('--t', action='store', default=10, type=int)

    parser.add_argument('--epochs', action='store', default=400, type=int, help='epochs')  #600, 400
    parser.add_argument('--lr', action='store', default=0.001, type=float)
    parser.add_argument('--weight_decay', action='store', default=0, type=float)
    args = parser.parse_args()
    return args


def calculate_test_acc(graph):
    student_model.eval()
    y = graph.y
    out = student_model.inference(graph)
    test_mask = graph.test_mask
    _, pred = out.max(dim=1)
    if test_mask.sum().item() > 0:
        correct = int(pred[test_mask].eq(y[test_mask]).sum().item())
        acc = correct / len(y[test_mask])
        return acc
    else:
        return 0.


if __name__ == "__main__":
    args = get_args()
    model_path = os.path.join(args.save_dir, args.dataset_name)
    dataset = DyDataset(dataset_name=args.dataset_name, edge_type=args.edge_type, m_size=args.m_size)

    in_dim = dataset.in_dim
    out_dim = dataset.labels_num
    _, _, graph = dataset[len(dataset)-1]

    teacher_model = torch.load(os.path.join(model_path, '13.pt'))

    student_model = Sage(in_dim=in_dim, out_dim=out_dim)
    opt = optim.Adam(params=student_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.BCELoss()

    for epoch in range(args.epochs):
        teacher_model.eval()
        target = teacher_model.inference(graph, "test")
        pred = student_model.inference(graph=graph)

        target = F.softmax(target.detach()/args.t, dim=1)
        pred = F.softmax(pred, dim=1)
        loss = loss_fn(pred, target)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if epoch % 10 == 0:
            print(loss.item())

    torch.save(student_model, os.path.join("tmp", 'student.pt'))
    acc_list = []
    for batch in range(len(dataset)):
        (rec_graph, ret_graph, now_graph) = dataset[batch]
        if batch == 0:
            acc = calculate_test_acc(rec_graph)
            acc_list.append(acc)
        else:
            acc = calculate_test_acc(now_graph)
            acc_list.append(acc)

    print(acc_list)