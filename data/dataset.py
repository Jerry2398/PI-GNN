import os
import torch
import random
import numpy as np

from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph
from torch.utils.data import Dataset
from torch_geometric.utils import to_undirected
from .config import DATA_CONFIG


class DyDataset(Dataset):
    def __init__(self, dataset_name='dblp', edge_type='stream_edges', m_size=128):
        self.dataset_name = dataset_name
        self.edge_type = edge_type
        config = DATA_CONFIG[dataset_name]
        self.basic_t = config['basic_t']
        self.hop_num = config['hop_num']
        self.task_nums = config['task_nums']
        self.datapath = os.path.join('data', dataset_name)
        self.m_size = m_size

        self.datalist = os.listdir(os.path.join(self.datapath, edge_type))
        self.features = torch.tensor(np.loadtxt(os.path.join(self.datapath, 'attributes')), dtype=torch.float)

        self.labeled_nodes = np.loadtxt(os.path.join(self.datapath, 'labels'))[:, 0]
        self.label_nodes_dict = {self.labeled_nodes[i]: i for i in range(len(self.labeled_nodes))}

        self.labels = torch.tensor(np.loadtxt(os.path.join(self.datapath, 'labels'))[:, 1], dtype=torch.long)

        self.train_nodes_set = set(np.loadtxt(os.path.join(self.datapath, 'train_nodes')))
        self.test_nodes_set = set(np.loadtxt(os.path.join(self.datapath, 'test_nodes')))
        self.val_nodes_set = set(np.loadtxt(os.path.join(self.datapath, 'val_nodes')))

        self.basic_edges, self.basic_nodes_set = self.load_basic_edges()
        self.labels_num = torch.max(self.labels) + 1
        self.in_dim = self.features.shape[1]
        self.memory = list()

    def __getitem__(self, index):
        if index == 0:
            edges = self.basic_edges
            reverse_edges = edges[:, [1, 0]]
            edges = torch.tensor(np.concatenate((edges, reverse_edges), axis=0).transpose(), dtype=torch.long)

            nodes, edge_index, _, _ = k_hop_subgraph(
                node_idx=torch.tensor(list(self.basic_nodes_set), dtype=torch.long),
                num_hops=self.hop_num, edge_index=edges,
                relabel_nodes=True)

            edge_index = to_undirected(edge_index)
            features = self.features[nodes]

            label_idx = [self.label_nodes_dict[item.item()] if item.item() in self.label_nodes_dict else 0 for item in nodes]
            labels = self.labels[label_idx]

            train_set = self.basic_nodes_set.intersection(self.train_nodes_set)
            test_set = self.basic_nodes_set.intersection(self.test_nodes_set)
            val_set = self.basic_nodes_set.intersection(self.val_nodes_set)

            train_mask = np.array([nodes[i].item() in train_set for i in range(len(nodes))],
                                    dtype=np.bool)
            test_mask = np.array([nodes[i].item() in test_set for i in range(len(nodes))],
                                    dtype=np.bool)
            val_mask = np.array([nodes[i].item() in val_set for i in range(len(nodes))],
                                    dtype=np.bool)

            train_mask = torch.tensor(np.argwhere(train_mask).flatten().tolist(), dtype=torch.long)
            test_mask = torch.tensor(np.argwhere(test_mask).flatten().tolist(), dtype=torch.long)
            val_mask = torch.tensor(np.argwhere(val_mask).flatten().tolist(), dtype=torch.long)

            graph = Data(x=features, edge_index=edge_index, y=labels)
            graph.train_mask = train_mask
            graph.test_mask = test_mask
            graph.val_mask = val_mask
            graph.new_nodes_test_mask = test_mask
            self._update_memory(nodes=list(train_set))
            return graph, None, None
        else:
            end_t = self.basic_t + index
            last_edges = self.basic_edges
            for i in range(self.basic_t, end_t - 1):
                last_edges = np.concatenate(
                    (last_edges, np.loadtxt(os.path.join(self.datapath, self.edge_type, str(i)))), axis=0)
            last_nodes_set = set(last_edges.flatten().tolist())

            reverse_last_edges = last_edges[:, [1, 0]]
            last_edges = torch.tensor(np.concatenate((last_edges, reverse_last_edges), axis=0).transpose(),
                                      dtype=torch.long)

            new_edges = np.loadtxt(os.path.join(self.datapath, self.edge_type, str(end_t - 1)))

            affected_nodes_set = set(new_edges.flatten().tolist())

            last_affected_nodes_set = last_nodes_set.intersection(affected_nodes_set)
            now_nodes_set = last_nodes_set.union(affected_nodes_set)

            memory_nodes_set = set(self.memory)

            reverse_new_edges = new_edges[:, [1, 0]]
            new_edges = torch.tensor(np.concatenate((new_edges, reverse_new_edges), axis=0).transpose(),
                                     dtype=torch.long)

            now_edges = torch.cat((last_edges, new_edges), dim=1)

            rec_nodes_set = memory_nodes_set.union(last_affected_nodes_set)
            rec_nodes, rec_edge_index, _, _ = k_hop_subgraph(
                node_idx=torch.tensor(list(rec_nodes_set), dtype=torch.long), num_hops=self.hop_num,
                edge_index=last_edges, relabel_nodes=True)

            rec_edge_index = to_undirected(rec_edge_index)
            rec_x = self.features[rec_nodes]

            label_idx = [self.label_nodes_dict[item.item()] if item.item() in self.label_nodes_dict else 0 for item in rec_nodes]
            rec_y = self.labels[label_idx]

            memory_train_set = memory_nodes_set.intersection(self.train_nodes_set)
            last_affected_train_set = last_affected_nodes_set.intersection(self.train_nodes_set)

            memory_train_mask = np.array([rec_nodes[i].item() in memory_train_set for i in range(len(rec_nodes))], dtype=np.bool)
            last_affected_train_mask = np.array([rec_nodes[i].item() in last_affected_train_set for i in range(len(rec_nodes))], dtype=np.bool)

            memory_train_mask = torch.tensor(np.argwhere(memory_train_mask).flatten().tolist(), dtype=torch.long)
            last_affected_train_mask = torch.tensor(np.argwhere(last_affected_train_mask).flatten().tolist(), dtype=torch.long)

            rec_data = Data(x=rec_x, edge_index=rec_edge_index, y=rec_y)
            rec_data.memory_train_mask = memory_train_mask
            rec_data.last_affected_train_mask = last_affected_train_mask

            ret_nodes_set = memory_nodes_set.union(affected_nodes_set)
            ret_nodes, ret_edge_index, _, _ = k_hop_subgraph(
                node_idx=torch.tensor(list(ret_nodes_set), dtype=torch.long), num_hops=self.hop_num,
                edge_index=now_edges, relabel_nodes=True)

            ret_edge_index = to_undirected(ret_edge_index)
            ret_x = self.features[ret_nodes]

            label_idx = [self.label_nodes_dict[item.item()] if item.item() in self.label_nodes_dict else 0 for item in ret_nodes]

            ret_y = self.labels[label_idx]

            affected_train_set = ret_nodes_set.intersection(self.train_nodes_set)
            affected_test_set = ret_nodes_set.intersection(self.test_nodes_set)
            affected_val_set = ret_nodes_set.intersection(self.val_nodes_set)

            memory_train_mask = np.array([ret_nodes[i].item() in memory_train_set for i in range(len(ret_nodes))], dtype=np.bool)
            affected_train_mask = np.array(
                [ret_nodes[i].item() in affected_train_set for i in range(len(ret_nodes))],
                dtype=np.bool)
            affected_test_mask = np.array([ret_nodes[i].item() in affected_test_set for i in range(len(ret_nodes))],
                                     dtype=np.bool)
            affected_val_mask = np.array([ret_nodes[i].item() in affected_val_set for i in range(len(ret_nodes))],
                                    dtype=np.bool)

            memory_train_mask = torch.tensor(np.argwhere(memory_train_mask).flatten().tolist(), dtype=torch.long)
            affected_train_mask = torch.tensor(np.argwhere(affected_train_mask).flatten().tolist(), dtype=torch.long)
            affected_test_mask = torch.tensor(np.argwhere(affected_test_mask).flatten().tolist(), dtype=torch.long)
            affected_val_mask = torch.tensor(np.argwhere(affected_val_mask).flatten().tolist(), dtype=torch.long)

            ret_data = Data(x=ret_x, edge_index=ret_edge_index, y=ret_y)
            ret_data.memory_train_mask = memory_train_mask
            ret_data.affected_train_mask = affected_train_mask
            ret_data.affeted_test_mask = affected_test_mask
            ret_data.affected_val_mask = affected_val_mask

            now_nodes, now_edge_index, _, _ = k_hop_subgraph(
                node_idx=torch.tensor(list(now_nodes_set), dtype=torch.long), num_hops=self.hop_num,  # self.hop_num 4
                edge_index=now_edges, relabel_nodes=True)

            now_edge_index = to_undirected(now_edge_index)
            now_x = self.features[now_nodes]

            label_idx = [self.label_nodes_dict[item.item()] if item.item() in self.label_nodes_dict else 0 for item in now_nodes]
            now_y = self.labels[label_idx]

            train_set = now_nodes_set.intersection(self.train_nodes_set)
            test_set = now_nodes_set.intersection(self.test_nodes_set)
            val_set = now_nodes_set.intersection(self.val_nodes_set)
            new_nodes_test_set = affected_nodes_set.intersection(self.test_nodes_set)

            train_mask = torch.tensor([now_nodes[i].item() in train_set for i in range(len(now_nodes))],
                                      dtype=torch.bool)
            test_mask = torch.tensor([now_nodes[i].item() in test_set for i in range(len(now_nodes))],
                                     dtype=torch.bool)
            val_mask = torch.tensor([now_nodes[i].item() in val_set for i in range(len(now_nodes))],
                                    dtype=torch.bool)

            new_nodes_test_mask = np.array([now_nodes[i].item() in new_nodes_test_set for i in range(len(now_nodes))],
                                           dtype=np.bool)

            new_nodes_test_mask = torch.tensor(np.argwhere(new_nodes_test_mask).flatten().tolist(), dtype=torch.long)

            now_data = Data(x=now_x, edge_index=now_edge_index, y=now_y)
            now_data.train_mask = train_mask
            now_data.test_mask = test_mask
            now_data.val_mask = val_mask
            now_data.new_nodes_test_mask = new_nodes_test_mask
            self._update_memory(list(train_set))
            return rec_data, ret_data, now_data

    def load_basic_edges(self):
        edges = None
        for i in range(self.basic_t):
            if i == 0:
                edges = np.loadtxt(os.path.join(self.datapath, self.edge_type, str(i)))
            else:
                edges = np.concatenate((edges, np.loadtxt(os.path.join(self.datapath, self.edge_type, str(i)))), axis=0)
        basic_nodes_set = set(edges.flatten().tolist())
        return edges, basic_nodes_set

    def _update_memory(self, nodes):
        if len(nodes) <= self.m_size:
            self.memory = nodes
        else:
            self.memory = random.sample(nodes, self.m_size)

    def __len__(self):
        return self.task_nums

