from typing import Union, Tuple
from torch_geometric.typing import OptPairTensor, Adj, Size

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Parameter
import torch.nn.functional as F
from torch_sparse import SparseTensor, matmul
from torch_geometric.nn.conv import MessagePassing


class SAGEConv(MessagePassing):
    def __init__(self, in_channels: Union[int, Tuple[int, int]],
                 out_channels: int, normalize: bool = False,
                 root_weight: bool = True,
                 bias: bool = True, **kwargs):  # yapf: disable
        kwargs.setdefault('aggr', 'mean')
        super(SAGEConv, self).__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize
        self.root_weight = root_weight

        if isinstance(in_channels, int):
            self.in_channels = [in_channels, in_channels]

        self.l_weights_list = nn.ParameterList()
        self.l_bias_list = nn.ParameterList()

        self.l_weights_list.append(nn.Parameter(Tensor(self.in_channels[0], out_channels)))
        self.l_bias_list.append(nn.Parameter(Tensor(1, out_channels)))

        self.expand_l_weights = None
        self.expand_l_bias = None
        if self.root_weight:
            self.r_weights_list = nn.ParameterList()
            self.r_weights_list.append(nn.Parameter(Tensor(self.in_channels[1], out_channels)))
            self.expand_r_weights = None

        self.init_parameters()

    def combine(self):
        if self.expand_l_weights is not None:
            self.l_weights_list.append(self.expand_l_weights)
            self.l_bias_list.append(self.expand_l_bias)
            if self.root_weight:
                self.r_weights_list.append(self.expand_r_weights)

            self.expand_l_weights = None
            self.expand_l_bias = None
            if self.root_weight:
                self.expand_r_weights = None
            # self.isolate_parameters()

    def isolate_parameters(self):
        for index, item in enumerate(self.l_weights_list):
            item.requires_grad = False
        for index, item in enumerate(self.l_bias_list):
            item.requires_grad = False
        if self.root_weight:
            for index, item in enumerate(self.r_weights_list):
                item.requires_grad = False

    def open_parameters(self):
        for index, item in enumerate(self.l_weights_list):
            item.requires_grad = True
        for index, item in enumerate(self.l_bias_list):
            item.requires_grad = True
        if self.root_weight:
            for index, item in enumerate(self.r_weights_list):
                item.requires_grad = True

    def expand(self, e_in, e_out):
        if self.expand_l_weights is not None:
            print(f'expand weights should be saved first!')
            exit(0)

        if e_in > 0:
            self.in_channels[0] = self.in_channels[0] + e_in
            if self.root_weight:
                self.in_channels[1] = self.in_channels[1] + e_in
            else:
                self.in_channels[1] = self.in_channels[0] + e_in

        self.out_channels = self.out_channels + e_out

        self.expand_l_weights = nn.Parameter(torch.randn(size=(self.in_channels[0], e_out)), requires_grad=True)
        self.expand_l_bias = nn.Parameter(torch.randn(size=(1, e_out)), requires_grad=True)
        if self.root_weight:
            self.expand_r_weights = nn.Parameter(torch.randn(size=(self.in_channels[1], e_out)), requires_grad=True)
        # self.init_expanded_parameters()

    def init_parameters(self):
        for param in self.parameters():
            nn.init.xavier_uniform_(param)

    def init_expanded_parameters(self):
        if self.expand_l_weights is not None:
            nn.init.xavier_uniform_(self.expand_l_weights)
            nn.init.xavier_uniform_(self.expand_l_bias)
            if self.root_weight:
                nn.init.xavier_uniform_(self.expand_r_weights)

    def old_reduce(self, f, x_r):
        f_out = None
        x_r_out = None

        for i in range(len(self.l_weights_list)):
            l_weight = self.l_weights_list[i]
            l_bias = self.l_bias_list[i]

            size = l_weight.shape[0]
            if self.in_channels[0] - size > 0:
                split_list = [size, self.in_channels[0] - size]
                split_f = torch.split(f, split_size_or_sections=split_list, dim=1)[0]
            else:
                split_f = f

            if f_out is None:
                f_out = torch.matmul(split_f, l_weight) + l_bias
            else:
                f_out = torch.cat([f_out, torch.matmul(split_f, l_weight) + l_bias], dim=1)

        if self.root_weight and x_r is not None:
            for i in range(len(self.r_weights_list)):
                r_weight = self.r_weights_list[i]

                size = r_weight.shape[0]
                if self.in_channels[1] - size > 0:
                    split_list = [size, self.in_channels[1] - size]
                    split_x_r = torch.split(x_r, split_size_or_sections=split_list, dim=1)[0]
                else:
                    split_x_r = x_r

                if x_r_out is None:
                    x_r_out = torch.matmul(split_x_r, r_weight)
                else:
                    x_r_out = torch.cat([x_r_out, torch.matmul(split_x_r, r_weight)], dim=1)

        return f_out + x_r_out

    def new_reduce(self, f, x_r):
        f = torch.matmul(f, self.expand_l_weights)
        f = f + self.expand_l_bias

        if self.root_weight and x_r is not None:
            f += torch.matmul(x_r, self.expand_r_weights)

        return f

    def forward(self, phase, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                size: Size = None) -> Tuple:
        """"""
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        # propagate_type: (x: OptPairTensor)
        out = self.propagate(edge_index, x=x, size=size)

        old_out = self.old_reduce(out, x[1])
        if self.normalize:
            old_out = F.normalize(old_out, p=2., dim=-1)

        new_out = None
        if self.expand_l_weights is not None and phase == 'retrain':
            new_out = self.new_reduce(out, x[1])
            if self.normalize:
                new_out = F.normalize(new_out, p=2., dim=-1)
            out = torch.cat([old_out, new_out], dim=1)
        else:
            out = old_out

        # if self.normalize:
        #     out = F.normalize(out, p=2., dim=-1)

        return out, old_out, new_out

    def message(self, x_j: Tensor) -> Tensor:
        return x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: OptPairTensor) -> Tensor:
        adj_t = adj_t.set_value(None, layout=None)
        return matmul(adj_t, x[0], reduce=self.aggr)

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)
