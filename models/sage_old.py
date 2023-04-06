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
    r"""The GraphSAGE operator from the `"Inductive Representation Learning on
    Large Graphs" <https://arxiv.org/abs/1706.02216>`_ paper

    .. math::
        \mathbf{x}^{\prime}_i = \mathbf{W}_1 \mathbf{x}_i + \mathbf{W}_2 \cdot
        \mathrm{mean}_{j \in \mathcal{N(i)}} \mathbf{x}_j

    Args:
        in_channels (int or tuple): Size of each input sample. A tuple
            corresponds to the sizes of source and target dimensionalities.
        out_channels (int): Size of each output sample.
        normalize (bool, optional): If set to :obj:`True`, output features
            will be :math:`\ell_2`-normalized, *i.e.*,
            :math:`\frac{\mathbf{x}^{\prime}_i}
            {\| \mathbf{x}^{\prime}_i \|_2}`.
            (default: :obj:`False`)
        root_weight (bool, optional): If set to :obj:`False`, the layer will
            not add transformed root node features to the output.
            (default: :obj:`True`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """

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

        self.l_weights = nn.Parameter(Tensor(self.in_channels[0], out_channels))
        self.l_bias = nn.Parameter(Tensor(1, out_channels))
        self.expand_l_weights = None
        self.expand_l_bias = None
        if self.root_weight:
            self.r_weights = nn.Parameter(Tensor(self.in_channels[1], out_channels))
            self.expand_r_weights = None

        self.init_parameters()

    # 整个流程是：扩张->隔离->训练新网络->合并网络->隔离

    def combine(self):  #合并参数
        if self.expand_l_weights is not None:
            self.l_weights = Parameter(torch.cat([self.l_weights, self.expand_l_weights], dim=1))
            self.l_bias = Parameter(torch.cat([self.l_bias, self.expand_l_bias], dim=1))
            if self.root_weight:
                self.r_weights = Parameter(torch.cat([self.r_weights, self.expand_r_weights], dim=1))

    def isolate_parameters(self):
        self.l_weights.requires_grad = True
        self.l_bias.requires_grad = True
        if self.root_weight:
            self.r_weights.requires_grad = True

    def open_parameters(self):   # 隔离已经训练好的参数
        self.l_weights.requires_grad = False
        self.l_bias.requires_grad = False
        if self.root_weight:
            self.r_weights.requires_grad = False

    def expand(self, e_in, e_out):  # 扩张网络，其实就是扩张原来的weights 和 增加的weights
        if e_in > 0:
            self.l_weights = Parameter(torch.cat([self.l_weights, torch.zeros(size=(e_in, self.l_weights.shape[1])).detach()], dim=0))
            self.in_channels[0] = self.l_weights.shape[0]
            if self.root_weight:
                self.r_weights = Parameter(torch.cat([self.r_weights, torch.zeros(size=(e_in, self.r_weights.shape[1])).detach()], dim=0))
                self.in_channels[1] = self.r_weights.shape[0]
            else:
                self.in_channels[1] = self.l_weights.shape[0]
        self.out_channels = self.l_weights.shape[1] + e_out

        self.expand_l_weights = nn.Parameter(torch.zeros(size=(self.in_channels[0], e_out)))
        self.expand_l_bias = nn.Parameter(torch.zeros(size=(1, e_out)))
        if self.root_weight:
            self.expand_r_weights = nn.Parameter(torch.zeros(size=(self.in_channels[1], e_out)))
        # self.init_expand_parameters()

    def init_parameters(self):
        for param in self.parameters():
            nn.init.xavier_uniform_(param)

    # def init_expand_parameters(self):
    #     nn.init.xavier_uniform_(self.expand_l_weights)
    #     nn.init.xavier_uniform_(self.expand_l_bias)
    #     if self.root_weight:
    #         nn.init.xavier_uniform_(self.expand_r_weights)

    def init_expanded_parameters(self):
        if self.expand_l_weights is not None:
            nn.init.xavier_uniform_(self.expand_l_weights)
            nn.init.xavier_uniform_(self.expand_l_bias)
            if self.root_weight:
                nn.init.xavier_uniform_(self.expand_r_weights)

    def old_reduce(self, f, x_r):
        f = torch.matmul(f, self.l_weights)
        f = f + self.l_bias

        if self.root_weight and x_r is not None:
            f += torch.matmul(x_r, self.r_weights)

        return f

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
        new_out = None
        if self.expand_l_weights is not None and phase == 'retrain':
            new_out = self.new_reduce(out, x[1])
            out = torch.cat([old_out, new_out], dim=1)
        else:
            out = old_out

        if self.normalize:
            out = F.normalize(out, p=2., dim=-1)

        return out, old_out, new_out   # 拼接之后的输出， 旧的网络的输出， 新的网络的输出部分,但是归一化还得商榷，尤其是新网络的归一化部分

    def message(self, x_j: Tensor) -> Tensor:
        return x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: OptPairTensor) -> Tensor:
        adj_t = adj_t.set_value(None, layout=None)
        return matmul(adj_t, x[0], reduce=self.aggr)

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)
