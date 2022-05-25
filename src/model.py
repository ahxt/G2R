
import numpy as np
from torch_geometric.nn.conv import MessagePassing
from torch_scatter import scatter_add
import torch_geometric.transforms as T
from torch_geometric.utils import add_remaining_self_loops
from torch.nn import Parameter
import torch.nn.functional as F
import torch.nn as nn
import torch
import math



def glorot(tensor):
    if tensor is not None:
        stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
        tensor.data.uniform_(-stdv, stdv)


def zeros(tensor):
    if tensor is not None:
        tensor.data.fill_(0)


class GCNConv(MessagePassing):

    def __init__(self,
                 in_channels,
                 out_channels,
                 improved=False,
                 cached=False,
                 bias=True,
                 normalize=True,
                 **kwargs):
        super(GCNConv, self).__init__(aggr='add', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached
        self.normalize = normalize

        self.weight = Parameter(torch.Tensor(in_channels, out_channels))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        zeros(self.bias)
        self.cached_result = None
        self.cached_num_edges = None

    @staticmethod
    def norm(edge_index, num_nodes, edge_weight=None, improved=False, dtype=None):
        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype, device=edge_index.device)

        fill_value = 1 if not improved else 2
        edge_index, edge_weight = add_remaining_self_loops(edge_index, edge_weight, fill_value, num_nodes)

        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    def forward(self, x, edge_index, edge_weight=None):
        """"""
        x = torch.matmul(x, self.weight)

        if self.cached and self.cached_result is not None:
            if edge_index.size(1) != self.cached_num_edges:
                raise RuntimeError(
                    'Cached {} number of edges, but found {}. Please '
                    'disable the caching behavior of this layer by removing '
                    'the `cached=True` argument in its constructor.'.format(
                        self.cached_num_edges, edge_index.size(1)))

        if not self.cached or self.cached_result is None:
            self.cached_num_edges = edge_index.size(1)
            if self.normalize:
                edge_index, norm = self.norm(edge_index, x.size(self.node_dim), edge_weight, self.improved, x.dtype)
            else:
                norm = edge_weight
            self.cached_result = edge_index, norm

        edge_index, norm = self.cached_result

        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j if norm is not None else x_j

    def update(self, aggr_out):
        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)


class Encoder(torch.nn.Module):
    def __init__(self,
                 in_channels: int,
                 hidden_channels: int,
                 out_channels: int,
                 activation,
                 base_model=GCNConv,
                 k: int = 2):
        super(Encoder, self).__init__()
        self.base_model = base_model
        self.k = k
        self.conv_1 = self.base_model(in_channels,  hidden_channels)
        self.conv_0 = self.base_model(in_channels,  out_channels)

        for i in range(2, 10):
            exec("self.conv_%s = self.base_model( hidden_channels, hidden_channels )" % i)

        self.conv_last_layer = self.base_model(hidden_channels, out_channels)
        self.conv_layers_list = [self.conv_1, self.conv_2, self.conv_3]
        self.conv_layers_list.append(self.conv_last_layer)
        self.activation = activation
        self.prelu = nn.PReLU(out_channels)
        self.lin0 = nn.Linear(in_channels, hidden_channels, bias=True)
        self.lin1 = nn.Linear(hidden_channels, hidden_channels, bias=True)
        self.lin2 = nn.Linear(hidden_channels, hidden_channels, bias=True)
        self.lin3 = nn.Linear(hidden_channels, out_channels, bias=True)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        if self.k == 0:
            x = self.conv_0( x, edge_index )
            x = F.normalize(x, p=1)
            return x
        for i in range(0, self.k):
            x = self.activation(self.conv_layers_list[i](x, edge_index))
        x = self.conv_last_layer(x, edge_index)
        x = F.normalize(x, p=1)
        return x


class Model(torch.nn.Module):
    def __init__(self,encoder: Encoder):
        super(Model, self).__init__()
        self.encoder: Encoder = encoder

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.encoder(x, edge_index)