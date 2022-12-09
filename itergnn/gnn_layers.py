#!/usr/bin/env python
# coding=utf-8

import torch
import torch.nn as nn

from torch_geometric.nn import GCNConv as _GCNConv, GATConv as _GATConv
from torch_geometric.nn.conv import GINConv as _GINConv
from torch_geometric.nn import MessagePassing as _MessagePassing

from .classical_layers import cal_size_list, MLP
from .gnn_aggregation import AggregationLayers

class _GNNLayer(nn.Module):
    def __init__(self, input_feat_flag=False, *args, **kwargs):
        super(_GNNLayer, self).__init__()
        self.input_feat_flag = input_feat_flag
    def cal_node_feat(self, data):
        if self.input_feat_flag:
            return torch.cat([data.input_x, data.x], dim=-1)
        else:
            return data.x
    def forward(self, data):
        raise NotImplementedError

class MPNNConv(_MessagePassing):
    def __init__(self, in_channel, out_channel, edge_channel,
                 homogeneous_flag=False, edge_embedding_layer_num=2,
                 update_layer_num=0,):
        super(MPNNConv, self).__init__(aggr='add')
        self.build_model(in_channel, out_channel, edge_channel,
                         homogeneous_flag, edge_embedding_layer_num,
                         update_layer_num)
    def build_model(self, in_channel, out_channel, edge_channel,
                    homogeneous_flag=False, edge_embedding_layer_num=2,
                    update_layer_num=0,):
        if edge_embedding_layer_num == 0:
            self.edge_embedding_module = nn.Identity()
            mid_channel = in_channel*2+edge_channel
        else:
            edge_embedding_size_list = cal_size_list(in_channel*2+edge_channel, out_channel, edge_embedding_layer_num)
            self.edge_embedding_module = MLP(edge_embedding_size_list, bias=not homogeneous_flag)
            mid_channel = out_channel
        if update_layer_num == 0:
            self.update_module = nn.Identity()
        else:
            update_size_list = cal_size_list(mid_channel, out_channel, update_layer_num)
            self.update_module = MLP(update_size_list, bias=not homogeneous_flag)
            mid_channel = out_channel
        assert(mid_channel == out_channel)
    def forward(self,x, edge_index, edge_attr):
        return self.propagate(edge_index, size=(x.size(0),x.size(0)), x=x, edge_attr=edge_attr)
    def message(self, x_i, x_j, edge_attr):
        edge_feat = torch.cat([x_i,x_j,edge_attr], dim=-1)
        return self.edge_embedding_module(edge_feat)
    def update(self, aggr_out):
        return self.update_module(aggr_out)

class MPNNMaxConv(MPNNConv):
    def __init__(self, in_channel, out_channel, edge_channel,
                 homogeneous_flag=False, edge_embedding_layer_num=2,
                 update_layer_num=0,):
        super(MPNNConv, self).__init__(aggr='max')
        self.build_model(in_channel, out_channel, edge_channel,
                         homogeneous_flag, edge_embedding_layer_num,
                         update_layer_num)

class PathConv(nn.Module):
    def __init__(self, in_channel, out_channel, edge_channel,
                 homogeneous_flag=False, edge_embedding_layer_num=2,):
        super(PathConv, self).__init__()
        self.homogeneous_flag = homogeneous_flag
        assert(edge_embedding_layer_num > 0 or out_channel==in_channel*2+edge_channel)
        edge_embedding_size_list = cal_size_list(in_channel*2+edge_channel, out_channel, edge_embedding_layer_num)
        self.edge_embedding_module = MLP(edge_embedding_size_list, bias=not self.homogeneous_flag)
        self.aggregation_module = AggregationLayers(layer_name='Max',
                                                    key_dim=in_channel*2+edge_channel,
                                                    value_dim=out_channel,
                                                    query_dim=0,
                                                    embedding_layer_num=0,
                                                    output_dim=None,
                                                    homogeneous_flag=self.homogeneous_flag)
    def forward(self,x, edge_index, edge_attr):
        num_nodes = x.size(0)
        rol, col = edge_index

        edge_attr_values = torch.cat([x[rol], x[col], edge_attr], dim=-1)
        edge_attr_keys = torch.cat([x[rol], x[col], edge_attr], dim=-1)
        edge_attr_values = self.edge_embedding_module(edge_attr_values)
        x_cand = self.aggregation_module(keys=edge_attr_keys,
                                         values=edge_attr_values,
                                         query=torch.zeros_like(edge_attr_values[0,0:0]),
                                         index=col, size=num_nodes)
        # kind of ugly (maybe due to the imperfect decomposition)....
        # Here, x[:, -out_channel:] corresponds to the original node_feat without input_x
        # The behavior should be consistent with _GNNLayer.cal_node_feat
        x = torch.max(x_cand, x[:,-x_cand.size(-1):])
        return x

class PathSimConv(PathConv):
    def __init__(self, in_channel, out_channel, edge_channel,
                 homogeneous_flag=False, edge_embedding_layer_num=2,):
        super(PathSimConv, self).__init__(in_channel, out_channel, edge_channel,
                                          homogeneous_flag, edge_embedding_layer_num)
        assert(edge_embedding_layer_num > 0 or out_channel==in_channel+edge_channel)
        edge_embedding_size_list = cal_size_list(in_channel+edge_channel, out_channel, edge_embedding_layer_num)
        self.edge_embedding_module = MLP(edge_embedding_size_list, bias=not self.homogeneous_flag)
    def forward(self,x, edge_index, edge_attr):
        num_nodes = x.size(0)
        rol, col = edge_index

        edge_attr_values = torch.cat([x[rol], edge_attr], dim=-1)
        edge_attr_keys = torch.cat([x[rol], x[col], edge_attr], dim=-1)
        edge_attr_values = self.edge_embedding_module(edge_attr_values)
        x_cand = self.aggregation_module(keys=edge_attr_keys,
                                         values=edge_attr_values,
                                         query=torch.zeros_like(edge_attr_values[0,0:0]),
                                         index=col, size=num_nodes)
        # kind of ugly (maybe due to the imperfect decomposition)....
        # Here, x[:, -out_channel:] corresponds to the original node_feat without input _x
        # The behavior should be consistent with _GNNLayer.cal_node_feat
        x = torch.max(x_cand, x[:,-x_cand.size(-1):])
        return x

class PathGNNLayers(_GNNLayer):
    def __init__(self, layer_name='MPNNMaxConv', x_dim=None, input_x_dim=None,
                 output_x_dim=None, edge_attr_dim=None, input_feat_flag=False,
                 homogeneous_flag=False, *args, **kwargs):
        if x_dim is None or (input_x_dim is None and input_feat_flag) or edge_attr_dim is None or output_x_dim is None:
            raise ValueError('Concrete dimensionality required for x', x_dim,
                             'input_x', input_x_dim, 'output_x', output_x_dim,)
        super(PathGNNLayers, self).__init__(input_feat_flag=input_feat_flag, *args, **kwargs)
        self.input_feat_flag, self.homogeneous_flag = input_feat_flag, homogeneous_flag
        print(layer_name)
        self.gnn_module = PathConv(x_dim + (input_x_dim if input_feat_flag else 0),
        #self.gnn_module = globals()[layer_name](x_dim + (input_x_dim if input_feat_flag else 0),
                                                output_x_dim, edge_attr_dim,
                                                homogeneous_flag=homogeneous_flag)
    def forward(self, data):
        x, edge_index, edge_attr = self.cal_node_feat(data), data.edge_index, data.edge_attr
        x = self.gnn_module(x, edge_index, edge_attr)
        return x

class GCNConv(nn.Module):
    def __init__(self, in_channel, out_channel, homogeneous_flag=False):
        super(GCNConv, self).__init__()
        self.module = _GCNConv(in_channel, out_channel, bias=not homogeneous_flag)
    def forward(self, x, edge_index):
        return self.module(x, edge_index)

class GATConv(nn.Module):
    def __init__(self, in_channel, out_channel, homogeneous_flag=False):
        super(GATConv, self).__init__()
        if not homogeneous_flag:
            self.module = _GATConv(in_channel, out_channel)
        else:
            self.module = _HomoGATConv(in_channel, out_channel, bias=False)
    def forward(self, x, edge_index):
        return self.module(x, edge_index)

class GINConv(nn.Module):
    def __init__(self, in_channel, out_channel, homogeneous_flag=False):
        super(GINConv, self).__init__()
        size_list = cal_size_list(in_channel, out_channel, 2)
        self.module = _GINConv(MLP(size_list,bias=not homogeneous_flag))
    def forward(self, x, edge_index):
        return self.module(x, edge_index)

class EpsGINConv(nn.Module):
    def __init__(self, in_channel, out_channel, homogeneous_flag=False):
        super(EpsGINConv, self).__init__()
        size_list = cal_size_list(in_channel, out_channel, 2)
        self.module = _GINConv(MLP(size_list,bias=not homogeneous_flag), train_eps=True)
    def forward(self, x, edge_index):
        return self.module(x, edge_index)

class ClassicalGNNLayers(_GNNLayer):
    def __init__(self, layer_name='GCNConv', x_dim=None, input_x_dim=None,
                 output_x_dim=None, input_feat_flag=False, homogeneous_flag=False,
                 *args, **kwargs):
        if x_dim is None or (input_x_dim is None and input_feat_flag) or output_x_dim is None:
            raise ValueError('Concrete dimensionality required for x', x_dim,
                             'input_x', input_x_dim, 'output_x', output_x_dim,)
        super(ClassicalGNNLayers, self).__init__(*args, **kwargs)
        self.input_feat_flag, self.homogeneous_flag = input_feat_flag, homogeneous_flag
        self.gnn_module = globals()[layer_name](x_dim + (input_x_dim if input_feat_flag else 0),
                                                output_x_dim, homogeneous_flag=homogeneous_flag)
    def forward(self, data):
        x, edge_index = self.cal_node_feat(data), data.edge_index
        assert(not torch.sum(torch.isnan(x)))
        x = self.gnn_module(x, edge_index)
        assert(not torch.sum(torch.isnan(x)))
        return x

class GNNLayers(nn.Module):
    def __init__(self, layer_name, *args, **kwargs):
        super(GNNLayers, self).__init__()
        if layer_name in ['GCNConv', 'GATConv', 'GINConv', 'EpsGINConv']:
            self.module = ClassicalGNNLayers(layer_name=layer_name, *args, **kwargs)
        else:
            self.module = PathGNNLayers(layer_name=layer_name, *args, **kwargs)
    @property
    def homogeneous_flag(self,):
        return self.module.homogeneous_flag
    def forward(self, data):
        return self.module(data)

#=======HomoGATConv=============
from .utils import homo_gnn_softmax
import torch.nn.functional as F

class _HomoGATConv(_GATConv):
    """
    Configed GATConv from torch_geometric.nn (1.3.2)
    Single change: softmax --> homo_gnn_softmax
    """
    def message(self, edge_index_i, x_i, x_j, size_i):
        # Compute attention coefficients.
        assert(not torch.sum(torch.isnan(x_i)))
        assert(not torch.sum(torch.isnan(x_j)))
        assert(not torch.sum(torch.isnan(edge_index_i)))
        x_j = x_j.view(-1, self.heads, self.out_channels)
        if x_i is None:
            alpha = (x_j * self.att[:, :, self.out_channels:]).sum(dim=-1)
        else:
            x_i = x_i.view(-1, self.heads, self.out_channels)
            alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)

        assert(not torch.sum(torch.isnan(alpha)))
        alpha = F.leaky_relu(alpha, self.negative_slope)
        assert(not torch.sum(torch.isnan(alpha)))
        alpha = homo_gnn_softmax(alpha, edge_index_i, size_i)
        assert(not torch.sum(torch.isnan(alpha)))

        # Sample attention coefficients stochastically.
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        return x_j * alpha.view(-1, self.heads, 1)