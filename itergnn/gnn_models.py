#!/usr/bin/env python
# coding=utf-8

import torch.nn as nn
from torch_geometric.data import Data, Batch
import torch.nn.functional as F
from .classical_layers import cal_size_list, MLP
from .gnn_layers import GNNLayers
from .gnn_aggregation import ReadoutLayers
from .gnn_architecture import GNNArchitectures

class _GNNModels(nn.Module):
    def __init__(self, in_channel, edge_channel, out_channel,
                 num_predictions=1,
                 embedding_layer_num=2,
                 architecture_name='IterGNN', layer_num=10,
                 module_num=1,
                 layer_name='PathGNN', hidden_size=64,
                 input_feat_flag=True, homogeneous_flag=1,
                 readout_name='Max',
                 confidence_layer_num=1,
                 head_layer_num=1,):
        super(_GNNModels, self).__init__()
        self.pointwise_head_layer_flag= False
        if readout_name == 'HeadTail':
            self.pointwise_head_layer_flag = True
            out_channel = int(out_channel/2)
        if homogeneous_flag == 1:
            gnn_homogeneous_flag = True
            other_homogeneous_flag = True
        elif homogeneous_flag == 2:
            gnn_homogeneous_flag = True
            other_homogeneous_flag = False
        elif homogeneous_flag == 0:
            gnn_homogeneous_flag = False
            other_homogeneous_flag = False
        else:
            raise ValueError('Wrong homogeneous_flag as ', homogeneous_flag)
        embedding_size_list = cal_size_list(in_channel, hidden_size, embedding_layer_num)
        self.embedding_module = MLP(embedding_size_list, bias=not other_homogeneous_flag)
        self.readout_module = ReadoutLayers(x_dim=hidden_size, input_x_dim=in_channel,
                                            output_x_dim=hidden_size, input_feat_flag=input_feat_flag,
                                            layer_name=readout_name, homogeneous_flag=gnn_homogeneous_flag)
        confidence_size_list = cal_size_list(hidden_size, 1, confidence_layer_num)
        if readout_name == 'HeadTail' and 'Node' not in architecture_name:
            confidence_size_list = cal_size_list(2*hidden_size, 1, confidence_layer_num)

        self.final_readouts = nn.ModuleList([nn.Linear(out_channel, out_channel) for _ in range(num_predictions)])
        self.num_predictions = num_predictions

        self.module_num = module_num
        confidence_module_list, gnn_layer_list, gnn_module_list = [], [], []
        for _ in range(self.module_num):
            confidence_module = MLP(confidence_size_list, last_activation=nn.Sigmoid, bias=not other_homogeneous_flag)
            gnn_layer = GNNLayers(x_dim=hidden_size, input_x_dim=in_channel,
                                    edge_attr_dim=edge_channel, output_x_dim=hidden_size,
                                    layer_name=layer_name, input_feat_flag=input_feat_flag,
                                    homogeneous_flag=gnn_homogeneous_flag)
            gnn_module = GNNArchitectures(gnn_layer_module=gnn_layer,
                                            readout_module=self.readout_module,
                                            confidence_module=confidence_module,
                                            layer_name=architecture_name,
                                            layer_num=layer_num)
            confidence_module_list.append(confidence_module)
            gnn_layer_list.append(gnn_layer)
            gnn_module_list.append(gnn_module)
        self.confidence_module_list = nn.ModuleList(confidence_module_list)
        self.gnn_layer_list = nn.ModuleList(gnn_layer_list)
        self.gnn_module_list = nn.ModuleList(gnn_module_list)

        head_size_list = cal_size_list(hidden_size, out_channel, head_layer_num)
        self.head_module = MLP(head_size_list, activation=nn.Identity, bias=not other_homogeneous_flag)
    def readout(self, data):
        raise NotImplementedError
        # return self.readout_module(data)
    def forward(self, data, output_node_feat_flag=False,
                output_layer_num_flag=False,
                output_residual_confidence_flag=False):
        kwargs = {k:v for k,v in data.__dict__.items()}
        kwargs = kwargs['_store']
        kwargs['input_x'] = x = kwargs['x']
        #kwargs.pop('x')
        del kwargs['x']

        x = self.embedding_module(x)
        layer_num = 0
        x_list = []
        residual_confidence_list = []
        for gnn_module in self.gnn_module_list:
            if 'ACT' in gnn_module.__class__.__name__:
                x, cur_layer_num, cur_residual_confidence = gnn_module(Batch(x=x, **kwargs))
                residual_confidence_list.append(cur_residual_confidence)
            else:
                x, cur_layer_num = gnn_module(Batch(x=x, **kwargs))
            layer_num += cur_layer_num
            x_list.append(x)
        if len(residual_confidence_list):
            residual_confidence = torch.sum(torch.stack(residual_confidence_list, dim=0), dim=0)
        if self.pointwise_head_layer_flag:
            x_list = [self.head_module(x) for x in x_list]
        global_feat = self.readout([Batch(x=x, **kwargs) for x in x_list])
        # To avoid information-leak between nodes, we perform pointwise head-module for the physical simulation task
        if not self.pointwise_head_layer_flag:
            out = self.head_module(global_feat)
        else:
            out = global_feat

        output = (out,)
        if output_node_feat_flag:
            output = output + (x, )
        if output_layer_num_flag:
            output = output + (layer_num, )
        if output_residual_confidence_flag:
            output = output + (residual_confidence, )
        preds = [F.log_softmax(self.final_readouts[p](output[0]), dim=-1) for p in range(self.num_predictions)]
        return preds, layer_num  # return F.log_softmax(output[0], dim=-1), 0
    @property
    def tao(self,):
        return self.gnn_module_list[0].tao

class GraphGNNModels(_GNNModels):
    def readout(self, data_list):
        return self.readout_module(data_list[-1])

class NodeGNNModels(_GNNModels):
    def readout(self, data_list):
        return data_list[-1].x

    def reset_parameters(self):
        pass

class JKGraphGNNModels(_GNNModels):
    def __init__(self, in_channel, edge_channel, out_channel,
                 hidden_size=64, homogeneous_flag=1,
                 head_layer_num=1, **kwargs):
        super(JKGraphGNNModels, self).__init__(in_channel, edge_channel, out_channel,
                                               hidden_size = hidden_size,
                                               homogeneous_flag = homogeneous_flag,
                                               head_layer_num=head_layer_num,
                                               **kwargs)
        if homogeneous_flag == 1:
            other_homogeneous_flag = True
        elif homogeneous_flag in [0,2]:
            other_homogeneous_flag = False
        else:
            raise ValueError('Wrong homogeneous_flag as ', homogeneous_flag)
        head_size_list = cal_size_list(int(hidden_size*self.module_num), out_channel, head_layer_num)
        self.head_module = MLP(head_size_list, activation=nn.Identity, bias=not other_homogeneous_flag)
    def readout(self, data_list):
        return torch.cat([self.readout_module(data) for data in data_list], dim=-1)

