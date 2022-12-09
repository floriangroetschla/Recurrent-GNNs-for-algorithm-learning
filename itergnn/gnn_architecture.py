#!/usr/bin/env python
# coding=utf-8

import torch
import torch.nn as nn
from torch_geometric.data import Data, Batch

class DeepGNN(nn.Module):
    def __init__(self, gnn_layer_module=None, layer_num=1, *args, **kwargs):
        assert(gnn_layer_module is not None)
        super(DeepGNN, self).__init__()
        self.layers = nn.ModuleList([copy.deepcopy(gnn_layer_module) for _ in range(layer_num)])
    def forward(self, data):
        kwargs = {k:v for k,v in data.__dict__.items()}
        for l in self.layers:
            data = Batch(**kwargs)
            kwargs['x'] = l(data)
            assert(not torch.sum(torch.isnan(kwargs['x'])))
        return kwargs['x'], len(self.layers)

class SharedDeepGNN(nn.Module):
    def __init__(self, gnn_layer_module=None, layer_num=1, *args, **kwargs):
        assert(gnn_layer_module is not None)
        super(SharedDeepGNN, self).__init__()
        self.gnn_layer_module = gnn_layer_module
        self.layer_num = layer_num
    def forward(self, data):
        kwargs = {k:v for k,v in data.__dict__.items()}
        for _ in range(self.layer_num):
            data = Batch(**kwargs)
            kwargs['x'] = self.gnn_layer_module(data)
            assert(not torch.sum(torch.isnan(kwargs['x'])))
        return kwargs['x'], self.layer_num

# Adaptive Computational Time variants
class ACTIterGNN(nn.Module):
    def __init__(self, tao, gnn_layer_module=None, readout_module=None, confidence_module=None,
                 layer_num=1, *args, **kwargs):
        assert(gnn_layer_module is not None and readout_module is not None and confidence_module is not None)
        super(ACTIterGNN, self).__init__()
        self.gnn_layer_module = gnn_layer_module
        self.readout_module = readout_module
        self.confidence_module = confidence_module
        self.layer_num = layer_num
        self.tao = tao
    @staticmethod
    def update_x(x, new_x, left_confidence, current_confidence, decreasing_ratio=1):
        return x + left_confidence*current_confidence*new_x
    @staticmethod
    def next_x(x, new_x, left_confidence, decreasing_ratio=1):
        return x
    @staticmethod
    def update_confidence(left_confidence, current_confidence, decreasing_ratio=1):
        return left_confidence*(1.-current_confidence)
    @property
    def decreasing_ratio(self,):
        return None
    def forward(self, data):
        if self.layer_num == 0:
            return data.x, 0, torch.zeros_like(data.x[:,0:1])
        x, batch = data.x, data.batch
        kwargs = {k:v for k,v in data.__dict__.items()}
        kwargs = kwargs['_store']
        #kwargs.pop('x')
        del kwargs['x']
        new_x = x

        left_confidence = torch.ones_like(x[:,0:1])
        residual_confidence = torch.ones_like(x[:,0:1])
        zero_mask = torch.zeros_like(x[:,0:1])
        for iter_num in range(self.layer_num):
            data = Batch(x=self.next_x(x, new_x, left_confidence, self.decreasing_ratio), **kwargs)
            new_x = self.gnn_layer_module(data)
            global_feat = self.readout_module(Batch(x=new_x, **kwargs))
            current_confidence = self.confidence_module(global_feat)[batch]

            left_confidence = left_confidence - current_confidence*(1-zero_mask)
            current_zero_mask = (left_confidence < 1e-7).type(torch.float)
            residual_confidence = residual_confidence - current_confidence*(1-current_zero_mask)
            x = x + (current_confidence*(1-current_zero_mask)+residual_confidence*current_zero_mask*(1-zero_mask))*new_x
            zero_mask = current_zero_mask
            if torch.min(zero_mask).item() > 0.5:
                break;
        return x, iter_num, residual_confidence
def ACT0IterGNN(*args, **kwargs):
    return ACTIterGNN(0, *args, **kwargs)
def ACT1IterGNN(*args, **kwargs):
    return ACTIterGNN(0.1, *args, **kwargs)
def ACT2IterGNN(*args, **kwargs):
    return ACTIterGNN(0.01, *args, **kwargs)
def ACT3IterGNN(*args, **kwargs):
    return ACTIterGNN(0.001, *args, **kwargs)

class IterGNN(nn.Module):
    def __init__(self, gnn_layer_module=None, readout_module=None, confidence_module=None,
                 layer_num=1, *args, **kwargs):
        assert(gnn_layer_module is not None and readout_module is not None and confidence_module is not None)
        super(IterGNN, self).__init__()
        self.gnn_layer_module = gnn_layer_module
        self.readout_module = readout_module
        self.confidence_module = confidence_module
        self.layer_num = layer_num
    def forward(self, data):
        if self.layer_num == 0:
            return data.x, 0
        x, batch = data.x, data.batch
        kwargs = {k:v for k,v in data.__dict__.items()}
        kwargs = kwargs['_store']
        #kwargs.pop('x')
        del kwargs['x']
        new_x = x

        left_confidence = torch.ones_like(x[:,0:1])

        for iter_num in range(self.layer_num):
            #print("try iter", iter_num)
            if torch.max(left_confidence).item() > 1e-7:
                data = Batch(x=self.next_x(x, new_x, left_confidence, self.decreasing_ratio), **kwargs)
                new_x = self.gnn_layer_module(data)
                global_feat = self.readout_module(Batch(x=new_x, **kwargs))
                current_confidence = self.confidence_module(global_feat)[batch]
                x = self.update_x(
                    x if iter_num != 0 else torch.zeros_like(x),
                    new_x, left_confidence, current_confidence, self.decreasing_ratio
                )
                left_confidence = self.update_confidence(left_confidence, current_confidence, self.decreasing_ratio)
            else:
                break

        return x, iter_num
    @staticmethod
    def update_x(x, new_x, left_confidence, current_confidence, decreasing_ratio=1):
        return x + left_confidence*current_confidence*new_x
    @staticmethod
    def update_confidence(left_confidence, current_confidence, decreasing_ratio=1):
        return left_confidence*(1.-current_confidence)
    @property
    def decreasing_ratio(self,):
        return None
    @staticmethod
    def next_x(x, new_x, left_confidence, decreasing_ratio=1):
        return new_x
class IterNodeGNN(nn.Module):
    def __init__(self, gnn_layer_module=None, readout_module=None, confidence_module=None,
                 layer_num=1, *args, **kwargs):
        assert(gnn_layer_module is not None and readout_module is not None and confidence_module is not None)
        super(IterNodeGNN, self).__init__()
        self.gnn_layer_module = gnn_layer_module
        self.readout_module = readout_module
        self.confidence_module = confidence_module
        self.layer_num = layer_num
    def forward(self, data):
        if self.layer_num == 0:
            return data.x, 0
        x, batch = data.x, data.batch
        kwargs = {k:v for k,v in data.__dict__.items()}
        kwargs.pop('x')
        new_x = x

        left_confidence = torch.ones_like(x[:,0:1])
        for iter_num in range(self.layer_num):
            if torch.max(left_confidence).item() > 1e-7:
                data = Batch(x=self.next_x(x, new_x, left_confidence, self.decreasing_ratio), **kwargs)
                new_x = self.gnn_layer_module(data)
                # global_feat = self.readout_module(Batch(x=new_x, **kwargs))
                # current_confidence = self.confidence_module(global_feat)[batch]
                current_confidence = self.confidence_module(new_x)
                x = self.update_x(
                    x if iter_num != 0 else torch.zeros_like(x),
                    new_x, left_confidence, current_confidence, self.decreasing_ratio
                )
                left_confidence = self.update_confidence(left_confidence, current_confidence, self.decreasing_ratio)
            else:
                break

        return x, iter_num
    @staticmethod
    def update_x(x, new_x, left_confidence, current_confidence, decreasing_ratio=1):
        return x + left_confidence*current_confidence*new_x
    @staticmethod
    def update_confidence(left_confidence, current_confidence, decreasing_ratio=1):
        return left_confidence*(1.-current_confidence)
    @property
    def decreasing_ratio(self,):
        return None
    @staticmethod
    def next_x(x, new_x, left_confidence, decreasing_ratio=1):
        return new_x
class DecIterGNN(IterGNN):
    @staticmethod
    def update_x(x, new_x, left_confidence, current_confidence, decreasing_ratio):
        return decreasing_ratio*x + left_confidence*current_confidence*new_x
    @staticmethod
    def update_confidence(left_confidence, current_confidence, decreasing_ratio):
        return left_confidence*(1.-current_confidence)*decreasing_ratio
    @property
    def decreasing_ratio(self,):
        return (1-1e-4)
class DecIterNodeGNN(IterNodeGNN):
    @staticmethod
    def update_x(x, new_x, left_confidence, current_confidence, decreasing_ratio):
        return decreasing_ratio*x + left_confidence*current_confidence*new_x
    @staticmethod
    def update_confidence(left_confidence, current_confidence, decreasing_ratio):
        return left_confidence*(1.-current_confidence)*decreasing_ratio
    @property
    def decreasing_ratio(self,):
        return (1-1e-4)

def GNNArchitectures(gnn_layer_module=None, readout_module=None, confidence_module=None,
                     layer_name='IterGNN', layer_num=1, *args, **kwargs):
    if layer_name in ['DeepGNN', 'SharedDeepGNN']:
        return globals()[layer_name](gnn_layer_module, layer_num=layer_num, *args, **kwargs)
    elif 'Iter' in layer_name:
        return globals()[layer_name](gnn_layer_module, readout_module, confidence_module,
                                     layer_num=layer_num, *args, **kwargs)
    else:
        raise NotImplementedError('There is no GNN architecture named %s'%layer_name)

