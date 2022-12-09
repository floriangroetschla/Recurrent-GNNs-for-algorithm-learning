import torch
import torch.nn.functional as F

import torch_geometric.transforms as T
from torch_geometric.nn import GATv2Conv, GINConv, GatedGraphConv

from tqdm import tqdm
import networkx as nx
from torch_geometric.loader import DataLoader
from torch import nn


from torch_geometric.nn import MessagePassing

class GRUConv(MessagePassing):
    def __init__(self, emb_dim, aggr):
        super(GRUConv, self).__init__(aggr=aggr)
        self.rnn = torch.nn.GRUCell(emb_dim, emb_dim)

    def forward(self, x, edge_index):
        out = self.rnn(self.propagate(edge_index, x=x), x)
        return out

class GRUMLPConv(MessagePassing):
    def __init__(self, emb_dim, mlp_edge, aggr):
        super(GRUMLPConv, self).__init__(aggr=aggr)
        self.rnn = torch.nn.GRUCell(emb_dim, emb_dim)
        self.mlp_edge = mlp_edge

    def forward(self, x, edge_index):
        out = self.rnn(self.propagate(edge_index, x=x), x)
        return out

    def message(self, x_j, x_i):
        concatted = torch.cat((x_j, x_i), dim=1)
        return self.mlp_edge(concatted)      

class FConv(MessagePassing):
    def __init__(self, mlp, aggr):
        super(FConv, self).__init__(aggr=aggr)
        self.mlp = mlp

    def forward(self, x, edge_index):
        out = self.mlp(torch.cat([self.propagate(edge_index, x=x), x], dim=1))
        return out


class GINMLPConv(MessagePassing):
    def __init__(self, mlp, mlp_edge, aggr):
        '''
            emb_dim (int): node embedding dimensionality
        '''

        super(GINMLPConv, self).__init__(aggr=aggr)

        self.mlp = mlp
        self.mlp_edge = mlp_edge
        self.eps = torch.nn.Parameter(torch.Tensor([0]))

    def forward(self, x, edge_index):
        out = self.mlp((1 + self.eps) *x + self.propagate(edge_index, x=x))

        return out

    def message(self, x_j, x_i):
        concatted = torch.cat((x_j, x_i), dim=1)
        return self.mlp_edge(concatted)

    def update(self, aggr_out):
        return aggr_out

class RecGNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, hidden_state_factor, dropout, conv='gin', skip_input=False, skip_prev=False, aggregation='add', normalization=torch.nn.Identity, activation_function=torch.nn.ReLU, gumbel=False):
        super(RecGNN, self).__init__()

        self.dropout = dropout
        self.encoder = self.get_mlp(in_channels, hidden_state_factor * hidden_channels, hidden_channels, normalization, activation_function)
       
        if conv == 'gin-mlp':
            self.conv = GINMLPConv(self.get_mlp(hidden_channels, hidden_state_factor * hidden_channels, hidden_channels, normalization, activation_function), self.get_mlp(2*hidden_channels, hidden_state_factor *2 *hidden_channels, hidden_channels, normalization, activation_function), aggr=aggregation)
        elif conv == 'gat':
            self.conv = GATv2Conv(hidden_channels, hidden_channels)
        elif conv == 'gin':
            self.conv = GINConv(self.get_mlp(hidden_channels, hidden_state_factor * hidden_channels, hidden_channels, normalization, activation_function), aggr=aggregation)
        elif conv == 'skip-conv':
            self.conv = FConv(self.get_mlp(2*hidden_channels, hidden_state_factor * hidden_channels, hidden_channels, normalization, activation_function), aggr=aggregation)
        elif conv == 'gru':
            self.conv = GRUConv(hidden_channels, aggr=aggregation)
        elif conv == 'gru-mlp':
            self.conv = GRUMLPConv(hidden_channels, self.get_mlp(2*hidden_channels, hidden_state_factor * 2*hidden_channels, hidden_channels, normalization, activation_function), aggr=aggregation)
        elif conv == 'gated':
            self.conv = GatedGraphConv(hidden_channels, 1, aggr=aggregation)
        else:
            raise Exception('Unrecognized option: ' + conv)

        self.decoder = self.get_mlp(hidden_channels, hidden_state_factor * hidden_channels, out_channels, normalization, activation_function, last_activation = False)
        
        self.skip_input = self.get_mlp(hidden_channels + in_channels, hidden_state_factor * hidden_channels, hidden_channels, normalization, activation_function) if skip_input else None
        self.skip_previous = self.get_mlp(2*hidden_channels, hidden_state_factor * hidden_channels, hidden_channels, normalization, activation_function) if skip_prev else None
        self.gumbel = gumbel


    def get_mlp(self, input_dim, hidden_dim, output_dim, normalization, activation_function, last_activation = True):
        modules = [torch.nn.Linear(input_dim, int(hidden_dim)), normalization(int(hidden_dim)), activation_function(), torch.nn.Dropout(self.dropout), torch.nn.Linear(int(hidden_dim), int(hidden_dim)), normalization(int(hidden_dim)), activation_function(), torch.nn.Linear(int(hidden_dim), output_dim)]
        if last_activation:
            modules.append(normalization(output_dim))
            modules.append(activation_function())
        return torch.nn.Sequential(*modules)

    def forward(self, batched_data, iterations, return_layers=False):
        x, edge_index = batched_data.x, batched_data.edge_index
        x_orig = x
        layers = []

        x = self.encoder(x)

        previous = x
        if return_layers:
            layers.append(x)

        for i in range(iterations):
            if self.skip_input is not None:
                x = self.skip_input(torch.cat([x, x_orig], dim=1))
            x = self.conv(x, edge_index)

            if self.skip_previous is not None:
                x = self.skip_previous(torch.cat([x, previous], dim=1))
            if self.gumbel:
                x = F.gumbel_softmax(x, tau=1, hard=True)
            previous = x
            if return_layers:
                layers.append(x)

        x = self.decoder(x)
        if return_layers:
            return x, layers
        else:
            return x


