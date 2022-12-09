import pandas as pd
import json

from model import RecGNN
import torch
import train
import datasets
from torch_geometric.loader import DataLoader
from functools import partialmethod
from tqdm import tqdm
import numpy as np

from itergnn import NodeGNNModels


df_best_parameters = pd.read_csv('output_itergnn.csv')
dataclass = datasets.PrefixSum()

df_best_parameters = df_best_parameters[df_best_parameters.epoch == 100]

graph_sizes = [10, 50, 100, 1000, 10000]

tree_models = []
for index, row in df_best_parameters[(df_best_parameters.dataset=='tree-path')].iterrows():
    dataclass = datasets.Trees()
    model = NodeGNNModels(in_channel=2, edge_channel=1, hidden_size=8,
                                  num_predictions=2,
                                  out_channel=2, embedding_layer_num=2, architecture_name='IterGNN',
                                  layer_num=12, module_num=1, layer_name='PathGNN', input_feat_flag=True,
                                  homogeneous_flag=1, readout_name='Max', confidence_layer_num=1, head_layer_num=1)

    model.load_state_dict(torch.load('models/model_train_' + row['model_name'], map_location=torch.device('cpu')))
    tree_models.append([model] + [row['run'], row['use_l2']])

prefix_models = []
for index, row in df_best_parameters[(df_best_parameters.dataset=='prefix')].iterrows():
    dataclass = datasets.PrefixSum()
    model = NodeGNNModels(in_channel=4, edge_channel=1, hidden_size=8,
                                  num_predictions=2,
                                  out_channel=2, embedding_layer_num=2, architecture_name='IterGNN',
                                  layer_num=12, module_num=1, layer_name='PathGNN', input_feat_flag=True,
                                  homogeneous_flag=1, readout_name='Max', confidence_layer_num=1, head_layer_num=1)

    model.load_state_dict(torch.load('models/model_train_' + row['model_name'], map_location=torch.device('cpu')))
    prefix_models.append([model] + [row['run'], row['use_l2']])

distance_models = []
for index, row in df_best_parameters[(df_best_parameters.dataset=='distance')].iterrows():
    dataclass = datasets.Distance()
    model = NodeGNNModels(in_channel=2, edge_channel=1, hidden_size=8,
                                  num_predictions=2,
                                  out_channel=2, embedding_layer_num=2, architecture_name='IterGNN',
                                  layer_num=12, module_num=1, layer_name='PathGNN', input_feat_flag=True,
                                  homogeneous_flag=1, readout_name='Max', confidence_layer_num=1, head_layer_num=1)

    model.load_state_dict(torch.load('models/model_train_' + row['model_name'], map_location=torch.device('cpu')))
    distance_models.append([model] + [row['run'], row['use_l2']])
print(len(distance_models))

data_generalization = []
tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)
for n in graph_sizes:
    print(n)
    dataset_tree = datasets.Trees().makedata(num_graphs=10, num_nodes=n)
    dataset_prefix = datasets.PrefixSum().makedata(num_graphs=10, num_nodes=n)
    dataset_distance = datasets.Distance().makedata(num_graphs=10, num_nodes=n)
    loader_tree = DataLoader(dataset_tree, batch_size=1, shuffle=False)
    loader_prefix = DataLoader(dataset_prefix, batch_size=1, shuffle=False)
    loader_distance = DataLoader(dataset_distance, batch_size=1, shuffle=False)
    for j in range(len(tree_models)):
        tree_models[j][0].layer_num = int(1.2*n)
        for i in range(len(tree_models[j][0].gnn_module_list)):
            tree_models[j][0].gnn_module_list[i].layer_num = int(1.2*n)
        prefix_models[j][0].layer_num = int(1.2*n)
        for i in range(len(prefix_models[j][0].gnn_module_list)):
            prefix_models[j][0].gnn_module_list[i].layer_num = int(1.2*n)
        distance_models[j][0].layer_num = int(1.2*n)
        for i in range(len(distance_models[j][0].gnn_module_list)):
            distance_models[j][0].gnn_module_list[i].layer_num = int(1.2*n)


        data_generalization.append(['tree-path', n] + list(train.test(tree_models[j][0], 'cpu', loader_tree, 1.2, baseline='itergnn'))[1:] + tree_models[j][1:])
        data_generalization.append(['prefix', n] + list(train.test(prefix_models[j][0], 'cpu', loader_prefix, 1.2, baseline='itergnn'))[1:] + prefix_models[j][1:])
        data_generalization.append(['distance', n] + list(train.test(distance_models[j][0], 'cpu', loader_distance, 1.2, baseline='itergnn'))[1:] + distance_models[j][1:])


values = pd.DataFrame(data_generalization, columns =['dataset', 'size', 'accuracy', 'f1', 'run', 'use_l2'])
values.to_csv('output_extrapolation_itergnn.csv', encoding='utf-8', index=False)

