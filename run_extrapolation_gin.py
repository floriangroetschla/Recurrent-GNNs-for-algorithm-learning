import pandas as pd
import json

from gin_model import GIN
import torch
import train
import datasets
from torch_geometric.loader import DataLoader
from functools import partialmethod
from tqdm import tqdm
import numpy as np

runs = pd.read_csv('output_gin.csv')

conv = 'gin'

graph_sizes = [10, 50, 100, 1000, 10000]
runs = runs[(runs.epoch == 100)]

tree_models = []
for index, row in runs[(runs.dataset=='tree-path')].iterrows():
    dataclass = datasets.Trees()
    model = GIN(2, 8, 2, 4, 0.2, conv, True, False, num_layers=10)
    model.load_state_dict(torch.load('models/model_train_' + row['model_name'], map_location=torch.device('cpu')))
    tree_models.append([model] + [row['run'], row['use_l2']])

prefix_models = []
for index, row in runs[(runs.dataset=='prefix')].iterrows():
    dataclass = datasets.PrefixSum()
    model = GIN(4, 8, 2, 4, 0.2, conv, True, False, num_layers=10)
    model.load_state_dict(torch.load('models//model_train_' + row['model_name'], map_location=torch.device('cpu')))
    prefix_models.append([model] + [row['run'], row['use_l2']])

distance_models = []
for index, row in runs[(runs.dataset=='distance')].iterrows():
    dataclass = datasets.Distance()
    model = GIN(2, 8, 2, 4, 0.2, conv, True, False, num_layers=10)
    model.load_state_dict(torch.load('models/model_train_' + row['model_name'], map_location=torch.device('cpu')))
    distance_models.append([model] + [row['run'], row['use_l2']])

print(len(distance_models))
data_generalization = []
tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)
for i in tqdm(graph_sizes):
    print(i)
    dataset_tree = datasets.Trees().makedata(num_graphs=10, num_nodes=i)
    dataset_prefix = datasets.PrefixSum().makedata(num_graphs=10, num_nodes=i)
    dataset_distance = datasets.Distance().makedata(num_graphs=10, num_nodes=i)
    loader_tree = DataLoader(dataset_tree, batch_size=1, shuffle=False)
    loader_prefix = DataLoader(dataset_prefix, batch_size=1, shuffle=False)
    loader_distance = DataLoader(dataset_distance, batch_size=1, shuffle=False)
    for j in range(len(tree_models)):
        scores = train.test(tree_models[j][0], 'cpu', loader_tree, 1.2, baseline='gin')
        data_generalization.append(['tree-path', i, scores[1], scores[2]] + tree_models[j][1:])
        scores = train.test(prefix_models[j][0], 'cpu', loader_prefix, 1.2, baseline='gin')
        data_generalization.append(['prefix', i, scores[1], scores[2]] + prefix_models[j][1:])
        scores = train.test(distance_models[j][0], 'cpu', loader_distance, 1.2, baseline='gin')
        data_generalization.append(['distance', i, scores[1], scores[2]] + distance_models[j][1:])


values = pd.DataFrame(data_generalization, columns =['dataset', 'size', 'accuracy', 'f1', 'run', 'use_l2'])
values.to_csv(f'output_extrapolation_gin.csv', encoding='utf-8', index=False)

