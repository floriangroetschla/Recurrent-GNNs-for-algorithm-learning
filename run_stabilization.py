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

conv = sys.argv[1]

runs = pd.read_csv('output_recGNN.csv')

iteration_factors = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 1000.0]
runs = runs[(runs.skip_input == True) & (runs.epoch == 100)]

tree_models = []
for index, row in runs[(runs.dataset=='tree-path')&(runs.conv==conv)].iterrows():
    dataclass = datasets.Trees()
    model = RecGNN(2, 8, 2, 4, 0.2, conv, True, False, aggregation=row['aggregation'])
    model.load_state_dict(torch.load('models/model_train_' + row['model_name'], map_location=torch.device('cpu')))
    tree_models.append([model] + [row['run'], row['use_l2'], row['vary_layers']])

prefix_models = []
for index, row in runs[(runs.dataset=='prefix')&(runs.conv==conv)].iterrows():
    dataclass = datasets.PrefixSum()
    model = RecGNN(4, 8, 2, 4, 0.2, conv, True, False, aggregation=row['aggregation'])
    model.load_state_dict(torch.load('models/model_train_' + row['model_name'], map_location=torch.device('cpu')))
    prefix_models.append([model] + [row['run'], row['use_l2'], row['vary_layers']])

distance_models = []
for index, row in runs[(runs.dataset=='distance')&(runs.conv==conv)].iterrows():
    dataclass = datasets.Distance()
    model = RecGNN(2, 8, 2, 4, 0.2, conv, True, False, aggregation=row['aggregation'])
    model.load_state_dict(torch.load('models/model_train_' + row['model_name'], map_location=torch.device('cpu')))
    distance_models.append([model] + [row['run'], row['use_l2'], row['vary_layers']])

print(len(distance_models))
dataset_tree = datasets.Trees().makedata(num_graphs=10, num_nodes=100)
dataset_prefix = datasets.PrefixSum().makedata(num_graphs=10, num_nodes=100)
dataset_distance = datasets.Distance().makedata(num_graphs=10, num_nodes=100)
loader_tree = DataLoader(dataset_tree, batch_size=1, shuffle=False)
loader_prefix = DataLoader(dataset_prefix, batch_size=1, shuffle=False)
loader_distance = DataLoader(dataset_distance, batch_size=1, shuffle=False)

data_stabilization = []
tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)
for i in iteration_factors:
    print(i)
    for j in range(len(tree_models)):
        scores = train.test(tree_models[j][0], 'cpu', loader_tree, i)
        data_stabilization.append(['tree-path', i, scores[1], scores[2]] + tree_models[j][1:])
        scores = train.test(prefix_models[j][0], 'cpu', loader_prefix, i)
        data_stabilization.append(['prefix', i, scores[1], scores[2]] + prefix_models[j][1:])
        scores = train.test(distance_models[j][0], 'cpu', loader_distance, i)
        data_stabilization.append(['distance', i, scores[1], scores[2]] + distance_models[j][1:])


values = pd.DataFrame(data_stabilization, columns =['dataset', 'factor', 'accuracy', 'f1', 'run', 'use_l2', 'vary_layers'])
values.to_csv(f'output_stabilization_{conv}.csv', encoding='utf-8', index=False)

