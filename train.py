
import torch

from tqdm import tqdm
from torch_geometric.loader import DataLoader
from torch import nn

import random
import numpy as np
import json
import os
from functools import partialmethod
import datetime 
import torch.nn.functional as F
from sklearn.metrics import f1_score 

from torch.utils.tensorboard import SummaryWriter

from itergnn import NodeGNNModels
from model import RecGNN
from gin_model import GIN
import datasets
TENSORBOARD_DIRECTORY = 'runs/'
MODEL_DIRECTORY = 'models/'

class HLoss(nn.Module):
    def __init__(self):
        super(HLoss, self).__init__()

    def forward(self, x):
        #b = x * torch.log(x)
        b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
        b = -1.0 * b.sum()
        return b

def train(model, device, loader, optimizer, class_imbalance_weight, train_layer_fraction, use_l1, l1_weight, use_l2, l2_weight, use_entropy_loss, baseline=""):
    model.train()

    cls_criterion = torch.nn.CrossEntropyLoss(weight = class_imbalance_weight)

    loss = 0
    l1_all = 0
    l2_all = 0
    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)
        n = len(batch.x)/batch.num_graphs

        optimizer.zero_grad()

        layers = int(train_layer_fraction * n)
        states = []
        
        #itergnn code
        if baseline == "itergnn":
            model.layer_num = layers
            for i in range(len(model.gnn_module_list)):
                model.gnn_module_list[i].layer_num = layers
            pred, aux_loss = model(batch)
            cls_loss = cls_criterion(pred[0], batch.y.to(torch.long))

        elif baseline == "gin":
            pred, states = model(batch, layers, return_layers=True)
            cls_loss = cls_criterion(pred, batch.y.to(torch.long))
            l2_norm = torch.mean(torch.linalg.norm(states[-1], dim=1))
            if use_l2:
                cls_loss += l2_weight * l2_norm        
        else: 
            pred, states = model(batch, layers, return_layers=True)
            cls_loss = cls_criterion(pred, batch.y.to(torch.long))
            l1_norm = torch.norm(states[-1], 1)/n
            l2_norm = torch.mean(torch.linalg.norm(states[-1], dim=1))
            if use_l1:
                cls_loss += l1_weight * l1_norm
            if use_l2:
                cls_loss += l2_weight * l2_norm
            if use_entropy_loss:
                for state in states:
                    #if np.isnan(HLoss()(state).item()):
                    #    print(state)
                    cls_loss += 0.00001 * HLoss()(state)


        cls_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        torch.nn.utils.clip_grad_value_(model.parameters(), 1.0)
        optimizer.step()
        loss += cls_loss.item() * batch.num_graphs
        if not (baseline == "itergnn" or baseline == "gin"):
            l1_all += l1_norm.item() * batch.num_graphs
            l2_all += l2_norm.item() * batch.num_graphs
    return loss/len(loader.dataset), l1_all/len(loader.dataset), l2_all/len(loader.dataset)


@torch.no_grad()
def test(model, device, loader, layer_fraction, baseline = False):
    model.eval()

    criterion = torch.nn.CrossEntropyLoss()

    loss = 0
    acc = 0
    tot_nodes = 0
    ys = torch.tensor([], dtype=torch.int)
    y_hats = torch.tensor([], dtype=torch.int)
    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)
        n = len(batch.x)/batch.num_graphs
        tot_nodes += len(batch.x)
        layers = int(layer_fraction * n) 
        
        
        
        with torch.no_grad():

            if baseline == "itergnn": 
                model.layer_num = layers
                for i in range(len(model.gnn_module_list)):
                    model.gnn_module_list[i].layer_num = layers
                pred, iter_num = model(batch)
                #print(iter_num)
                pred = pred[0]
            elif baseline == "gin":
                pred = model(batch, layers)
            else: 
                pred = model(batch, layers)
        
        loss += criterion(pred, batch.y.to(torch.long)).item() * batch.num_graphs

        y_pred = torch.argmax(pred,dim=1)
        acc += torch.sum(y_pred == batch.y)

        ys  = torch.cat((ys, y_pred))
        y_hats = torch.cat((y_hats, batch.y))
    acc = acc.item()
    if pred.shape[1] == 2:
        f1 = f1_score(ys, y_hats, average='binary')
    else:
        f1 = f1_score(ys, y_hats, average='micro')
    return (loss/len(loader.dataset), acc/tot_nodes, f1)
    


def get_class_imbalance(dataset, num_classes = 2):
    tot = torch.tensor([0.0 for i in range(num_classes)])
    for data in dataset:
        tot += torch.bincount(data.y.to(torch.int),minlength = num_classes)
    x = torch.div(torch.sum(tot), tot)
    #x = torch.sum(tot)/tot
    x= torch.div(x,num_classes)
    #print(x)
    return x

def train_and_eval(config, cluster=None):
    if cluster is not None:
        print(json.dumps(config.__getstate__()))

    if not config.verbose:
        tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)

    device = f'cuda:{config.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    random.seed(0)
    np.random.seed(0)

    name = TENSORBOARD_DIRECTORY + f'{datetime.datetime.now()}'+'Run' + config.conv + '_skipP_' + str(config.skip_previous) + '_skipI_' + str(config.skip_input) + '_emb_' + str(config.hidden_dimension) + '_dropout_' + str(config.dropout) + '_decay_' + str(config.use_weight_decay) + '_hidden_state_factor_' + str(config.hidden_state_factor) + '_dataset_' + config.dataset + '_l1_' + str(config.use_l1)

    if config.use_tensorboard:
        writer = SummaryWriter(name)
    
    val_split = 0.8

    train_layer_fraction = config.train_layer_fraction



    num_train_graphs = config.num_train_graphs
    num_train_nodes = config.num_train_nodes
    
    num_generalize_graphs = config.num_generalize_graphs
    num_generalize_nodes  = config.num_generalize_nodes


    if config.dataset == 'tree-path':
        dataclass = datasets.Trees()
        dataset = dataclass.makedata(num_graphs=num_train_graphs, num_nodes=num_train_nodes)
        bigger_trees = dataclass.makedata(num_graphs=num_generalize_graphs, num_nodes=num_generalize_nodes)

        np.random.shuffle(dataset)
        train_size = int(val_split*len(dataset))

        train_loader = DataLoader(dataset[:train_size], batch_size=config.batch_size, shuffle=True)
        valid_loader = DataLoader(dataset[train_size:], batch_size=1, shuffle=False)
        bigger_graph_loader = DataLoader(bigger_trees, batch_size=1, shuffle=False)

        in_channels = dataclass.num_features
        out_channels = dataclass.num_classes
    if config.dataset == 'tree-path-diameter':
        dataclass = datasets.Trees()
        dataset = dataclass.makedata(num_graphs=num_train_graphs, num_nodes=15)
        bigger_trees = dataclass.makedata(num_graphs=num_generalize_graphs, num_nodes=num_generalize_nodes)


        train_layer_fraction = 0.68
        np.random.shuffle(dataset)
        train_size = int(val_split*len(dataset))

        train_loader = DataLoader(dataset[:train_size], batch_size=config.batch_size, shuffle=True)
        valid_loader = DataLoader(dataset[train_size:], batch_size=1, shuffle=False)
        bigger_graph_loader = DataLoader(bigger_trees, batch_size=1, shuffle=False)

        in_channels = dataclass.num_features
        out_channels = dataclass.num_classes

    elif config.dataset == 'distance':
        dataclass = datasets.Distance()

        dataset = dataclass.makedata(num_graphs = num_train_graphs, num_nodes = num_train_nodes)
        bigger_distance = dataclass.makedata(num_graphs = num_generalize_graphs, num_nodes = num_generalize_nodes)

        np.random.shuffle(dataset)
        train_size = int(val_split*len(dataset))

        train_loader = DataLoader(dataset[:train_size], batch_size=config.batch_size, shuffle=True)
        valid_loader = DataLoader(dataset[train_size:], batch_size=1, shuffle=False)
        bigger_graph_loader = DataLoader(bigger_distance, batch_size=1, shuffle=False)

        in_channels = dataclass.num_features
        out_channels = dataclass.num_classes

    elif config.dataset == 'distance_delaunay':
        dataclass = datasets.Distance_Delaunay()

        dataset = dataclass.makedata(num_graphs = num_train_graphs, num_nodes = num_train_nodes)
        bigger_distance = dataclass.makedata(num_graphs = num_generalize_graphs, num_nodes = num_generalize_nodes)

        np.random.shuffle(dataset)
        train_size = int(val_split*len(dataset))

        train_loader = DataLoader(dataset[:train_size], batch_size=config.batch_size, shuffle=True)
        valid_loader = DataLoader(dataset[train_size:], batch_size=1, shuffle=False)
        bigger_graph_loader = DataLoader(bigger_distance, batch_size=1, shuffle=False)

        in_channels = dataclass.num_features
        out_channels = dataclass.num_classes

    elif config.dataset == 'distance_delaunay-diameter':
        dataclass = datasets.Distance_Delaunay()

        dataset = dataclass.makedata(num_graphs = num_train_graphs, num_nodes = 25)
        bigger_distance = dataclass.makedata(num_graphs = num_generalize_graphs, num_nodes = num_generalize_nodes)

        np.random.shuffle(dataset)
        train_size = int(val_split*len(dataset))

        train_loader = DataLoader(dataset[:train_size], batch_size=config.batch_size, shuffle=True)
        valid_loader = DataLoader(dataset[train_size:], batch_size=1, shuffle=False)
        bigger_graph_loader = DataLoader(bigger_distance, batch_size=1, shuffle=False)
        train_layer_fraction = 0.33
        in_channels = dataclass.num_features
        out_channels = dataclass.num_classes


    elif config.dataset == 'distanceK':
        dataclass = datasets.DistanceK(k = 3)

        dataset = dataclass.makedata(num_graphs = num_train_graphs, num_nodes = num_train_nodes)
        bigger_distance = dataclass.makedata(num_graphs = num_generalize_graphs, num_nodes = num_generalize_nodes)

        np.random.shuffle(dataset)
        train_size = int(val_split*len(dataset))

        train_loader = DataLoader(dataset[:train_size], batch_size=config.batch_size, shuffle=True)
        valid_loader = DataLoader(dataset[train_size:], batch_size=1, shuffle=False)
        bigger_graph_loader = DataLoader(bigger_distance, batch_size=1, shuffle=False)

        in_channels = dataclass.num_features
        out_channels = dataclass.num_classes

    elif config.dataset == 'prefix':
        dataclass = datasets.PrefixSum()
        dataset = dataclass.makedata(num_graphs = num_train_graphs, num_nodes = num_train_nodes)
        bigger_prefix = dataclass.makedata(num_graphs = num_generalize_graphs, num_nodes = num_generalize_nodes)

        np.random.shuffle(dataset)
        train_size = int(val_split*len(dataset))

        train_loader = DataLoader(dataset[:train_size], batch_size=config.batch_size, shuffle=True)
        valid_loader = DataLoader(dataset[train_size:], batch_size=1, shuffle=False)
        bigger_graph_loader = DataLoader(bigger_prefix, batch_size=1, shuffle=False)
        
        in_channels = dataclass.num_features
        out_channels = dataclass.num_classes


    elif config.dataset == 'prefix-k':
        dataclass = datasets.PrefixSumK(k = 5, inp = 5)
        dataset = dataclass.makedata(num_graphs = num_train_graphs, num_nodes = num_train_nodes)
        bigger_prefix = dataclass.makedata(num_graphs = num_generalize_graphs, num_nodes = num_generalize_nodes)

        np.random.shuffle(dataset)
        train_size = int(val_split*len(dataset))

        train_loader = DataLoader(dataset[:train_size], batch_size=config.batch_size, shuffle=True)
        valid_loader = DataLoader(dataset[train_size:], batch_size=1, shuffle=False)
        bigger_graph_loader = DataLoader(bigger_prefix, batch_size=1, shuffle=False)
        
        in_channels = dataclass.num_features
        out_channels = dataclass.num_classes


    elif config.dataset == 'midpoint':
        dataclass = datasets.MidPoint()
        dataset = dataclass.makedata(num_graphs = num_train_graphs, num_nodes=num_train_nodes, allow_sizes=True)
        bigger_midpoint = dataclass.makedata(num_graphs = num_generalize_graphs, num_nodes=num_generalize_nodes)

        np.random.shuffle(dataset)
        train_size = int(val_split*len(dataset))

        train_loader = DataLoader(dataset[:train_size], batch_size=config.batch_size, shuffle=True)
        valid_loader = DataLoader(dataset[train_size:], batch_size=1, shuffle=False)
        bigger_graph_loader = DataLoader(bigger_midpoint, batch_size=1, shuffle=False)
        
        in_channels = dataclass.num_features
        out_channels = dataclass.num_classes
        


    
    class_imbalance_weight = get_class_imbalance(dataset, out_channels)

    seed = config.run_number
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if config.vary_layers:
        train_layer_fraction += np.random.uniform(-0.2,0.2)


    if config.normalization == 'LayerNorm':
        normalization_function = torch.nn.LayerNorm
    elif config.normalization == 'BatchNorm':
        normalization_function = torch.nn.BatchNorm1d
    elif config.normalization == 'None':
        normalization_function = torch.nn.Identity
    else:
        print('Unrecognized normalization function: ' + config.normalization)
        exit(1)

    if config.activation_function == "ReLU":
        activation_function = torch.nn.ReLU
    elif config.activation_function == "SiLU":
        activation_function = torch.nn.SiLU
    elif config.activation_function == "LeakyReLU":
        activation_function = torch.nn.LeakyReLU
    elif config.activation_function == 'Sigmoid':
        activation_function = torch.nn.Sigmoid
    else:
        print("Unrecognized activation function: " + config.activation_function)
        exit(1)

    model = RecGNN(in_channels, config.hidden_dimension, out_channels, config.hidden_state_factor, config.dropout, conv=config.conv, skip_prev=config.skip_previous, skip_input=config.skip_input, aggregation=config.aggregation, normalization=normalization_function, activation_function=activation_function, gumbel=config.gumbel).to(device)

    if config.baseline == "itergnn":
        model = NodeGNNModels(in_channel=in_channels, edge_channel=1, hidden_size=config.hidden_dimension,
                                  num_predictions=out_channels,
                                  out_channel=out_channels, embedding_layer_num=2, architecture_name='IterGNN',
                                  layer_num=12, module_num=1, layer_name='PathGNN', input_feat_flag=True,
                                  homogeneous_flag=1, readout_name='Max', confidence_layer_num=1, head_layer_num=1).to(device)
    elif config.baseline == "gin":
        model = GIN(in_channels, config.hidden_dimension, out_channels, config.hidden_state_factor, config.dropout, 
                    conv=config.conv, skip_prev=config.skip_previous, skip_input=config.skip_input, aggregation=config.aggregation, 
                    normalization=normalization_function, activation_function=activation_function, gumbel=config.gumbel, num_layers = config.num_train_nodes).to(device)
    elif config.baseline != "none":
        raise Exception('Unrecognized option: ' + config.baseline)



    if config.use_weight_decay:
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=1e-5)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    #scheduler_lambda = lambda epoch: 0.7 ** epoch
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                       factor=0.7, patience=20,
                                                       min_lr=0.00001)

    best_valid_loss = np.Inf
    best_generalization_loss = np.Inf
    if config.verbose:
        print("Start Training")
        
    for epoch in range(1, 1 + config.epochs):
        if config.verbose:
            print("=====Epoch {}".format(epoch))
            print('Training...')

        loss, l1_loss, l2_loss = train(model=model, device=device, loader=train_loader, optimizer = optimizer, class_imbalance_weight=class_imbalance_weight, train_layer_fraction=train_layer_fraction, 
                                use_l1 = config.use_l1, l1_weight = config.l1_weight, use_l2=config.use_l2, l2_weight = config.l2_weight, use_entropy_loss = config.use_entropy_loss, baseline = config.baseline
                                )
        if config.verbose:
            print('Evaluating...')
        train_loss, train_acc, train_f1 = test(model, device, train_loader, train_layer_fraction, baseline = config.baseline)
        valid_loss, valid_acc, valid_f1 = test(model, device, valid_loader, train_layer_fraction, baseline = config.baseline)
        generalization_loss, generalization_acc, generalization_f1 = test(model, device, bigger_graph_loader, train_layer_fraction, baseline = config.baseline)

        print(json.dumps({'run': config.run_number, 'epoch': epoch, 'lr': optimizer.param_groups[0]['lr'], 'optimization_l1': l1_loss, 'optimization_l2': l2_loss, 'optimization_loss': loss, 'train_loss': train_loss, 'train_acc': train_acc, 'train_f1': train_f1, 'valid_loss': valid_loss, 'valid_acc': valid_acc, 'valid_f1': valid_f1, 'generalization_loss': generalization_loss, 'generalization_acc': generalization_acc, 'generalization_f1': generalization_f1}))

        if config.use_tensorboard:
            writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)
            writer.add_scalar('optimization_loss', loss, epoch)
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Loss/validation', valid_loss, epoch)
            writer.add_scalar('Loss/generalization', generalization_loss, epoch)
            writer.add_scalar('Loss/l1_train', l1_loss, epoch)
            writer.add_scalar('Loss/l2_train', l2_loss, epoch)
            writer.add_scalar('Accuracy/train', train_acc, epoch)
            writer.add_scalar('Accuracy/validation', valid_acc, epoch)
            writer.add_scalar('Accuracy/generalization', generalization_acc, epoch)


        if config.store_models:
            if valid_loss <= best_valid_loss:
                best_valid_loss = valid_loss
                file_name = MODEL_DIRECTORY + 'model_train_' + config.model_name
                if os.path.exists(file_name):
                    os.remove(file_name)
                torch.save(model.state_dict(), file_name)
                if config.verbose:
                    print('Saved new train model')

            if generalization_loss <= best_generalization_loss:
                best_generalization_loss = generalization_loss
                file_name = MODEL_DIRECTORY + 'model_generalization_' + config.model_name
                if os.path.exists(file_name):
                    os.remove(file_name)
                torch.save(model.state_dict(), file_name)
                if config.verbose:
                    print('Saved new generalization model')

            if epoch == config.epochs:
                file_name = MODEL_DIRECTORY + 'model_last_' + config.model_name
                if os.path.exists(file_name):
                    os.remove(file_name)
                torch.save(model.state_dict(), file_name)
                if config.verbose:
                    print('Saved last model')

        if config.use_scheduler:
            scheduler.step(valid_loss)

    print('Finished')

