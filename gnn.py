
import torch

from tqdm import tqdm
from torch_geometric.loader import DataLoader
from torch import nn
from test_tube import HyperOptArgumentParser
from test_tube.hpc import SlurmCluster

import random
import numpy as np
import json
import os
from functools import partialmethod
import datetime 
import hashlib
import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter


from model import RecGNN
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

def train(model, device, loader, optimizer, class_imbalance_weight, train_layer_fraction, use_l1, l1_weight, use_l2, l2_weight, use_entropy_loss):
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
        l1_all += l1_norm.item() * batch.num_graphs
        l2_all += l2_norm.item() * batch.num_graphs
    return loss/len(loader.dataset), l1_all/len(loader.dataset), l2_all/len(loader.dataset)


@torch.no_grad()
def test(model, device, loader, layer_fraction):
    model.eval()

    criterion = torch.nn.CrossEntropyLoss()

    loss = 0
    acc = 0
    tot_nodes = 0
    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)
        n = len(batch.x)/batch.num_graphs
        tot_nodes += len(batch.x)
        layers = int(layer_fraction * n) 
        with torch.no_grad():
            pred = model(batch, layers)

        loss += criterion(pred, batch.y.to(torch.long)).item() * batch.num_graphs

        y_pred = torch.argmax(pred,dim=1)
        acc += torch.sum(y_pred == batch.y)
    acc = acc.item()
    return (loss/len(loader.dataset), acc/tot_nodes)
    


def get_class_imbalance(dataset, num_classes = 2):
    tot = torch.tensor([0.0 for i in range(num_classes)])
    for data in dataset:
        tot += torch.bincount(data.y.to(torch.int),minlength = num_classes)
    x = torch.div(torch.sum(tot), tot)
    #x = torch.sum(tot)/tot
    x= torch.div(x,num_classes)
    #print(x)
    return x

def main(args, cluster=None):
    hash_object = hashlib.sha256(json.dumps(args.__getstate__()).encode())
    model_string = hash_object.hexdigest()
    setattr(args, 'model_string', model_string)
    print(json.dumps(args.__getstate__()))
    if not args.verbose:
        tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    random.seed(0)
    np.random.seed(0)

    name = TENSORBOARD_DIRECTORY + f'{datetime.datetime.now()}'+'Run' + args.conv + '_skipP_' + str(args.skip_previous) + '_skipI_' + str(args.skip_input) + '_emb_' + str(args.hidden_channels) + '_dropout_' + str(args.dropout) + '_decay_' + str(args.use_weight_decay) + '_hidden_state_factor_' + str(args.hidden_state_factor) + '_dataset_' + args.dataset + '_l1_' + str(args.use_l1)

    if args.use_tensorboard:
        writer = SummaryWriter(name)
    
    val_split = 0.8

    if args.dataset == 'tree-path':
        dataclass = datasets.Trees()
        dataset = dataclass.makedata(num_graphs=200, num_nodes=10)
        bigger_trees = dataclass.makedata(num_graphs=10, num_nodes=100)

        np.random.shuffle(dataset)
        train_size = int(val_split*len(dataset))

        train_loader = DataLoader(dataset[:train_size], batch_size=args.batch_size, shuffle=True)
        valid_loader = DataLoader(dataset[train_size:], batch_size=1, shuffle=False)
        bigger_graph_loader = DataLoader(bigger_trees, batch_size=1, shuffle=False)

        in_channels = dataclass.num_features
        out_channels = dataclass.num_classes
        train_layer_fraction = 1.2

    elif args.dataset == 'distance':
        dataclass = datasets.Distance()

        dataset = dataclass.makedata(num_graphs = 200, num_nodes = 10)
        bigger_cycles = dataclass.makedata(num_graphs = 10, num_nodes = 100)

        np.random.shuffle(dataset)
        train_size = int(val_split*len(dataset))

        train_loader = DataLoader(dataset[:train_size], batch_size=args.batch_size, shuffle=True)
        valid_loader = DataLoader(dataset[train_size:], batch_size=1, shuffle=False)
        bigger_graph_loader = DataLoader(bigger_cycles, batch_size=1, shuffle=False)

        in_channels = dataclass.num_features
        out_channels = dataclass.num_classes
        train_layer_fraction = 1.2

    elif args.dataset == 'prefix':
        dataclass = datasets.PrefixSum()
        dataset = dataclass.makedata(num_graphs = 400, num_nodes = 10)
        bigger_cycles = dataclass.makedata(num_graphs = 10, num_nodes = 100)

        np.random.shuffle(dataset)
        train_size = int(val_split*len(dataset))

        train_loader = DataLoader(dataset[:train_size], batch_size=args.batch_size, shuffle=True)
        valid_loader = DataLoader(dataset[train_size:], batch_size=1, shuffle=False)
        bigger_graph_loader = DataLoader(bigger_cycles, batch_size=1, shuffle=False)
        
        in_channels = dataclass.num_features
        out_channels = dataclass.num_classes

        train_layer_fraction = 1.2

    elif args.dataset == 'prefix-k':
        dataclass = datasets.PrefixSumK(k = 5, inp = 5)
        dataset = dataclass.makedata(num_graphs = 1000, num_nodes = 10)
        bigger_cycles = dataclass.makedata(num_graphs = 10, num_nodes = 100)

        np.random.shuffle(dataset)
        train_size = int(val_split*len(dataset))

        train_loader = DataLoader(dataset[:train_size], batch_size=args.batch_size, shuffle=True)
        valid_loader = DataLoader(dataset[train_size:], batch_size=1, shuffle=False)
        bigger_graph_loader = DataLoader(bigger_cycles, batch_size=1, shuffle=False)
        
        in_channels = dataclass.num_features
        out_channels = dataclass.num_classes

        train_layer_fraction = 1.2

    elif args.dataset == 'midpoint':
        dataclass = datasets.MidPoint()
        dataset = dataclass.makedata(num_graphs = 200, num_nodes=20, allow_sizes=True)
        bigger_paths = dataclass.makedata(num_graphs = 10, num_nodes=200)

        np.random.shuffle(dataset)
        train_size = int(val_split*len(dataset))

        train_loader = DataLoader(dataset[:train_size], batch_size=args.batch_size, shuffle=True)
        valid_loader = DataLoader(dataset[train_size:], batch_size=1, shuffle=False)
        bigger_graph_loader = DataLoader(bigger_paths, batch_size=1, shuffle=False)
        
        in_channels = dataclass.num_features
        out_channels = dataclass.num_classes
        
        train_layer_fraction = 1.2



    class_imbalance_weight = get_class_imbalance(dataset, out_channels)

    seed = args.run_number
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if args.normalization == 'LayerNorm':
        normalization_function = torch.nn.LayerNorm
    elif args.normalization == 'BatchNorm':
        normalization_function = torch.nn.BatchNorm1d
    else:
        print('Unrecognized normalization function: ' + args.normalization)
        exit(1)

    model = RecGNN(in_channels, args.hidden_channels, out_channels, args.hidden_state_factor, args.dropout, conv=args.conv, skip_prev=args.skip_previous, skip_input=args.skip_input, aggregation=args.aggregation, gumbel=args.gumbel, normalization=normalization_function).to(device)
    
    if args.use_weight_decay:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler_lambda = lambda epoch: 0.7 ** epoch
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                       factor=0.7, patience=20,
                                                       min_lr=0.00001)

    best_valid_loss = np.Inf
    best_generalization_loss = np.Inf
        
    for epoch in range(1, 1 + args.epochs):
        if args.verbose:
            print("=====Epoch {}".format(epoch))
            print('Training...')

        loss, l1_loss, l2_loss = train(model=model, device=device, loader=train_loader, optimizer = optimizer, class_imbalance_weight=class_imbalance_weight, train_layer_fraction=train_layer_fraction, 
                                use_l1 = args.use_l1, l1_weight = args.l1_weight, use_l2=args.use_l2, l2_weight = args.l2_weight, use_entropy_loss = args.use_entropy_loss 
                                )
        if args.verbose:
            print('Evaluating...')
        train_loss, train_acc = test(model, device, train_loader, train_layer_fraction)
        valid_loss, valid_acc = test(model, device, valid_loader, train_layer_fraction)
        generalization_loss, generalization_acc = test(model, device, bigger_graph_loader, train_layer_fraction)

        print(json.dumps({'run': args.run_number, 'epoch': epoch, 'lr': optimizer.param_groups[0]['lr'], 'optimization_l1': l1_loss, 'optimization_l2': l2_loss, 'optimization_loss': loss, 'train_loss': train_loss, 'train_acc': train_acc, 'valid_loss': valid_loss, 'valid_acc': valid_acc, 'generalization_loss': generalization_loss, 'generalization_acc': generalization_acc}))

        if args.use_tensorboard:
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


        if args.store_models:
            if valid_loss <= best_valid_loss:
                best_valid_loss = valid_loss
                file_name = MODEL_DIRECTORY + 'model_train_' + model_string
                if os.path.exists(file_name):
                    os.remove(file_name)
                torch.save(model.state_dict(), file_name)
                if args.verbose:
                    print('Saved new train model')

            if generalization_loss <= best_generalization_loss:
                best_generalization_loss = generalization_loss
                file_name = MODEL_DIRECTORY + 'model_generalization_' + model_string
                if os.path.exists(file_name):
                    os.remove(file_name)
                torch.save(model.state_dict(), file_name)
                if args.verbose:
                    print('Saved new generalization model')

            if epoch == args.epochs:
                file_name = MODEL_DIRECTORY + 'model_last_' + model_string
                if os.path.exists(file_name):
                    os.remove(file_name)
                torch.save(model.state_dict(), file_name)
                if args.verbose:
                    print('Saved last model')

        if args.use_scheduler:
            scheduler.step(valid_loss)

    print('Done')


if __name__ == "__main__":
    parser = HyperOptArgumentParser(strategy='grid_search')
    parser.add_argument('--device', type=int, default=0)
    parser.opt_list('--hidden_channels', type=int, default=8, tunable=True, options=[8, 16])
    parser.opt_list('--dropout', type=float, default=0.1, tunable=True, options=[0.2])
    parser.opt_list('--run_number', type=int, default=0, tunable=True, options=[0, 1, 2, 3, 4])
    parser.opt_list('--lr', type=float, default=0.0004, tunable=True, options=[0.0004])
    parser.add_argument('--epochs', type=int, default=1000)
    parser.opt_list('--batch_size', type=int, default=1, tunable=True, options=[1])
    parser.opt_list('--conv', type=str, default='gru', tunable=True, options=['gin-mlp', 'gru-mlp', 'gin', 'gru'])
    parser.opt_list('--skip_previous', type=bool, default=False, tunable=True, options=[False])
    parser.opt_list('--skip_input', type=bool, default=False, tunable=True, options=[True])
    parser.opt_list('--hidden_state_factor', type=float, default=2, tunable=True, options=[4])
    parser.opt_list('--use_weight_decay', type=bool, default=False, tunable=True, options=[False])
    parser.opt_list('--use_scheduler', type=bool, default=False, tunable=True, options=[True])
    parser.opt_list('--use_l1', type=bool, default=False, tunable=True, options=[False])
    parser.opt_list('--l1_weight', type=float, default=0.0001, tunable=True, options=[0.0001])
    parser.opt_list('--use_l2', type=bool, default=False, tunable=True, options=[True, False])
    parser.opt_list('--l2_weight', type=float, default=0.0001, tunable=True, options=[0.0001])
    parser.opt_list('--dataset', type=str, default='tree-path', tunable=True, options=['prefix', 'tree-path', 'distance'])
    parser.opt_list('--gumbel', type=bool, default=False, tunable=True, options=[False])
    parser.opt_list('--aggregation', type=str, default='add', tunable=True, options=['add', 'max'])
    parser.opt_list('--use_entropy_loss', type=bool, default=False, tunable=True, options=[True, False])
    parser.opt_list('--normalization', type=str, default='LayerNorm', tunable=True, options=['BatchNorm', 'LayerNorm'])
    parser.add_argument('--store_models', type=bool, default=False)
    parser.add_argument('--use_tensorboard', type=bool, default=False)
    parser.add_argument('--slurm', action='store_true', default=False)
    parser.add_argument('--verbose', action='store_true', default=False)
    args = parser.parse_args()
    if args.slurm:
        print('Launching SLURM jobs')
        cluster = SlurmCluster(
            hyperparam_optimizer=args,
            log_path='slurm_log/',
            python_cmd='python'
        )
        cluster.job_time = '48:00:00'

        args.filename = ""
        cluster.memory_mb_per_node = '1G'
        job_name = 'RecGNN_test'
        cluster.per_experiment_nb_cpus = 1
        cluster.per_experiment_nb_gpus = 0
        #cluster.add_slurm_cmd(cmd='gres', value='gpu:0', comment='Specify gpu type')
        cluster.optimize_parallel_cluster_gpu(main, nb_trials=None, job_name=job_name, job_display_name='RecGNN')
    else:
        main(args)
