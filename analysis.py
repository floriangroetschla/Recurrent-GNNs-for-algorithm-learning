import pandas

file = 'output.csv'
data = pandas.read_csv(file)
import itertools




# what information do I need for a specific run?
# if generalized
# last generalized
# min. val loss
# l2 loss etc

x = {'Dropout': 0.2}


# get all the columns where we set params

params = ['dropout', 'lr', '']
params = ['hidden_channels', 'dropout', 'epochs', 'batch_size', 'conv',
         'skip_previous', 'skip_input','hidden_state_factor', 'use_weight_decay',
          'use_scheduler', 'use_l1','l1_weight', 'use_l2', 'l2_weight', 'dataset']

measures = ['epoch', 'optimization_l1', 'optimization_l2', 'optimization_loss',
       'train_loss', 'train_acc', 'valid_loss', 'valid_acc',
       'generalization_loss', 'generalization_acc']

results = []
params_values = {param:data[param].unique() for param in params}
keys = list(params_values)

run_config = [dict(zip(keys,values)) for values in itertools.product(*map(params_values.get, keys))]

run = run_config[0]
query_str = ' && '.join([ f'data[{param}] == {val}' for param, val in run.items()])

#from functools import reduce
#import numpy as np
#l = [data[param] == val for param,val in run.items()]
#x = data[reduce(np.logical_and, l), params]

#x = data.loc[, measures]
import numpy as np

for run in run_config:
    if run['dataset'] != 'midpoint' or run['conv'] != 'gin': #or run['dropout'] != 0.2 or run['skip_previous'] != False or run['hidden_state_factor'] == 1:
        continue
    print(run)
    x = data.loc[np.prod([data[k] == v for k,v in run.items()],axis=0).astype(bool),measures]
    print(x['generalization_acc'].max())

    print(x)

#data[(data.dataset == 'midpoint')&(data.conv == 'gin-')]