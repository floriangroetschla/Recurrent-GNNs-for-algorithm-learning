import json
import itertools
import os
from tqdm import tqdm

configs = ['experiment_config_recGNN.json', 'experiment_config_gin_baseline.json', 'experiment_config_itergnn_baseline.json']
output_dirs = ['configs_recGNN', 'configs_gin', 'configs_itergnn']

for config, output_dir in zip(configs, output_dirs):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    with open(config, 'r') as f:
        args = json.load(f)
        configs = list(itertools.product(*args.values()))
        for i, config in tqdm(enumerate(configs)):
            config_dict = {}
            for key, value in zip(args.keys(), config):
                config_dict[key] = value
            with open(os.path.join(output_dir, f'config_{i}.json'), 'w') as fp:
                json.dump(config_dict, fp)

