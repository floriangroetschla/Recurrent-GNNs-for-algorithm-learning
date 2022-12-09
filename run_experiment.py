from test_tube import HyperOptArgumentParser
from test_tube.hpc import SlurmCluster
from attrdict import AttrDict
import json
import hashlib


from train import train_and_eval

if __name__ == "__main__":
    parser = HyperOptArgumentParser(strategy='grid_search')
    parser.add_argument('--device', type=int, default=0)
    parser.opt_list('--hidden_dimension', type=int, default=8, tunable=True, options=[16, 32])
    parser.opt_list('--dropout', type=float, default=0.1, tunable=True, options=[0.0, 0.2])
    parser.opt_list('--run_number', type=int, default=0, tunable=True, options=[0, 1])
    parser.opt_list('--lr', type=float, default=0.0004, tunable=True, options=[0.0004])
    parser.add_argument('--epochs', type=int, default=1000)
    parser.opt_list('--batch_size', type=int, default=1, tunable=True, options=[1])
    parser.opt_list('--conv', type=str, default='gru', tunable=True, options=['gin-mlp', 'gru-mlp'])
    parser.opt_list('--skip_previous', type=bool, default=False, tunable=True, options=[False])
    parser.opt_list('--skip_input', type=bool, default=False, tunable=True, options=[True])
    parser.opt_list('--hidden_state_factor', type=float, default=2, tunable=True, options=[4])
    parser.opt_list('--use_weight_decay', type=bool, default=False, tunable=True, options=[False])
    parser.opt_list('--use_scheduler', type=bool, default=False, tunable=True, options=[True])
    parser.opt_list('--use_l1', type=bool, default=False, tunable=True, options=[False])
    parser.opt_list('--l1_weight', type=float, default=0.0001, tunable=True, options=[0.0001])
    parser.opt_list('--use_l2', type=bool, default=False, tunable=True, options=[True])
    parser.opt_list('--l2_weight', type=float, default=0.0001, tunable=True, options=[0.0001])
    parser.opt_list('--dataset', type=str, default='tree-path', tunable=True, options=['prefix-k'])
    parser.opt_list('--gumbel', type=bool, default=False, tunable=True, options=[False])
    parser.opt_list('--aggregation', type=str, default='add', tunable=True, options=['add', 'max'])
    parser.opt_list('--use_entropy_loss', type=bool, default=False, tunable=True, options=[True, False])
    parser.opt_list('--normalization', type=str, default='LayerNorm', tunable=True, options=['BatchNorm', 'LayerNorm'])
    parser.opt_list('--activation_function', type=str, default='ReLU', tunable=True, options=['ReLU', 'SiLU', 'LeakyReLU', 'Sigmoid'])
    parser.add_argument('--store_models', type=bool, default=False)
    parser.add_argument('--use_tensorboard', type=bool, default=False)
    parser.add_argument('--slurm', action='store_true', default=False)
    parser.add_argument('--verbose', action='store_true', default=False)
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--num_train_graphs', type=int, default=100)
    parser.add_argument('--num_train_nodes', type=int, default=10)
    parser.add_argument('--num_generalize_graphs', type=int, default=10)
    parser.add_argument('--num_generalize_nodes', type=int, default=100)
    parser.add_argument('--train_layer_fraction', type=float, default=1.2)
    parser.add_argument('--vary_layers', type=bool, default=False)
    parser.add_argument('--baseline', type=str, default="")
    parser.add_argument('--model_name', type=str, default=None)

    args = parser.parse_args()

    if args.model_name is None:
        hash_object = hashlib.sha256(json.dumps(args.__getstate__()).encode())
        model_hash = hash_object.hexdigest()[-20:]
        model_name = model_hash
    else:
        model_name = args.model_name

    if args.config is not None:
        with open(args.config, 'r') as f:
            args = AttrDict(json.load(f))

    args.model_name = model_name

    print(args)
    
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
        job_name = 'RecGNN'
        cluster.per_experiment_nb_cpus = 1
        cluster.per_experiment_nb_gpus = 0
        #cluster.add_slurm_cmd(cmd='gres', value='gpu:0', comment='Specify gpu type')
        cluster.optimize_parallel_cluster_gpu(main, nb_trials=None, job_name=job_name, job_display_name='RecGNN')
    else:
        train_and_eval(args)
