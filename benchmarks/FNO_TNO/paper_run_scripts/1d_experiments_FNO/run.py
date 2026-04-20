import sys
sys.path.append('../../')
sys.path.append('../')
from models.FNO1D.runner import Runner
from utils import dict_combiner
import argparse

# use argparse to get command line argument for which experiment to run
parser = argparse.ArgumentParser()
parser.add_argument('--project_name', type=str, default='DeepNewtonOperator_Burgers')
parser.add_argument('--id', type=int, default=0)
args = parser.parse_args()

# once working, need to update readme with:
# conda install -c anaconda scikit-learn

# build a dict of experimental conditions
exp_dict = {
    'project_name': [args.project_name],
    # data settings
    'split_frac': [{'train': 0.95, 'val': 0.05}],
    'random_state': [0],
    'domain_dim': [1], # 1 for timeseries, 2 for 2D spatial
    'train_sample_rate': [1],
    'test_sample_rates': [[1]],
    'input_dim': [1],
    'output_dim': [151],
    'batch_size': [1],
    'dyn_sys_name': ['Burgers'],
    # optimizer settings
    'learning_rate': [1e-3, 8e-4, 6e-4],
    'dropout': [1e-4],
    'lr_scheduler_params': [
                            {'patience': 2, 'factor': 0.5},
                             ],
    'max_epochs': [100],
    'monitor_metric': ['loss/val/mse'],
    # model settings (modest model size for debugging)
    'd_model': [96],
    'modes': [[12]],
    'num_layers': [6],
    'activation': ['gelu'],
    'gradient_clip_val':[None] #or 10.0 whenever want to use
}

exp_list = dict_combiner(exp_dict)

# Print the length of the experiment list
print('Number of experiments to sweep: ', len(exp_list))

# run the experiment
Runner(**exp_list[args.id])