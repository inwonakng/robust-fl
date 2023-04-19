from torch.cuda import is_available
import yaml
from argparse import ArgumentParser
import logging
from pathlib import Path
import warnings
import torch

from simulator import Simulator

# comment this line if you want to see the warnings
warnings.filterwarnings("ignore")


parser = ArgumentParser()
parser.add_argument('--agg_config',required=True)
parser.add_argument('--client_config',required=True)
parser.add_argument('--data_config',default='configurations/datasets/mnist_0_1.yaml')
# parser.add_argument('--model_config',default='configurations/models/mlp.yaml')
parser.add_argument('--sim_epoch', default=100)
args = parser.parse_args()

agg_config_file = Path(args.agg_config)
agg_config = yaml.safe_load(open(agg_config_file))

client_config_file = Path(args.client_config)
client_config = yaml.safe_load(open(client_config_file))

data_config_file = Path(args.data_config)
data_config = yaml.safe_load(open(data_config_file))

# model_config_file = Path(args.model_config)
# model_config = yaml.safe_load(open(model_config_file))



sim = Simulator(
    **agg_config,
    **client_config,
    **data_config,
    # **model_config,
    output_dir = Path('output') / agg_config_file.stem / data_config_file.stem / client_config_file.stem
)

# set the default device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)
sim.run(
    n_epoch = args.sim_epoch
)
