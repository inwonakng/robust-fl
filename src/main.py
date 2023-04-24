import torch
import yaml
from argparse import ArgumentParser
from pathlib import Path
import warnings

from simulator import Simulator

# comment this line if you want to see the warnings
warnings.filterwarnings("ignore")

parser = ArgumentParser()
parser.add_argument('-c', '--config_file', required=True, type=str)
parser.add_argument('-l', '--staleness_lambda', default=-1, type=float)
parser.add_argument('-g', '--staleness_gamma', default=1, type=float)
parser.add_argument('-e', '--sim_epoch', default=100, type=int)
parser.add_argument('-o', '--overwrite', action='store_true')
args = parser.parse_args()

# set the default device
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# torch.set_default_device(device)

run_config = yaml.safe_load(open(args.config_file))
learning_config_file = Path(f"configurations/learning/{run_config['learning_setting']}.yaml")
learning_config = yaml.safe_load(open(learning_config_file))

for agg_config in run_config['aggregator_settings']:
    agg_config_file = Path(f"configurations/aggregators/{agg_config}.yaml")
    agg_config = yaml.safe_load(open(agg_config_file))
    agg_config['agg_args']['staleness_lambda'] = args.staleness_lambda
    agg_config['agg_args']['staleness_gamma'] = args.staleness_gamma
    for client_config in run_config['client_settings']:
        client_config_file = Path(f"configurations/clients/{client_config}.yaml")
        client_config = yaml.safe_load(open(client_config_file))


        sim = Simulator(
            **agg_config,
            **client_config,
            **learning_config,
            output_dir = (
                Path('output')
                / agg_config_file.stem 
                / learning_config_file.stem 
                / client_config_file.stem
                / f'staleness_lambda_{args.staleness_lambda}'.replace('.','_')
            ),
        )


        # print()
        sim.run(
            n_epoch = args.sim_epoch,
            overwrite = args.overwrite
        )