import torch
import yaml
from argparse import ArgumentParser
from pathlib import Path
import warnings

from simulator import Simulator

# comment this line if you want to see the warnings
warnings.filterwarnings("ignore")

# if __name__ == '__main__':
parser = ArgumentParser()
parser.add_argument('-C', '--config_file', required=True)
parser.add_argument('-L', '--staleness_lambda', required=True, type=int)
parser.add_argument('-E', '--sim_epoch', default=100)
args = parser.parse_args()

run_config = yaml.safe_load(open(parser.config_file))
learning_config_file = f"configurations/learning/{run_config['learning_setting']}.yaml"
learning_config = yaml.safe_load(open(learning_config_file))

for agg_config in run_config['aggregator_settings']:
    agg_config_file = f"configurations/aggregators/{agg_config}.yaml"
    agg_config = yaml.safe_load(open(agg_config_file))
    agg_config['staleness_lambda'] = args.staleness_lambda
    for client_config in run_config['client_settings']:
        client_config_file = f"configurations/clients/{client_config}.yaml"
        client_config = yaml.safe_load(open(client_config_file))


        sim = Simulator(
            **agg_config,
            **client_config,
            **learning_config,
            # **model_config,
            output_dir = Path('output') / agg_config_file.stem / learning_config_file.stem / client_config_file.stem
        )

        # set the default device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.set_default_device(device)

        # print()
        sim.run(
            n_epoch = args.sim_epoch
        )