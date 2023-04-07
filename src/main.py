import yaml
from argparse import ArgumentParser

from simulator import Simulator

parser = ArgumentParser()
parser.add_argument('-C','--config',required=True)
args = parser.parse_args()
config = yaml.safe_load(open(args.config))

sim = Simulator(**config)
sim.run(
    n_epoch = config['sim_epoch'],        
)