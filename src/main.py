import yaml
from argparse import ArgumentParser
import logging

from simulator import Simulator


parser = ArgumentParser()
parser.add_argument('-C','--config',required=True)
args = parser.parse_args()

config = yaml.safe_load(open(args.config))
sim = Simulator(**config)
sim.run(config['sim_epoch'])