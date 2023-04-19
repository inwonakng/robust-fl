import yaml
from argparse import ArgumentParser
from pathlib import Path
from simulator import Simulator


parser = ArgumentParser()
parser.add_argument('--cfg',default="../configurations/config.yaml")
args = parser.parse_args()

config = yaml.safe_load(open(Path(args.cfg)))

sim = Simulator(cfg=config)
