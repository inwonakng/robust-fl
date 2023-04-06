from .aggregators import *
from .fedavg import FedAvg

AGG_MAPPING = {
    "FedAvg": FedAvg,
}