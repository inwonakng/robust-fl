from .aggregators import *
from .fedavg import FedAvg
from .rfa import RFA

AGG_MAPPING = {
    "FedAvg": FedAvg,
    "RFA": RFA,
}