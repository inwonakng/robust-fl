from .aggregators import *
from .fedavg import FedAvg
from .rfa import RFA
from .clusteragg import ClusterAgg

AGG_MAPPING = {
    "FedAvg": FedAvg,
    "RFA": RFA,
    "ClusterAgg": ClusterAgg
}