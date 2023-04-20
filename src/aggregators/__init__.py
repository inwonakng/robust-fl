from .aggregators import *
from .fedavg import FedAvg
from .rfa import RFA
from .clusteragg import ClusterAgg
from .random_sample import RandomSampleSimple

AGG_MAPPING = {
    "FedAvg": FedAvg,
    "RFA": RFA,
    "ClusterAgg": ClusterAgg,
    "RandomSampleSimple": RandomSampleSimple,
}
