from typing import Union
import torch

from aggregators import Aggregator
from aggregators.fedavg import FedAvg

from models import Model
from models.mlp import MLP

from dataset import load_MNIST


MODEL_MAPPING = {
    "MLP": MLP,
}

AGG_MAPPING = {
    "FedAvg": FedAvg,
}

DATASET_MAPPING = {
    "MNIST": load_MNIST,
}


def load_model(model_name:str,model_args:dict) -> Model:
    return Model(MODEL_MAPPING[model_name],**model_args)

def load_aggregator(agg_name:str,agg_args:dict) -> Aggregator:
    return AGG_MAPPING[agg_name](**agg_args)

def load_dataset(dataset_name:str) -> Union[torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor]:
    return DATASET_MAPPING[dataset_name]()
