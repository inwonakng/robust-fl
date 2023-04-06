from typing import Union
import torch

from aggregators import Aggregator, AGG_MAPPING
from models import Trainer, MODEL_MAPPING
from dataset import DATASET_MAPPING


def load_trainer(model_name:str,model_args:dict) -> Trainer:
    return Trainer(MODEL_MAPPING[model_name],**model_args)

def load_aggregator(agg_name:str,agg_args:dict) -> Aggregator:
    return AGG_MAPPING[agg_name](**agg_args)

def load_dataset(dataset_name:str) -> Union[torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor]:
    return DATASET_MAPPING[dataset_name]()
