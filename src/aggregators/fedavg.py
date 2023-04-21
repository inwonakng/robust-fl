from typing import List
import torch

from models import Trainer
from update import Update
from aggregators import Aggregator

from .utils import weighted_average


class FedAvg(Aggregator):
    def __init__(
        self,
        **kwargs,
    ) -> None: 
        super(FedAvg, self).__init__(**kwargs)
    
    def aggregate(
        self,
        cur_epoch: int,
        global_model: Trainer,
        updates:List[Update],
    ) -> dict:
        new_global_state = global_model.get_state()
        client_weights, update_weights = self.parse_updates(cur_epoch, updates)

        for key, components in zip(new_global_state.keys(), zip(*client_weights)):
            new_global_state[key] = weighted_average(components, update_weights)
        return new_global_state

