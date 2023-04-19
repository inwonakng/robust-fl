from typing import List
import torch

from models import Trainer
from update import Update
from aggregators import Aggregator

from .utils import weighted_average


class FedAvg(Aggregator):
    def __init__(
        self,
    ) -> None: 
        super(FedAvg, self).__init__()
    
    def aggregate(
        self,
        global_model: Trainer,
        updates:List[Update],
    ) -> dict:
        new_global_state = global_model.get_state()
        model_weights, update_weights = self.parse_updates(updates)


        # print(update_weights.sum())

        for key, components in zip(new_global_state.keys(), zip(*model_weights)):
            print(components)
            new_global_state[key] = weighted_average(components, update_weights)
        return new_global_state

