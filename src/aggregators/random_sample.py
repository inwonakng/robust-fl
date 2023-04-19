from typing import List
import torch

from models import Trainer
from update import Update
from aggregators import Aggregator

from .utils import weighted_average
from .utils import random_sample_average


# class FedAvg(Aggregator):
#     def __init__(
#         self,
#     ) -> None: 
#         super(FedAvg, self).__init__()
#     
#     def aggregate(
#         self,
#         global_model: Trainer,
#         updates:List[Update],
#     ) -> dict:
#         new_global_state = global_model.get_state()
#         model_weights, update_weights = self.parse_updates(updates)
# 
#         # print(update_weights.sum())
# 
#         for key, components in zip(new_global_state.keys(), zip(*model_weights)):
#             new_global_state[key] = weighted_average(components, update_weights)
#         return new_global_state


class random_sample_simple(Aggregator):
    def __init__(
        self,
        select_p: float,
    ) -> None: 
        super(random_sample_simple, self).__init__()
        self.select_p = select_p
    
    def aggregate(
        self,
        global_model: Trainer,
        updates:List[Update],
    ) -> dict:
        new_global_state = global_model.get_state()
        model_weights, update_weights = self.parse_updates(updates)

        # make the new set of weights

        for key, components in zip(new_global_state.keys(), zip(*model_weights)):
            # take a random sample
            stacked_components = torch.stack(components)
            new_global_state[key] = random_sample_average(stacked_components, new_global_state[key], self.select_p)


        # print(update_weights.sum())
        return new_global_state

