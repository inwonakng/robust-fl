from typing import List
import torch

from models import Trainer
from update import Update
from aggregators import Aggregator

from .utils import weighted_average
from .utils import random_sample_average

class RandomSampleSimple(Aggregator):
    def __init__(
        self,
        select_p: float,
        **kwargs,
    ) -> None: 
        super(RandomSampleSimple, self).__init__(**kwargs)
        self.select_p = select_p
    
    def aggregate(
        self,
        global_model: Trainer,
        client_weights: List[dict],
        update_weights: torch.Tensor,
    ) -> dict:
        new_global_state = global_model.get_state()
        # client_weights, update_weights = self.parse_updates(cur_epoch, updates)

        # make the new set of weights
        for key, components in zip(new_global_state.keys(), zip(*client_weights)):
            if not new_global_state[key].size(): continue
            # take a random sample
            stacked_components = torch.stack(components)

            # weight them using given weights
            # weighted_stacked_components = (
            #     stacked_components * 
            #     update_weights.reshape([len(update_weights)] + [1] * (stacked_components.dim()-1))
            # )
            new_global_state[key] = random_sample_average(stacked_components, new_global_state[key], self.select_p)

        return new_global_state

