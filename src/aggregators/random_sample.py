from typing import List
import torch

from models import Trainer
from update import Update
from aggregators import Aggregator

from .utils import weighted_average
# from .utils import random_sample_average

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
        cur_epoch: int,
        global_model: Trainer,
        updates:List[Update],
    ) -> dict:
        new_global_state = global_model.get_state()
        model_weights, update_weights = self.parse_updates(cur_epoch, updates)

        # make the new set of weights

        for key, components in zip(new_global_state.keys(), zip(*model_weights)):
            # take a random sample
            stacked_components = torch.stack(components)
            weighted_stacked_components = (stacked_components * update_weights.reshape((len(update_weights),1,1)))

            selected_params = torch.zeros(weighted_stacked_components.size())
            rand_mask = torch.rand(weighted_stacked_components.size()) < self.select_p
            selected_params[rand_mask] = weighted_stacked_components[rand_mask]

            keep_original_mask = rand_mask.sum(0) == 0
            final_params = selected_params.sum(0)
            final_params[~keep_original_mask] = final_params[~keep_original_mask] / rand_mask.sum(0)[~keep_original_mask]
            final_params[keep_original_mask] = new_global_state[key][keep_original_mask]

            new_global_state[key] = final_params
            # new_global_state[key] = random_sample_average(stacked_components, new_global_state[key], self.select_p)

        return new_global_state

