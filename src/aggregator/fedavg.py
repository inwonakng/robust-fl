from typing import List
import torch

from model import Model
from update import Update
from aggregator import Aggregator


class FedAvg(Aggregator):
    def __init__(
        self,
        # global_model: Model
    ) -> None: 
        super(FedAvg, self).__init__()
        # self.max_staleness = max_staleness
    
    def aggregate(
        self,
        global_model: Model,
        updates:List[Update],
    ) -> dict:
        
        new_global_state = global_model.state_dict()
        for k in new_global_state:
            new_global_state[k] = torch.stack([
                u.new_state[k]
                for u in updates
            ]).mean(0)

        return new_global_state

