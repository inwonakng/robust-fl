from typing import List, Tuple
import torch
import logging

from models import Trainer
from update import Update
from .utils import normalize_weights

class Aggregator:
    def __init__(
        self, 
        staleness_lambda: bool = 0,
    ) -> None:
        self.staleness_lambda = staleness_lambda

    def aggregate(
        self,
        cur_epoch: int, 
        global_model:Trainer, 
        updates: List[Update]
    ) -> None:
        raise NotImplementedError
    
    def validate(
        self, 
        to_update_global
    ) -> None:
        if to_update_global is not None:
            for v in to_update_global.values():
                if v.isnan().any():
                    raise Exception('Aggregation results contain NaN')
                
    def parse_updates(
        self,
        cur_epoch: int,
        updates:List[Update]
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        points, update_weights, update_delays = [], [], []
        
        for u in updates:
            points.append(list(u.new_state.values()))
            update_weights.append(u.train_size)
            update_delays.append(u.counter)
        
        device = points[0][0].device

        update_weights = normalize_weights(update_weights).to(device)
        update_delays = torch.tensor(update_delays).long().to(device)

        # Our simpler staleness weighting
        staleness_weights = torch.ones(len(updates)).to(device) * cur_epoch + 1e-8
        staleness_weights -= update_delays * self.staleness_lambda
        # before normalization, the values of staleness_weights must be nonnnegative
        staleness_weights += staleness_weights.min()
        staleness_weights = normalize_weights(staleness_weights).to(device)

        update_weights = normalize_weights(update_weights + staleness_weights)

        return points, update_weights

    def __call__(self,cur_epoch: int, global_model:Trainer, updates:List[Update]) -> None:
        logging.debug(f'Aggregator -- received {len(updates)} updates to aggregate')
        to_update_global = None
        if len(updates) > 0:
            to_update_global = self.aggregate(cur_epoch, global_model, updates)
        self.validate(to_update_global)
        return to_update_global
    
