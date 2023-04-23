from typing import List, Tuple
import torch
import logging

from models import Trainer
from update import Update
from .utils import normalize_weights

class Aggregator:
    def __init__(
        self, 
        staleness_lambda: float = 0,
    ) -> None:
        self.staleness_lambda = staleness_lambda

    def aggregate(
        self,
        global_model:Trainer, 
        client_weights: List[dict],
        update_weights: torch.Tensor,
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
        client_weights, update_weights, update_delays = [], [], []
        
        for u in updates:
            client_weights.append(list(u.new_state.values()))
            update_weights.append(u.train_size)
            update_delays.append(u.counter)
        
        device = client_weights[0][0].device

        update_weights = normalize_weights(update_weights).to(device)
        update_delays = torch.tensor(update_delays).long().to(device)

        if self.staleness_lambda > -1:
            # Our simpler staleness weighting
            staleness_weights = torch.ones(len(updates)).to(device) * cur_epoch + 1e-8
            staleness_weights -= update_delays * self.staleness_lambda
            # before normalization, the values of staleness_weights must be nonnnegative
            staleness_weights += staleness_weights.min()
            staleness_weights = normalize_weights(staleness_weights).to(device)

            update_weights = normalize_weights(update_weights + staleness_weights)
        else:
            # reject delayed inputs
            accept_mask = update_delays == 0
            logging.debug(f'Aggregator -- {(~accept_mask).sum().item()} delayed updates out of {len(accept_mask)} total updates are rejected.')

            update_weights = update_weights[accept_mask]
            client_weights = [p for p, accept in zip(client_weights, accept_mask.tolist()) if accept]

        return client_weights, update_weights

    def __call__(self,cur_epoch: int, global_model:Trainer, updates:List[Update]) -> dict:
        logging.debug(f'Aggregator -- received {len(updates)} updates to aggregate')
        client_weights, update_weights = self.parse_updates(cur_epoch, updates)
        to_update_global = None
        if len(client_weights) > 0:
            to_update_global = self.aggregate(global_model, client_weights, update_weights,)
        self.validate(to_update_global)
        return to_update_global
    
