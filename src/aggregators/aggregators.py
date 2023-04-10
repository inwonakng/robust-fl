from typing import List, Tuple
import torch

from models import Trainer
from update import Update

from .utils import normalize_weights

class Aggregator:
    def __init__(self) -> None:
        pass

    def aggregate(self, global_model:Trainer, updates: List[Update]) -> None:
        raise NotImplementedError
    
    def validate(self, to_update_global) -> None:
        if to_update_global is not None:
            for v in to_update_global.values():
                if v.isnan().any():
                    raise Exception('Aggregation results contain NaN')
                
    def parse_updates(
        self,
        updates:List[Update]
    ) -> Tuple[List[torch.Tensor], torch.tensor, List[int]]:
        points, update_weights, update_delays = [], [], []
        for u in updates:
            points.append(list(u.new_state.values()))
            update_weights.append(u.train_size)
            update_delays.append(u.counter)

        update_weights = normalize_weights(update_weights).to(points[0][0].device)

        # TODO: do something wtih the delays to modify the model weights. 

        return points, update_weights

    def __call__(self,global_model:Trainer, updates:List[Update]) -> None:
        to_update_global = None
        if len(updates) > 0:
            to_update_global = self.aggregate(global_model, updates)
        self.validate(to_update_global)
        return to_update_global
    