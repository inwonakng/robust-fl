from typing import List,Union
from models import Trainer
from update import Update

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
            pass
            

    def __call__(self,global_model:Trainer, updates:List[Update]) -> None:
        to_update_global = None
        if len(updates) > 0:
            to_update_global = self.aggregate(global_model, updates)
        self.validate(to_update_global)
    