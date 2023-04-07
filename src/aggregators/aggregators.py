from typing import List,Union
from models import Trainer
from update import Update

class Aggregator:
    def __init__(self) -> None:
        pass

    def aggregate(self, global_model:Trainer, updates: List[Update]) -> None:
        raise NotImplementedError

    def __call__(self,global_model:Trainer, updates:List[Update]) -> None:
        if len(updates) > 0:
            return self.aggregate(global_model, updates)
        else: 
            return None
    