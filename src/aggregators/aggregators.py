from typing import List,Union
from models import Model
from update import Update

class Aggregator:
    def __init__(self) -> None:
        pass

    def aggregate(self,global_model:Model, updates:List[Update]) -> None:
        raise NotImplementedError