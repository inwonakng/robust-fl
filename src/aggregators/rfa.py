from typing import List
import torch

from models import Trainer
from update import Update
from aggregators import Aggregator


class RFA(Aggregator):
    def __init__(
        self
    ) -> None:
        super(RFA, self).__init__()

    def aggregate(
        self,
        global_model: Trainer,
        updates:List[Update],
    ) -> dict:
        # alpha : num samples
        

        return 