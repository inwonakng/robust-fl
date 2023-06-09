from typing import List

from models import Trainer
from update import Update
from aggregators import Aggregator

class Krum(Aggregator):
    def __init__(
        self
        **kwargs,
    ) -> None:
        """Creates an instance of RFA aggregator that uses geometric mean.

        Args:
            
        """
        super(Krum, self).__init__(**kwargs)

    def aggregate(
        self,
        cur_epoch: int,
        global_model: Trainer,
        updates:List[Update],
    ) -> dict:
        new_global_state = global_model.get_state()

        return new_global_state