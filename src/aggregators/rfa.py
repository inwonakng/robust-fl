from typing import List

from .utils import geometric_median
from models import Trainer
from update import Update
from aggregators import Aggregator

class RFA(Aggregator):
    def __init__(
        self,
        per_component:bool = False,
        **kwargs,
    ) -> None:
        """Creates an instance of RFA aggregator that uses geometric mean.

        Args:
            per_component (bool, optional): Take the geometric mean over the entire update or by each component. Each component refers to a layer in the network. Defaults to False.
        """
        super(RFA, self).__init__(**kwargs)
        self.per_component = per_component

    def aggregate(
        self,
        cur_epoch: int,
        global_model: Trainer,
        updates:List[Update],
    ) -> dict:
        new_global_state = global_model.get_state()

        model_weights, update_weights = self.parse_updates(cur_epoch, updates)

        if self.per_component:
            median = [
                geometric_median(components, update_weights)
                for components in zip(*model_weights)
            ]
        else:
            median = geometric_median(model_weights, update_weights)

        for i,k in enumerate(new_global_state.keys()):
            new_global_state[k] = median[i]

        return new_global_state