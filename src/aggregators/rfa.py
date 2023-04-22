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

        client_weights, update_weights = self.parse_updates(cur_epoch, updates)

        if self.per_component:
            component_median = {
                key: geometric_median(components, update_weights)
                for key, components in zip(new_global_state.keys(), zip(*client_weights))
                if new_global_state[key].size()
            }
            for k in component_median:
                new_global_state[k] = component_median[k]
        else:
            whole_median = geometric_median(client_weights, update_weights)

            for i,k in enumerate(new_global_state.keys()):
                new_global_state[k] = whole_median[i]

        return new_global_state