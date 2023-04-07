from typing import List

from .utils import geometric_median
from models import Trainer
from update import Update
from aggregators import Aggregator

class RFA(Aggregator):
    def __init__(
        self,
        per_component:bool = False
    ) -> None:
        """Creates an instance of RFA aggregator that uses geometric mean.

        Args:
            per_component (bool, optional): Take the geometric mean over the entire update or by each component. Each component refers to a layer in the network. Defaults to False.
        """
        super(RFA, self).__init__()
        self.per_component = per_component

    def aggregate(
        self,
        global_model: Trainer,
        updates:List[Update],
    ) -> dict:
        new_global_state = global_model.get_state()

        points, weights, update_delays = [], [], []
        for u in updates:
            points.append(list(u.new_state.values()))
            weights.append(u.train_size)
            update_delays.append(u.counter)

        if self.per_component:
            median = [
                geometric_median(components, weights)
                for components in zip(*points)
            ]
        else:
            median = geometric_median(points, weights)

        for i,k in enumerate(new_global_state.keys()):
            new_global_state[k] = median[i]

        return new_global_state