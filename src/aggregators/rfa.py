from typing import List, Union
import torch

from models import Trainer
from update import Update
from aggregators import Aggregator

"""
Utility functions for geometric-norm RFA.
Credits to https://github.com/krishnap25/geom_median/tree/main/src/geom_median/torch for original code.
modified to be completely torch friendly and no numpy for cuda.
"""

@torch.no_grad()
def l2distance(
    p1: Union[List[torch.Tensor], torch.Tensor], 
    p2: Union[List[torch.Tensor], torch.Tensor],
) -> float:
    if type(p1) == list:
        distance = torch.linalg.norm(
            torch.stack([
                torch.linalg.norm(x1 - x2) 
                for (x1, x2) in zip(p1, p2)
            ])
        )
    else:
        distance = torch.linalg.norm(p1 - p2)
    return distance

@torch.no_grad()
def weighted_average(
    points: Union[List[List[torch.Tensor]], List[torch.Tensor]], 
    weights: torch.Tensor
) -> Union[List[torch.Tensor], torch.Tensor]:
    w_avg = [
        # each point has two dimensions, and by stacking them we get 3 dimensions
        # instead of doing nasty transposing, we can just add matching dims for the weight
        (
            torch.stack(p) 
            * weights.reshape([len(points)] + [1] * p[0].ndim)
        ).sum(0)
        for p in zip(*points)
    ]

    if type(points[0]) == torch.Tensor:
        w_avg = torch.stack(w_avg)
    return w_avg

@torch.no_grad()
def geometric_median_objective(
    median: Union[List[torch.Tensor], torch.Tensor], 
    points: Union[List[List[torch.Tensor]], List[torch.Tensor]], 
    weights: torch.Tensor
) -> float:
    # normalize weights so we don't have to divid later. If they are already normalized it does nothing.
    weights /= weights.sum()
    norm_distances = torch.tensor([
        l2distance(p, median).item() 
        for p in points
    ]).to(points[0][0].device)
    return (norm_distances * weights).sum()

@torch.no_grad()
def geometric_median(
    points: Union[List[List[torch.Tensor]], List[torch.Tensor]], 
    weights: Union[torch.Tensor, List[int]]=None, 
    eps: float=1e-6, 
    maxiter: int=100, 
    ftol: float=1e-20,
) -> List[torch.Tensor] :
    if not weights:
        weights = torch.ones(len(points)).to(points[0][0].device)
    if type(weights) == list:
        weights = torch.tensor(weights).to(points[0][0].device)

    median = weighted_average(points, weights)
    new_weights = weights
    objective_value = geometric_median_objective(median, points, weights)

    # Weiszfeld iterations
    for _ in range(maxiter):
        prev_obj_value = objective_value
        denom = torch.stack([l2distance(p, median) for p in points])
        new_weights = weights / torch.clamp(denom, min=eps) 
        median = weighted_average(points, new_weights)
        objective_value = geometric_median_objective(median, points, weights)
        if abs(prev_obj_value - objective_value) <= ftol * objective_value:
            break

    return median

class RFA(Aggregator):
    def __init__(
        self,
        per_component=False, 
    ) -> None:
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
            pass
            median = [
                geometric_median(components, weights)
                for components in zip(*points)
            ]
        else:
            median = geometric_median(points, weights)

        for i,k in enumerate(new_global_state.keys()):
            new_global_state[k] = median[i]

        return new_global_state