from typing import List, Union
import torch

"""
Utility functions for geometric-norm RFA.
Credits to https://github.com/krishnap25/geom_median/tree/main/src/geom_median/torch for original code.
modified to be completely torch friendly and no numpy for cuda.
"""

def normalize_weights(weights: Union[torch.Tensor, List[int], List[float]]):
    if type(weights) == list:
        weights = torch.tensor(weights)
    weights = weights.float() / weights.sum()
    return weights

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
    model_weights: Union[List[List[torch.Tensor]], List[torch.Tensor], torch.Tensor], 
    update_weights: torch.Tensor
) -> Union[List[torch.Tensor], torch.Tensor]:
    # normalize update_weights so we don't have to divid later. If they are already normalized it does nothing.
    update_weights = normalize_weights(update_weights)
    w_avg = [
        # each point has two dimensions, and by stacking them we get 3 dimensions
        # instead of doing nasty transposing, we can just add matching dims for the weight
        (
            torch.stack(p) 
            * update_weights.reshape([len(model_weights)] + [1] * p[0].ndim)
        ).sum(0)
        for p in zip(*model_weights)
    ]
    if type(model_weights[0]) == torch.Tensor:
        w_avg = torch.stack(w_avg)

    return w_avg


@torch.no_grad()
def random_sample_average(
    model_weights: torch.Tensor, 
    global_weights: torch.Tensor,
    p: float,
) -> Union[List[torch.Tensor], torch.Tensor]:
    # normalize update_weights so we don't have to divid later. If they are already normalized it does nothing.
    
    size = model_weights.size()
    selected_params = torch.zeros(size)
    rand_mask = torch.rand(model_weights.size()) < p
    selected_params[rand_mask] = model_weights[rand_mask]

    keep_original_mask = rand_mask.sum(0) == 0
    final_params = selected_params.sum(0)
    final_params[~keep_original_mask] = final_params[~keep_original_mask] / rand_mask.sum(0)[~keep_original_mask]
    final_params[keep_original_mask] = global_weights[keep_original_mask]
    
    return final_params


@torch.no_grad()
def geometric_median_objective(
    median: Union[List[torch.Tensor], torch.Tensor], 
    model_weights: Union[List[List[torch.Tensor]], List[torch.Tensor], torch.Tensor], 
    update_weights: torch.Tensor
) -> float:
    # normalize update_weights so we don't have to divid later. If they are already normalized it does nothing.
    update_weights = normalize_weights(update_weights)
    norm_distances = torch.tensor([
        l2distance(p, median).item() 
        for p in model_weights
    ]).to(model_weights[0][0].device)
    return (norm_distances * update_weights).sum()

@torch.no_grad()
def geometric_median(
    model_weights: Union[List[List[torch.Tensor]], List[torch.Tensor], torch.Tensor], 
    update_weights: Union[torch.Tensor, List[int]]=None, 
    eps: float=1e-6, 
    maxiter: int=100, 
    ftol: float=1e-20,
) -> List[torch.Tensor]:
    # normalize update_weights
    update_weights = normalize_weights(update_weights)

    median = weighted_average(model_weights, update_weights)
    new_update_weights = update_weights
    objective_value = geometric_median_objective(median, model_weights, update_weights)

    # Weiszfeld iterations
    for _ in range(maxiter):
        prev_obj_value = objective_value
        denom = torch.stack([l2distance(p, median) for p in model_weights])
        new_update_weights = update_weights / torch.clamp(denom, min=eps) 
        median = weighted_average(model_weights, new_update_weights)
        objective_value = geometric_median_objective(median, model_weights, update_weights)
        if abs(prev_obj_value - objective_value) <= ftol * objective_value:
            break

    return median
