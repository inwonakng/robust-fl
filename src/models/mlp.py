from typing import List
import torch
from torch import nn

class MLP(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_classes: int,
        hidden_layers: List[int],
        activation_func: nn.Module = nn.ReLU,
        activation_args: dict = {},
    ):
        super(MLP, self).__init__()
        layer_sizes = [in_features] + hidden_layers + [out_classes]
        self.model = nn.Sequential(*[
            nn.Sequential(
                nn.Linear(layer_sizes[i], layer_sizes[i+1]),
                activation_func(**activation_args),
            )
                if i < len(layer_sizes) - 2 else
            nn.Linear(layer_sizes[i], layer_sizes[i+1])
            for i in range(len(layer_sizes) - 1)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.model(x))
