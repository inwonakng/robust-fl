from typing import List, Union, Tuple
import torch
from torch import nn

class ConvNet(nn.Module):
    def __init__(
        self,
        in_features: List[int],
        out_classes: int,
        conv_shapes: List[int],
        kernel_sizes: List[int],
        hidden_layers: List[int],
        dropout: float,
        activation_func: type[nn.Module] = nn.ReLU,
        activation_args: dict = {},
    ):
        super(ConvNet, self).__init__()

        conv_setup = [3, *conv_shapes]
        self.conv_layers = nn.Sequential(*[
            nn.Sequential(
                nn.Conv2d(conv_setup[i], c, kernel_size=k,padding='same'), 
                nn.BatchNorm2d(c),
                activation_func(**activation_args),
                nn.MaxPool2d(2, 2), # makes output shape /2 /2
            )
            for i,(c,k) in enumerate(zip(conv_setup[1:],kernel_sizes))
        ],
            nn.Dropout(dropout,inplace=True) 
        )

        output_shape = conv_shapes[-1] * in_features[0] * in_features[1]
        for _ in conv_setup[1:]: output_shape /= 4

        linear_setup = [int(output_shape), *hidden_layers]
        self.linear_layers = nn.Sequential(*[
            nn.Sequential(
                nn.Linear(linear_setup[i],s,bias=True),
                activation_func(**activation_args)
            )
            for i,s in enumerate(linear_setup[1:])
        ],
            nn.Linear(linear_setup[-1], out_classes, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(
            self.linear_layers(
                self.conv_layers(x).flatten(1)
            ),
            dim = 1
        )