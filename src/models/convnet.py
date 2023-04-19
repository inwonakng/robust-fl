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



# class ConvNet(nn.Module):
#     def __init__(
#             self,
#             conv_shapes=[16,32],
#             kernel_sizes=[3,3],
#             fc_scales=[0.7],
#             dropout=.1,
#             learning_rate = 1e-4,
#             weight_decay = 1e-4,
#         ):
#         if len(conv_shapes) != len(kernel_sizes): return
#         super(ConvNet, self).__init__() 
#         conv_setup = [3, *conv_shapes]
#         self.conv_layers = nn.Sequential(*[
#             nn.Sequential(
#                 nn.Conv2d(conv_setup[i], c, kernel_size=k,padding='same'), 
#                 nn.BatchNorm2d(c),
#                 # # https://discuss.pytorch.org/t/whats-the-difference-between-nn-relu-and-nn-relu-inplace-true/948
#                 # nn.LeakyReLU(negative_slope = 0.01, inplace=True),
#                 nn.ReLU(inplace=True),
#                 nn.MaxPool2d(2, 2), # makes output shape /2 /2
#             )
#             for i,(c,k) in enumerate(zip(conv_setup[1:],kernel_sizes))
#         ],
#             nn.Dropout(dropout,inplace=True) 
#         )
#         output_shape = conv_shapes[-1] * IMG_SHAPE[0] * IMG_SHAPE[1]
#         for _ in conv_setup[1:]: output_shape /= 4

#         fc_setup = [int(output_shape*sc)
#                     for sc in [1, *fc_scales]]

#         self.fc_layers = nn.Sequential(*[
#             nn.Sequential(
#                 nn.Linear(fc_setup[i],s,bias=True),
#                 nn.ReLU(inplace=True)
#             )
#             for i,s in enumerate(fc_setup[1:])
#         ],
#             nn.Linear(fc_setup[-1], len(LABEL_TO_NUM), bias=True),
#         )
#         self.name='ConvNet'

#         self.learning_rate = learning_rate
#         self.weight_decay = weight_decay
#         self.prefix = f'ConvNet_{conv_shapes}_{kernel_sizes}_{fc_scales}_{dropout}_{learning_rate}_{weight_decay}'

#     def forward(self, x):
#         x = self.conv_layers(x).flatten(1)
#         x = self.fc_layers(x)
#         x = F.softmax(x,dim=1)
#         return x 
