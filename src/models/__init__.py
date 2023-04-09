from .trainer import *
from .device import DEVICE

from .mlp import MLP
from .convnet import ConvNet

MODEL_MAPPING = {
    "MLP": MLP,
    "ConvNet": ConvNet,
}