from .mnist import *

DATASET_MAPPING = {
    "MNIST": load_MNIST,
    "CIFAR": load_CIFAR,
}