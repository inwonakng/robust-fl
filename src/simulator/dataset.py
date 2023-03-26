from typing import Union
import torch
import torchvision.datasets as datasets

def load_MNIST() -> Union[torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor]:
    mnist_trainset = datasets.MNIST(root='../data', train=True, download=True, transform=None)
    mnist_testset = datasets.MNIST(root='../data', train=False, download=True, transform=None)
    x_train = mnist_trainset.data.flatten(1).float()
    y_train = mnist_trainset.targets.long()
    x_test = mnist_testset.data.flatten(1).float()
    y_test = mnist_testset.targets.long()

    return x_train, y_train, x_test, y_test