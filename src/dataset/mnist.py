from typing import Union, List
import torch
import torchvision.datasets as datasets

def load_MNIST(
    target_labels: List[int] = [0,1],
) -> Union[torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor]:
    
    if len(target_labels) < 2:
        raise Exception('Target labels must include more than 1 labels!')

    mnist_trainset = datasets.MNIST(root='data', train=True, download=True, transform=None)
    mnist_testset = datasets.MNIST(root='data', train=False, download=True, transform=None)
    x_train_raw = mnist_trainset.data.flatten(1).float() / 255
    y_train_raw = mnist_trainset.targets.long()
    x_test_raw = mnist_testset.data.flatten(1).float() / 255
    y_test_raw = mnist_testset.targets.long()

    train_mask = torch.stack([(y_train_raw == t) for t in target_labels]).any(0)
    test_mask = torch.stack([(y_test_raw == t) for t in target_labels]).any(0)

    x_train = x_train_raw[train_mask]
    y_train = y_train_raw[train_mask]
    x_test = x_test_raw[test_mask]
    y_test = y_test_raw[test_mask]

    return x_train, y_train, x_test, y_test