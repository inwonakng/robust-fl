from typing import Union, List
import torch
import torchvision.datasets as datasets
from torchvision import transforms

def load_CIFAR(
    target_labels: List[int] = [0,1],
) -> Union[torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor]:
    
    trans = transforms.Compose([transforms.ToTensor()])

    trainset = datasets.CIFAR10(
        root='data',
        train=True, 
        download=True,     
    )
    testset = datasets.CIFAR10(
        root='data',
        train=False, 
        download=True,     
    )

    x_train_raw = torch.stack([trans(d) for d in trainset.data]).float()
    y_train_raw = torch.tensor(trainset.targets).long()
    x_test_raw = torch.stack([trans(d) for d in testset.data]).float()
    y_test_raw = torch.tensor(testset.targets).long()

    train_mask = torch.stack([(y_train_raw == t) for t in target_labels]).any(0)
    test_mask = torch.stack([(y_test_raw == t) for t in target_labels]).any(0)

    x_train = x_train_raw[train_mask]
    y_train = y_train_raw[train_mask]
    x_test = x_test_raw[test_mask]
    y_test = y_test_raw[test_mask]

    return x_train, y_train, x_test, y_test