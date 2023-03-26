import torch


class Model:
    def __init__(self) -> None:
        pass
    
    def fit(self,x: torch.Tensor, y: torch.Tensor, n_epoch: int = 1, batch_size:int = 100):
        raise NotImplementedError
    
    def predict(self,x:torch.Tensor):
        raise NotImplementedError