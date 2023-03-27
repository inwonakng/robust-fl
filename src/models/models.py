from typing import Tuple,Type
import torch
from torch import nn, optim
from tqdm.auto import tqdm

from .device import DEVICE

LOSS_MAPPING = {
    "CrossEntropyLoss": nn.CrossEntropyLoss,

}

OPTIMIZER_MAPPING = {
    "Adam": optim.Adam,

}

class Model:
    def __init__(
        self, 
        model_constructor: Type[nn.Module],
        loss_func: nn.Module = nn.CrossEntropyLoss,
        loss_args: dict = {},
        optimizer_func: torch.optim = optim.Adam,
        optimizer_args: dict = {'lr': 0.01},
        **model_args,
    ) -> None:
        """
        Constructor for the model class that's used by the global model and clients. 
        Includes wrapper for the prediction and training, following scikit-learn API.

        Args:
            model_constructor (Type[nn.Module]): Constructor for the nn.Module class.
            loss_func (nn.Module, optional): Loss function to minimize.
            loss_args (dict, optional): Parameters for the loss function. Defaults to {}.
            optimizer_func (torch.optim, optional): Optimizer function to use. Defaults to optim.Adam.
            optimizer_args (_type_, optional): Parameters for the optimizer function. Defaults to {'lr': 0.01}.
        """

        self.clone_args = {
            'model_constructor': model_constructor,
            'loss_func': loss_func,
            'loss_args': loss_args,
            'optimizer_func': optimizer_func,
            'optimizer_args': optimizer_args,
            **model_args
        }
        self.model = model_constructor(**model_args)
        self.optimizer = OPTIMIZER_MAPPING[optimizer_func](self.model.parameters(), **optimizer_args)
        self.criterion = LOSS_MAPPING[loss_func](**loss_args)
        self.device = DEVICE
        self.model.to(self.device)

    def fit(
        self, 
        x: torch.Tensor, 
        y: torch.Tensor, 
        n_epoch: int = 1, 
        batch_size:int = 100, 
        verbose:bool=False
    ) -> Tuple[torch.Tensor, float]:
        self.model.train()
        losses = []
        x = x.to(self.device)
        y = y.to(self.device)
        for epoch in tqdm(range(n_epoch), desc='Training..', leave=False, disable=not verbose):
            epoch_loss = []
            for batch_idx in range(0, len(x), batch_size):
                batch_x = x[batch_idx : batch_idx+batch_size]
                batch_y = y[batch_idx : batch_idx+batch_size]
                self.optimizer.zero_grad()
                out = self.model(batch_x)
                loss = self.criterion(out, batch_y)
                loss.backward()
                self.optimizer.step()
                epoch_loss.append(loss.item())
            losses.append(sum(epoch_loss))
        return sum(losses) / len(losses)
    
    @torch.no_grad()
    def predict(self, x:torch.Tensor) -> torch.Tensor:
        self.model.eval()
        x = x.to(self.device)
        out = torch.sigmoid(self.model(x))
        pred = out.argmax(1).long().cpu()
        return pred

    def get_state(self) -> dict:
        return self.model.state_dict()
    
    def set_state(self, new_state:dict) -> None:
        self.model.load_state_dict(new_state)


    def clone(self):
        return self.__class__(**self.clone_args)

