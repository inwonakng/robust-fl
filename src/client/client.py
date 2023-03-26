from typing import Union
import torch
import numpy as np

from update.update import Update

class Client:
    def __init__(
        self, 
        client_id:int, 
        malicious: bool,
        x: torch.Tensor,
        y: torch.Tensor,
        model: torch.nn.Module,
        n_train_epoch: int,
    ):
        self.id = client_id
        self.model = model
        self.malicious = malicious
        self.x = x
        self.y = y
        self.n_train_epoch = n_train_epoch

    def update(self,delay:int) -> dict:
        new_state,avg_loss = self.model.fit(self.x,self.y,self.n_train_epoch)
        return Update(
            new_state,
            avg_loss,
            delay
        )
    
    def sync(self, new_global_state: dict) -> None:        
        self.model.load_state_dict(new_global_state)