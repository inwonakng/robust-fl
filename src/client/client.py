import torch
import logging
from sklearn.metrics import accuracy_score

from update import Update
from models import Model

class Client:
    def __init__(
        self, 
        client_id:int, 
        model: Model,
        malicious: bool,
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        x_test: torch.Tensor,
        y_test: torch.Tensor,
        n_train_epoch: int,
    ):
        self.id = client_id
        self.model = model
        self.malicious = malicious
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.n_train_epoch = n_train_epoch
        # self.counter = 0

    def update(self,global_state:dict,delay:int) -> Update:
        self.model.load_state_dict(global_state)
        new_state,avg_loss = self.model.fit(self.x,self.y,self.n_train_epoch)
        train_acc_score = accuracy_score(self.y_train, self.model.predict(self.x_train))
        test_acc_score = accuracy_score(self.y_test,self.model.predict(self.x_test))
        
        return Update(
            self.id,
            new_state,
            avg_loss,
            train_acc_score,
            test_acc_score,
            delay
        )