import torch
import logging
from sklearn.metrics import accuracy_score

from update import Update
from models import Model

class Client:
    def __init__(
        self, 
        client_id:int, 
        malicious: bool,
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        x_test: torch.Tensor,
        y_test: torch.Tensor,
        n_train_epoch: int = 5,
        batch_size: int = 100,
    ):
        self.id = client_id
        self.malicious = malicious
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.n_train_epoch = n_train_epoch
        self.batch_size = batch_size
        # self.counter = 0

    def update(self,global_model:Model,delay:int) -> Update:
        # make copy of the gloabl model
        client_copy = global_model.clone()
        client_copy.set_state(global_model.get_state())

        avg_loss = client_copy.fit(self.x_train,self.y_train,self.n_train_epoch)
        new_state = client_copy.get_state()
        train_acc_score = accuracy_score(self.y_train, client_copy.predict(self.x_train))
        test_acc_score = accuracy_score(self.y_test,client_copy.predict(self.x_test))

        # delete the model from memory! hopefully this saves memory.
        del client_copy
        
        return Update(
            self.id,
            new_state,
            avg_loss,
            train_acc_score,
            test_acc_score,
            delay
        )