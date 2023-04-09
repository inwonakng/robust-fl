from typing import List
import torch
import logging
from sklearn.metrics import accuracy_score

from update import Update
from models import Trainer

class Client:
    def __init__(
        self, 
        client_id:int, 
        malicious: bool,
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        x_test: torch.Tensor,
        y_test: torch.Tensor,
        valid_labels = List[int],
        n_train_epoch: int = 1,
        batch_size: int = 100,
    ):
        self.id = client_id
        self.malicious = malicious
        self.x_train = x_train
        self.x_test = x_test
        self.y_test = y_test
        self.valid_labels = valid_labels
        self.n_train_epoch = n_train_epoch
        self.batch_size = batch_size
        self.update_counter = 0

        if self.malicious:
            self.y_train = self.flip_labels(y_train)
        else:
            self.y_train = y_train

    def flip_labels(self,y_train:torch.Tensor):
        # we are shuffling once and reusing it to make sure that there are no overlaps in the label flipping
        poisoned_y_train = y_train.clone()
        for t in self.valid_labels:
            available_labels = list(set(self.valid_labels) - set([t]))
            random_poison_labels = torch.tensor([
                available_labels[torch.randperm(len(available_labels))[0]]
                for _ in range(sum(y_train == t))
            ])
            poisoned_y_train[y_train == t] = random_poison_labels

        logging.debug(f'Client {self.id} -- successfully poisoned data')
        return poisoned_y_train

    def update(self,global_model:Trainer,delay:int) -> Update:
        logging.debug(f'Client {self.id} -- received request for update. Update #{self.update_counter}')
        # make copy of the gloabl model
        client_copy = global_model.clone()
        client_copy.set_state(global_model.get_state())
        avg_loss = client_copy.fit(self.x_train, self.y_train, self.n_train_epoch)
        train_size = len(self.x_train)
        new_state = client_copy.get_state()
        train_acc_score = accuracy_score(self.y_train, client_copy.predict(self.x_train))
        test_acc_score = accuracy_score(self.y_test, client_copy.predict(self.x_test))

        # sanity check model state... 
        if any(v.isnan().any() for v in new_state.values()):
            raise Exception('Client update result contains NaN!')
        
        logging.debug(f'Client {self.id} -- Successfully computed update.')
        self.update_counter += 1
        
        return Update(
            self.id,
            new_state,
            avg_loss,
            train_size,
            train_acc_score,
            test_acc_score,
            delay
        )