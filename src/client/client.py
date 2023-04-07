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
        n_train_epoch: int = 5,
        batch_size: int = 100,
        poison_ratio: float = 0.1,
    ):
        self.id = client_id
        self.malicious = malicious
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.n_train_epoch = n_train_epoch
        self.batch_size = batch_size
        self.poison_ratio = poison_ratio

    def poison_data(self):
        # we are shuffling once and reusing it to make sure that there are no overlaps in the label flipping
        shuffled_idxs = torch.randperm(len(self.y_train))
        target_labels = self.y_train.unique()
        poison_size_per_target = int(len(self.y_train) * self.poison_ratio / len(target_labels))
        poisoned_y_train = self.y_train.clone()
        for i,t in enumerate(target_labels):
            available_labels = list(set(target_labels) - set([t]))
            poison_idxs = shuffled_idxs[poison_size_per_target * i: poison_size_per_target * (i+1)]
            random_poison_labels = torch.tensor([
                available_labels[torch.randperm(len(available_labels))[0]]
                for _ in poison_idxs
            ])
            poisoned_y_train[poison_idxs] = random_poison_labels

        return poisoned_y_train

    def update(self,global_model:Trainer,delay:int) -> Update:
        # make copy of the gloabl model
        client_copy = global_model.clone()
        client_copy.set_state(global_model.get_state())

        if self.malicious:
            y_train = self.poison_data()
        else:
            y_train = self.y_train

        avg_loss = client_copy.fit(self.x_train, y_train, self.n_train_epoch)
        train_size = len(self.x_train)
        new_state = client_copy.get_state()
        train_acc_score = accuracy_score(self.y_train, client_copy.predict(self.x_train))
        test_acc_score = accuracy_score(self.y_test, client_copy.predict(self.x_test))
        
        return Update(
            self.id,
            new_state,
            avg_loss,
            train_size,
            train_acc_score,
            test_acc_score,
            delay
        )