from typing import List, Tuple
import torch
from torch import nn, optim
from model import Model,DEVICE

class MLP(nn.Module, Model):
    def __init__(
        self,
        in_features: int,
        out_classes: int,
        hidden_layers: List[int],
        activation_func: nn.Module = nn.ReLU,
        activation_args: dict = {},
        loss_func: nn.Module = nn.CrossEntropyLoss,
        loss_args: dict = {},
        optimizer_func: torch.optim = optim.Adam,
        optimizer_args: dict = {'lr': 0.01},
    ):
        super(MLP, self).__init__()
        layer_sizes = [in_features] + hidden_layers + [out_classes]
        self.model = nn.Sequential(*[
            nn.Sequential(
                nn.Linear(layer_sizes[i], layer_sizes[i+1]),
                activation_func(**activation_args),
            )
                if i < len(layer_sizes) - 2 else
            nn.Linear(layer_sizes[i], layer_sizes[i+1])
            for i in range(len(layer_sizes) - 1)
        ])

        self.optimizer = optimizer_func(self.parameters(), **optimizer_args)
        self.criterion = loss_func(**loss_args)
        self.device = DEVICE
        self.to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.model(x))

    def fit(self, x: torch.Tensor, y: torch.Tensor, n_epoch: int = 1, batch_size:int = 100) -> Tuple[torch.Tensor, float]:
        self.train()
        losses = []
        x = x.to(self.device)
        y = y.to(self.device)
        for epoch in range(n_epoch):
            epoch_loss = []
            for batch_idx in range(0, len(x), batch_size):
                batch_x = x[batch_idx : batch_idx+batch_size]
                batch_y = y[batch_idx : batch_idx+batch_size]
                # print(batch_idx, batch_x.shape, batch_y.shape, batch_idx * batch_size, (batch_idx+1) * batch_size)
                self.optimizer.zero_grad()
                out = self.forward(batch_x)
                loss = self.criterion(out, batch_y)
                loss.backward()
                self.optimizer.step()
                epoch_loss.append(loss.item())
                # print(epoch_loss)
            losses.append(sum(epoch_loss))
        return self.state_dict(), sum(losses) / len(losses)
    
    @torch.no_grad()
    def predict(self, x:torch.Tensor) -> torch.Tensor:
        self.eval()
        x = x.to(self.device)
        out = self.forward(x)
        pred = out.argmax(1).long()
        return pred
