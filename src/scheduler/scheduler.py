from typing import List, Union
import numpy as np
import logging
import torch

from client import Client


class Scheduler:
    def __init__(
        self,
        clients: List[Client],
        max_delay: int,
        n_delay_min: Union[int,float],
        n_delay_max: Union[int,float],
        n_clients_per_round: int,
    ) -> None:
        
        self.clients = clients

        self.max_delay = max_delay
        self.use_delay = max_delay != 0
        self.n_delay_min = n_delay_min
        self.n_delay_max = n_delay_max
        self.n_clients_per_round = n_clients_per_round

    def _sample_clients(
        self,
        pending_clients = List[int]
    ) -> List[Client]:
        """Samples clients to request per round

        Args:
            pending_clients (_type_, optional): List of client ids for clients who have not responded to a past request. Defaults to List[int].

        Returns:
            List[Client]: A list of clients ids who is not in pending_clients.
        """
        picked_clients = [self.clients[i] for i in torch.randperm(len(self.clients))[:self.n_clients_per_round]]
        not_pending_clients = [c for c in picked_clients if not c.id in pending_clients]
        
        return not_pending_clients
    
    def _sample_delay(
        self,
        n_clients: int,
    ) -> np.array:
        """Samples delay for chosen clients

        Args:
            n_clients (int): Number of clients picked in this round

        Returns:
            np.array: Returns an array of integers denoting the number of rounds the client will withhold from global server
        """
        if self.use_delay:
            n_delay = np.random.randint(self.n_delay_min, self.n_delay_max)
            if n_delay > n_clients: n_delay = n_clients
            delays = np.random.permutation(
                np.concatenate([
                    np.zeros(n_clients - n_delay),
                    np.random.randint(1, self.max_delay, n_delay)
                ])
            )
        else:
            delays = np.zeros(n_clients)
        return delays
    
    def step(
        self,
        delayed_client_ids: List[int]
    ):
        picked_clients = self._sample_clients(delayed_client_ids)

        logging.debug(f'Scheduler -- picked {len(picked_clients)} to request')
        delays = self._sample_delay(len(picked_clients))
            
        
        return picked_clients, delays