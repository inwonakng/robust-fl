from typing import Union, List
import torch
from tqdm.auto import tqdm

from client import Client
from simulator.loader import load_model,load_aggregator,load_dataset

from sklearn.metrics import accuracy_score



class Simulator:
    def __init__(
        self,
        dataset_name:str,
        agg_type: str,
        agg_args: dict,
        model_type: str,
        model_args: dict,
        n_clients: int,
        n_malicious_clients: Union[int, float],
        n_clients_per_round: Union[int, float],
        use_delay: bool = False,
        max_delay: int = 5,
        n_delay_min:Union[int,float] = 0.1,
        n_delay_max: Union[int,float] = 0.3,
        poison_data: bool = False,
        **kwargs,
    ) -> None:
        
        x_train,y_train,x_test,y_test = load_dataset(dataset_name)
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

        # set number of malicious clients per round 
        self.n_clients = n_clients
        self.n_malicious_clients = n_malicious_clients if type(n_malicious_clients) == int else int(n_malicious_clients * n_clients)
        self.n_clients_per_round = n_clients_per_round if type(n_clients_per_round) == int else int(n_clients_per_round * n_clients)
        is_malicious = [True] * self.n_malicious_clients + [False] * (n_clients - self.n_malicious_clients)

        # set ratio of stragglers per round
        self.n_delay_min = n_delay_min if type(n_delay_min) == int else int(n_delay_min * n_clients_per_round)
        self.n_delay_max = n_delay_min if type(n_delay_max) == int else int(n_delay_max * n_clients_per_round)
        self.use_delay = use_delay
        self.poison_data = poison_data

        # instantiate the clients. 
        train_idxs = torch.randperm(len(x_train))
        n_data_per_client = len(x_train) // n_clients

        self.clients = [
            Client(
                client_id = i,
                malicious = is_mal,
                x = x_train[train_idxs[i * n_data_per_client : (i+1) * n_data_per_client]],
                y = y_train[train_idxs[i * n_data_per_client : (i+1) * n_data_per_client]],
                model = load_model(model_type, model_args),
                n_train_epoch = 5,
            ) for i,is_mal in enumerate(is_malicious)
        ]
        self.global_model = load_model(model_type, model_args)
        self.aggregator = load_aggregator(agg_type, agg_args)

    def _sample_clients(
        self,
        pending_clients = List[int]
    ) -> List[Client]:
        picked_clients = [self.clients[i] for i in torch.randperm(len(self.clients))[:self.n_clients_per_round]]
        not_pending_clients = [c for c in picked_clients if not c.id in pending_clients]
        
        return not_pending_clients
    
    def _sample_delay(
        self,
        n_clients: int,
    ) -> torch.Tensor:
        
        if self.use_delay:
            n_delay = torch.randint(self.n_delay_min,self.n_delay_max)
            delays = torch.randint(1, self.max_delay, size=n_delay)[:n_clients]
        else:
            delays = torch.tensor([0] * n_clients)

        return delays

    def run(
        self,
        n_epoch: int,
    ):
        update_tracker = {}
        for epoch in tqdm(range(n_epoch), desc='running simulation...', leave=True):
            # Step 1. Pick clients to send job to
            picked_clients = self._sample_clients(pending_clients = [])

            print(f'picked {len(picked_clients)} to request')

            # Step 2. Randomly generate delay for each client and add update to queue
            delays = self._sample_delay(len(picked_clients))
            
            for c,d in zip(picked_clients,delays):
                # first sync the clients with global
                c.sync(self.global_model.state_dict())
                update_tracker[c.id] = c.update(d)
            
            # Step 3. Check update queue to see if there are updates that are done. If not, increment their counter
            to_update_global = []
            for c,u in update_tracker.items():
                if u.delay == u.counter:
                    to_update_global.append(u)
                else:
                    u.counter += 1

            print(f'got {len(to_update_global)} updates to incorporate')
            print(f'avg loss: {sum([u.avg_loss for u in to_update_global]) / len(to_update_global)}')

            # Step 4. Update the global model with the finished local updates
            new_state = self.aggregator.aggregate(self.global_model, to_update_global)
            self.global_model.load_state_dict(new_state)
            pred = self.global_model.predict(self.x_test)

            print(f'Global model test acc: {accuracy_score(self.y_test, pred)}')