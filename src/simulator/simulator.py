from typing import Union, List
import torch
from tqdm.auto import tqdm
import logging
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score

from client import Client
from update import UpdateTracker
from simulator.loader import load_model,load_aggregator,load_dataset



np.random.seed(0)
torch.manual_seed(0)


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
        output_dir: str,
        max_delay: int = 0,
        n_delay_min:Union[int,float] = 0,
        n_delay_max: Union[int,float] = 0,
        poison_data: bool = False,
        **kwargs,
    ) -> None:
        """Runs a simulation of FL

        Args:
            dataset_name (str): Name of dataset to use.
            agg_type (str): Name of aggregator to use.
            agg_args (dict): Arguments to pass into the aggregator constructor.
            model_type (str): Name of model to use.
            model_args (dict): Arguments to pass into the model constructor.
            n_clients (int): number of clients to create.
            n_malicious_clients (Union[int, float]): number of or a fraction of benign clients to turn malicious. 
            n_clients_per_round (Union[int, float]): number of or a fraction of clients to request per round
            max_delay (int, optional): _description_. Defaults to 5.
            n_delay_min (Union[int,float], optional): _description_. Defaults to 0.1.
            n_delay_max (Union[int,float], optional): _description_. Defaults to 0.3.
            poison_data (bool, optional): _description_. Defaults to False.
        """        

        # set up logging
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(filename=f'{output_dir}/run.log', filemode='w', format='%(levelname)s - %(message)s', level=logging.DEBUG)
        

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
        self.use_delay = max_delay != 0
        self.poison_data = poison_data

        # instantiate the clients. 
        train_idxs = np.random.permutation(len(x_train))
        n_data_per_client = len(x_train) // n_clients

        self.clients = [
            Client(
                client_id = i,
                model = load_model(model_type,model_args),
                malicious = is_mal,
                x_train = x_train[train_idxs[i * n_data_per_client : (i+1) * n_data_per_client]],
                y_train = y_train[train_idxs[i * n_data_per_client : (i+1) * n_data_per_client]],
                x_test = x_test,
                y_test = y_test,
                n_train_epoch = 5,
            ) for i,is_mal in enumerate(is_malicious)
        ]
        self.global_model = load_model(model_type, model_args)
        self.aggregator = load_aggregator(agg_type, agg_args)
        logging.debug('Simulator -- successfully constructed.')


    def _sample_clients(
        self,
        pending_clients = List[int]
    ) -> List[Client]:
        picked_clients = [self.clients[i] for i in np.random.permutation(len(self.clients))[:self.n_clients_per_round]]
        not_pending_clients = [c for c in picked_clients if not c.id in pending_clients]
        
        return not_pending_clients
    
    def _sample_delay(
        self,
        n_clients: int,
    ) -> np.array:
        
        if self.use_delay:
            n_delay = np.random.randint(self.n_delay_min,self.n_delay_max)
            delays = np.random.randint(1,self.max_delay,n_delay)[:n_clients]
        else:
            delays = np.zeros(n_clients)

        return delays

    def run(
        self,
        n_epoch: int,
    ):

        report = []
        update_tracker = UpdateTracker()
        for epoch in tqdm(range(n_epoch), desc='running simulation...', leave=True):
            # Step 1. Pick clients to send job to
            picked_clients = self._sample_clients(update_tracker.delayed_clients)

            logging.debug(f'Simulator -- picked {len(picked_clients)} to request')

            # Step 2. Randomly generate delay for each client and add update to queue
            delays = self._sample_delay(len(picked_clients))
            
            global_state = self.global_model.state_dict()
            for c,d in zip(picked_clients,delays):
                update_tracker.add(c.update(global_state,d))
            
            # Step 3. Check update queue to see if there are updates that are done. If not, increment their counter
            to_update_global = update_tracker.step()
            
            avg_losses = [u.avg_loss for u in to_update_global]
            train_acc_scores = [u.train_acc_score for u in to_update_global]
            test_acc_scores = [u.test_acc_score for u in to_update_global]

            logging.debug(f'Simulator -- got {len(to_update_global)} updates to incorporate')
            logging.debug(f'Simulator -- avg loss: {sum(avg_losses) / len(to_update_global)}')
            logging.debug(f'Simulator -- client avg train acc: {sum(train_acc_scores) / len(train_acc_scores)}')
            logging.debug(f'Simulator -- client avg test acc: {sum(test_acc_scores) / len(test_acc_scores)}')

            # Step 4. Update the global model with the finished local updates
            new_state = self.aggregator.aggregate(self.global_model, to_update_global)
            self.global_model.load_state_dict(new_state)
            pred = self.global_model.predict(self.x_test).cpu().numpy()
            logging.debug(f'Simulator -- Global model test acc: {accuracy_score(self.y_test, pred)}')


            report.append({
                'round': epoch,
                'client_req': len(picked_clients),
                'new_updates': len(to_update_global),
                'avg_loss': avg_losses,
                'queue_size': len(update_tracker.delayed_clients),
                # 'model_pred': pred,
                'client_train_acc': train_acc_scores,
                'client_test_acc': test_acc_scores,
                'accuracy_score': accuracy_score(self.y_test, pred)
            })
        
        pd.DataFrame(report).to_csv(self.output_dir/'report.csv',index=False)