from re import A
from typing import Union, List
import torch
from tqdm.auto import tqdm
import logging
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
import time

from client import Client
from update import UpdateTracker, Update
from scheduler import Scheduler
from .loader import load_trainer,load_aggregator,load_dataset

np.random.seed(0)
torch.manual_seed(0)


class Simulator:
    def __init__(
        self,
        dataset_args: dict,
        model_args: dict,
        agg_args: dict,
        client_args: dict,
        scheduler_args: dict,
        output_dir: str,
    ) -> None:
        """_summary_

        Args:
            dataset_args (dict): Arguments to pass into dataset loader.
            model_args (dict): Arguments to pass into model constructor.
            agg_args (dict): Arguments to pass into aggregator.
            client_args (dict): Arguments to pass into client creation.
            scheduler_args (dict): Arguments to pass into the update scheduler.
            output_dir (str): Output directory to save log file and results. Defaults to None.
        """

        # set up logging
        if output_dir:
            self.output_dir = Path(output_dir)
            self.output_dir.mkdir(parents=True, exist_ok=True)
            logging.basicConfig(filename=f'{output_dir}/run.log', filemode='w', format='%(name)s %(levelname)s - %(message)s', level=logging.DEBUG)
        else:
            self.output_dir = None

        # numba logger is annoying
        numba_logger = logging.getLogger('numba')
        numba_logger.setLevel(logging.WARNING)

        x_train,y_train,x_test,y_test = load_dataset(**dataset_args)        
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

        self.global_model = load_trainer(**model_args)
        self.aggregator = load_aggregator(**agg_args)

        self._set_scheduler_options(**scheduler_args)
        self._initiate_clients(**client_args)

        self.report = []
        self.scheduler = self._initiate_scheduler()
        self.update_tracker = UpdateTracker()
        
        logging.debug('Simulator -- successfully constructed.')


    def _set_scheduler_options(
        self,
        n_clients: int,
        n_malicious_clients: Union[int, float],
        n_clients_per_round: Union[int, float],
        max_delay: int = 5,
        n_delay_min:Union[int,float] = 0,
        n_delay_max: Union[int,float] = 0,
    ) -> None:
        """_summary_

        Args:
            n_clients (int): number of clients to create.
            n_malicious_clients (Union[int, float]): number of or a fraction of benign clients to turn malicious. 
            n_clients_per_round (Union[int, float]): number of or a fraction of clients to request per round
            max_delay (int, optional): Maximum delay to incur if a client is straggling. Defaults to 5.
            n_delay_min (Union[int,float], optional): Minimum number of or fraction of stragglers in picked clients. Defaults to 0.
            n_delay_max (Union[int,float], optional): Maximum number of or fraction of stragglers in picked clients. Defaults to 0.

        Returns:
            _type_: _description_
        """
        self.n_clients = n_clients
        self.n_malicious_clients = n_malicious_clients if type(n_malicious_clients) == int else int(n_malicious_clients * n_clients)
        self.n_clients_per_round = n_clients_per_round if type(n_clients_per_round) == int else int(n_clients_per_round * n_clients)
        
        # set ratio of stragglers per round
        self.n_delay_min = n_delay_min if type(n_delay_min) == int else int(n_delay_min * self.n_clients_per_round)
        self.n_delay_max = n_delay_max if type(n_delay_max) == int else int(n_delay_max * self.n_clients_per_round)
        self.max_delay = max_delay
        
    def _initiate_scheduler(self) -> Scheduler:
        return Scheduler(
            self.clients,
            self.max_delay,
            self.n_delay_min,
            self.n_delay_max,
            self.n_clients_per_round
        )

    def _initiate_clients(
        self,
        n_train_epoch: int = 1,
    ) -> None:
        """Creates the clients to consider

        Args:
            n_train_epoch (int, optional): Number of epochs each client should train for. Defaults to 1.
        """        

        is_malicious = [True] * self.n_malicious_clients + [False] * (self.n_clients - self.n_malicious_clients)

        train_idxs = torch.randperm(len(self.x_train))
        n_data_per_client = len(self.x_train) // self.n_clients

        self.clients = [
            Client(
                client_id = i,
                is_malicious = is_mal,
                x_train = self.x_train[train_idxs[i * n_data_per_client : (i+1) * n_data_per_client]],
                y_train = self.y_train[train_idxs[i * n_data_per_client : (i+1) * n_data_per_client]],
                x_test = self.x_test,
                y_test = self.y_test,
                valid_labels = self.y_train.unique().tolist(),
                n_train_epoch = n_train_epoch,
            ) for i,is_mal in enumerate(is_malicious)
        ]
    
    def step(
        self,
    ) -> List[Update]:
        # Step 1. Pick clients to send job to and compute delays
        picked_clients,delays = self.scheduler.step(self.update_tracker.delayed_client_ids)

        # Step 2. Train model on each client 
        for c,d in zip(picked_clients,delays):
            self.update_tracker.add(c.update(self.global_model,d))

        # Step 3. Check update queue to see if there are updates that are done. If not, increment their counter
        to_update_global = self.update_tracker.step()

        return picked_clients, to_update_global
    
    def find_client(self, client_id: int) -> Client:
        return [c for c in self.clients if c.id == client_id][0]

    def run(
        self,
        n_epoch: int,
        overwrite: bool = False
    ):
        
        """Runs the simluation for specified nubmer of rounds.

        Args:
            n_epoch (int): Nubmer of rounds to run in simulation.
            overwrite (bool): If true, the simulation skips if the report already exists.
        """
        if not overwrite and self.output_dir and (self.output_dir / 'report.csv').is_file():
            return
        
        print(f'Running {str(self.output_dir)}', flush=True)
        for epoch in tqdm(
            range(n_epoch), 
            # desc=f'Running {str(self.output_dir)}', 
            leave=True,
        ):
            
            logging.debug(f'Simulator -- Epoch: {epoch+1}')
            picked_clients, to_update_global = self.step()
            logging.debug(f'Simulator -- got {len(to_update_global)} updates to incorporate')
            
            avg_train_acc = 0
            avg_test_acc = 0
            avg_benign_train_acc = 0
            avg_benign_test_acc = 0
            avg_malicious_train_acc = 0
            avg_malicious_test_acc = 0

            if len(to_update_global) > 0:
                # only update the global model if we have any updates
                avg_losses, train_acc_scores, test_acc_scores, is_client_malicious = zip(*[
                    (
                        u.avg_loss,
                        u.train_acc_score,
                        u.test_acc_score,
                        self.find_client(u.client_id).is_malicious,
                    )
                    for u in to_update_global
                ])

                avg_losses = torch.tensor(avg_losses).float()
                train_acc_scores = torch.tensor(train_acc_scores).float()
                test_acc_scores = torch.tensor(test_acc_scores).float()
                is_client_malicious = torch.tensor(is_client_malicious).bool()
                
                avg_train_acc = train_acc_scores.mean().item()
                avg_test_acc = test_acc_scores.mean().item()
                avg_benign_train_acc = train_acc_scores[~is_client_malicious].mean().item()
                avg_benign_test_acc = train_acc_scores[~is_client_malicious].mean().item()

                if is_client_malicious.any():
                    avg_malicious_train_acc = train_acc_scores[is_client_malicious].mean().item()
                    avg_malicious_test_acc = test_acc_scores[is_client_malicious].mean().item()

                logging.debug(f'Simulator -- avg loss: {sum(avg_losses) / len(to_update_global)}')
                logging.debug(f'Simulator -- client avg train acc: {sum(train_acc_scores) / len(train_acc_scores)}')
                logging.debug(f'Simulator -- client avg test acc: {sum(test_acc_scores) / len(test_acc_scores)}')

                # Step 4. Update the global model with the finished local updates
                start = time.time()
                new_state = self.aggregator(epoch, self.global_model, to_update_global)
                agg_time = time.time() - start

                if new_state is not None:
                    self.global_model.set_state(new_state)
                    logging.debug('Simulator -- global model updated with new weights')
            else:
                logging.debug('Simulator -- No new weights to apply')

            
            pred = self.global_model.predict(self.x_test).cpu()
            global_accuracy_score = accuracy_score(self.y_test, pred)
            logging.debug(f'Simulator -- Global model test acc: {global_accuracy_score}')

            self.report.append({
                'round': epoch,
                'client_req': len(picked_clients),
                'new_updates': len(to_update_global),
                'avg_loss': avg_losses,
                'queue_size': len(self.update_tracker.delayed_client_ids),
                'client_train_acc': train_acc_scores,
                'client_test_acc': test_acc_scores,
                'client_train_acc_avg': avg_train_acc,
                'client_test_acc_avg': avg_test_acc,
                'benign_client_train_acc_avg': avg_benign_train_acc,
                'benign_client_test_acc_avg': avg_benign_test_acc,
                'malicious_client_train_acc_avg': avg_malicious_train_acc,
                'malicious_client_test_acc_avg': avg_malicious_test_acc,
                'accuracy_score': accuracy_score(self.y_test, pred),
                'aggregator_time': agg_time,
            })
        logging.debug('Simulator -- Simulation Complete')
        if self.output_dir:
            pd.DataFrame(self.report).to_csv(self.output_dir/'report.csv',index=False)
            logging.debug('Simulator -- saved results')
