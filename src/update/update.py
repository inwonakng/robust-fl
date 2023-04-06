class Update:
    def __init__(
        self, 
        client_id:int,
        new_state:dict, 
        avg_loss:float, 
        train_size: int,
        train_acc_score:float,
        test_acc_score:float,
        delay:int
    ) -> None:
        """Instantiates an Update object with the specified values

        Args:
            client_id (int): ID of client that generated this update.
            new_state (dict): New state outputed by the client.
            avg_loss (float): Average loss seen during client's training.
            train_size (int): Number of training samples used to generate the update.
            train_acc_score (float): Client's accuracy score on training data.
            test_acc_score (float): Client's accuracy score on testing data.
            delay (int): _description_
        """        
        self.client_id = client_id
        self.new_state = new_state
        self.avg_loss = avg_loss
        self.train_size = train_size
        self.train_acc_score = train_acc_score
        self.test_acc_score = test_acc_score
        self.delay = delay
        self.counter = 0
        
    
    def ready(self) -> bool:
        """_summary_

        Returns:
            bool: Whether the update is ready to be processed by the global server.
        """        
        return self.counter == self.delay

    def step(self) -> None:
        """_summary_

        Returns:
            bool: Whether the update is ready to be processed by the global server.
        """        
        self.counter += 1