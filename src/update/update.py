class Update:
    def __init__(
        self, 
        client_id:int,
        new_state:dict, 
        avg_loss:float, 
        train_acc_score:float,
        test_acc_score:float,
        delay:int
    ) -> None:
        self.client_id = client_id
        self.new_state = new_state
        self.avg_loss = avg_loss
        self.train_acc_score = train_acc_score
        self.test_acc_score = test_acc_score
        self.delay = delay
        self.counter = 0
    
    def ready(self) -> bool:
        return self.counter == self.delay

    def step(self) -> None:
        self.counter += 1