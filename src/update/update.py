class Update:
    def __init__(self, new_state, avg_loss, delay) -> None:
        self.new_state = new_state
        self.avg_loss = avg_loss
        self.delay = delay
        self.counter = 0