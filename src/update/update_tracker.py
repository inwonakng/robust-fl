from typing import List
from update import Update

class UpdateTracker:
    def __init__(self) -> None:
        self.updates = []
        self.delayed_clients = set()

    def add(self, new_update:Update) -> None:
        self.updates.append(new_update)
        self.delayed_clients.add(new_update.client_id)

    def step(self) -> List[Update]:
        waiting = []
        ready = []
        for u in self.updates:
            if u.ready():
                ready.append(u)
                self.delayed_clients.remove(u.client_id)
            else:
                u.step()
                waiting.append(u)

        self.updates = waiting
        return ready
