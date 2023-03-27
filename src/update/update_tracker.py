from typing import List
from update import Update

"""
Tracks the updates that are in the pending queue
"""

class UpdateTracker:
    def __init__(self) -> None:
        self.updates = []
        self.delayed_clients = set()

    def add(self, new_update:Update) -> None:
        """_summary_

        Args:
            new_update (Update): New update to add to the queue.
        """
        self.updates.append(new_update)
        self.delayed_clients.add(new_update.client_id)

    def step(self) -> List[Update]:
        """Returns a list of Updates that are ready to be incorporated.
        Increments the counter for updates that are still delayed.

        Returns:
            List[Update]: List of updates that are ready to be used by global server.
        """
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
