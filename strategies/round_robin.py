from .base import CraneAssignmentStrategy


class RoundRobin(CraneAssignmentStrategy):
    def __init__(self, n_cranes, n_bays, n_rows):
        super().__init__(n_cranes, n_bays, n_rows)
        self._next = 0

    def assign(self, env, target_stack, dest_stack):
        c = self._next
        self._next = (self._next + 1) % self.n_cranes
        return c

    def reset(self):
        self._next = 0
