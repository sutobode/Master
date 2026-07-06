from abc import ABC, abstractmethod


class CraneAssignmentStrategy(ABC):
    def __init__(self, n_cranes, n_bays, n_rows):
        self.n_cranes = n_cranes
        self.n_bays = n_bays
        self.n_rows = n_rows

    @abstractmethod
    def assign(self, env, target_stack, dest_stack) -> int:
        pass

    def reset(self):
        pass

    @property
    def name(self):
        return self.__class__.__name__
