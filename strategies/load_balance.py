from .base import CraneAssignmentStrategy


class LoadBalance(CraneAssignmentStrategy):
    def __init__(self, n_cranes, n_bays, n_rows):
        super().__init__(n_cranes, n_bays, n_rows)
        self.task_counts = [0] * n_cranes

    def assign(self, env, target_stack, dest_stack):
        min_idx = min(range(self.n_cranes), key=lambda i: self.task_counts[i])
        self.task_counts[min_idx] += 1
        return min_idx

    def reset(self):
        self.task_counts = [0] * self.n_cranes
