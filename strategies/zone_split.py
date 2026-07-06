from .base import CraneAssignmentStrategy


class ZoneSplit(CraneAssignmentStrategy):
    def __init__(self, n_cranes, n_bays, n_rows):
        super().__init__(n_cranes, n_bays, n_rows)
        bays_per = n_bays // n_cranes
        self.zones = []
        for c in range(n_cranes):
            start = c * bays_per
            end = start + bays_per if c < n_cranes - 1 else n_bays
            self.zones.append((start, end))

    def assign(self, env, target_stack, dest_stack):
        target_bay = target_stack // self.n_rows
        for c, (start, end) in enumerate(self.zones):
            if start <= target_bay < end:
                return c
        return self.n_cranes - 1
