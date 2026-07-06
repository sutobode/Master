from .base import CraneAssignmentStrategy


class GreedyOptimal(CraneAssignmentStrategy):
    def __init__(self, n_cranes, n_bays, n_rows):
        super().__init__(n_cranes, n_bays, n_rows)
        bays_per = n_bays // n_cranes
        self.zones = []
        for c in range(n_cranes):
            start = c * bays_per
            end = start + bays_per if c < n_cranes - 1 else n_bays
            self.zones.append((start, end))

    def assign(self, env, target_stack, dest_stack):
        dest_bay = dest_stack // self.n_rows
        best_crane = 0
        best_cost = float('inf')

        for c in range(self.n_cranes):
            cost = 0.0
            curr_bay = env.crane_bays[0, c].item()
            if curr_bay >= 0:
                bay_dist = abs(curr_bay - dest_bay)
                if bay_dist > 0:
                    cost += env.base_env.t_acc + env.base_env.t_bay * bay_dist

            for other in range(self.n_cranes):
                if other != c:
                    other_bay = env.crane_bays[0, other].item()
                    if other_bay == dest_bay:
                        cost += env.base_env.t_acc

            target_bay = target_stack // self.n_rows
            for zone_c, (start, end) in enumerate(self.zones):
                if start <= target_bay < end and zone_c == c:
                    cost -= 1.0

            if cost < best_cost:
                best_cost = cost
                best_crane = c

        return best_crane
