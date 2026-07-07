from .base import CraneAssignmentStrategy


class GreedyOptimal(CraneAssignmentStrategy):
    """1-step lookahead: simulate each crane's cost and pick the minimum.

    For each available crane, computes the total cost (travel + interference)
    to execute the relocation. Picks the crane with minimum projected cost.
    Complexity: O(C * (B + C)) per assignment.
    """

    def __init__(self, n_cranes, n_bays, n_rows):
        super().__init__(n_cranes, n_bays, n_rows)

    def assign(self, env, target_stack, dest_stack):
        dest_bay = (dest_stack // self.n_rows) + 1
        best_crane = 0
        best_cost = float('inf')

        for c in range(self.n_cranes):
            cost = 0.0

            curr_bay = env.crane_bays[0, c].item()
            if curr_bay >= 0:
                bay_dist = abs(curr_bay - dest_bay)
                if bay_dist > 0:
                    cost += env.base_env.t_acc + env.base_env.t_bay * bay_dist
            else:
                start_bay = env.crane_start_bays[c]
                bay_dist = abs(start_bay - dest_bay)
                if bay_dist > 0:
                    cost += env.base_env.t_acc + env.base_env.t_bay * bay_dist

            for other in range(self.n_cranes):
                if other != c:
                    other_bay = env.crane_bays[0, other].item()
                    if other_bay == dest_bay:
                        cost += env.base_env.t_acc

            if cost < best_cost:
                best_cost = cost
                best_crane = c

        return best_crane
