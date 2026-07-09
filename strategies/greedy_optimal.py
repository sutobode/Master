from .base import CraneAssignmentStrategy


class GreedyOptimal(CraneAssignmentStrategy):
    """One-step lookahead on estimated completion time.

    For each A7-compatible crane, estimates the FINISH time of the proposed
    relocation: the crane's release time, plus any A6 bay wait, plus the
    approach travel (crane -> source stack), the carry travel (source ->
    destination stack), and the handling time. Picks the crane with the
    earliest estimated finish. Complexity: O(C * B) per assignment.
    """

    def _travel(self, env, bay_a, row_a, bay_b, row_b):
        cost = 0.0
        if bay_a != bay_b:
            cost += env.t_acc + env.t_bay * abs(bay_a - bay_b)
        cost += env.t_row * abs(row_a - row_b)
        return cost

    def assign(self, env, target_stack, dest_stack):
        src_bay = (target_stack // self.n_rows) + 1
        src_row = (target_stack % self.n_rows) + 1
        dst_bay = (dest_stack // self.n_rows) + 1
        dst_row = (dest_stack % self.n_rows) + 1

        best_crane, best_finish = 0, float('inf')
        for c in range(self.n_cranes):
            if not env._a7_compatible(c, dst_bay):
                continue

            cur_bay = env.crane_bays[0, c].item()
            cur_row = env.crane_rows[0, c].item()
            if cur_bay < 0:
                approach = 0.0  # unpositioned crane materialises at the source
            else:
                approach = self._travel(env, cur_bay, cur_row, src_bay, src_row)
            carry = self._travel(env, src_bay, src_row, dst_bay, dst_row) + env.t_pd

            bays = sorted({b for b in (cur_bay, src_bay, dst_bay) if b >= 1})
            start = max(env.crane_time[c], env._bay_ready(bays))
            finish = start + approach + carry

            if finish < best_finish:
                best_finish, best_crane = finish, c

        return best_crane
