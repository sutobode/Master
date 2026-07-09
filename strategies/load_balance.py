from .base import CraneAssignmentStrategy


class LoadBalance(CraneAssignmentStrategy):
    """Assign to the crane with the smallest accumulated busy time.

    Uses the environment's per-crane clocks (env.crane_time), so the balance
    is over actual working time rather than task counts. (Balancing task
    COUNTS with index tie-breaking degenerates to round-robin — the earlier
    revision of this strategy produced assignments identical to RoundRobin
    on every instance.)

    Ties (common at t=0 and in symmetric yards, since crane_time starts
    identical for every crane) are broken by rotating a cursor rather than
    always favoring the lowest index — an always-lowest-index tie-break would
    silently route every tied assignment to crane 0, which is a worse
    degeneration than the original RoundRobin-equivalence bug this class was
    written to fix.
    """

    def __init__(self, n_cranes, n_bays, n_rows):
        super().__init__(n_cranes, n_bays, n_rows)
        self._cursor = 0

    def assign(self, env, target_stack, dest_stack):
        min_time = min(env.crane_time[c] for c in range(self.n_cranes))
        tied = [c for c in range(self.n_cranes) if env.crane_time[c] == min_time]
        chosen = tied[self._cursor % len(tied)]
        self._cursor += 1
        return chosen

    def reset(self):
        self._cursor = 0
