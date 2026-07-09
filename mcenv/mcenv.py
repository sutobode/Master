import torch
from env.env import Env


class MCEnv:
    """Multi-crane CRP environment with a per-crane timing model.

    State evolution is sequential (relocations are decided in global
    retrieval-priority order, per assumption A3), but each operation is
    scheduled on a per-crane timeline so that a MAKESPAN — the quantity
    multi-crane operation actually improves — is measured alongside the
    total-work sum.

    Timing model (documented in the paper, Section 4.5):
      * Each step splits into a relocation phase (duration d1) and an
        auto-retrieval phase (duration d2, possibly zero). Durations equal
        the base env's working-time costs, so sum(durations) == total work
        and MCEnv(C=1) remains numerically identical to the original Env.
      * The relocation phase starts when (i) the executing crane is free
        and (ii) its source and destination bays are free (A6 enforced as a
        DELAY, never by silently reassigning the strategy's crane choice).
      * The retrieval phase additionally waits for the previous retrieval
        to finish: containers leave the yard in strict global order.
      * A7 (non-crossing) is enforced at operation boundaries: an
        assignment that would move a crane past a neighbour along the bay
        axis is redirected to the nearest order-compatible crane and
        counted in `a7_reassignments`.
      * Idle cranes are assumed to dodge into adjacent free bays while
        another crane travels past (standard MCSP simplification); A6 is
        therefore enforced for active operations only.

    Interference bookkeeping: `interference_events` counts operations that
    had to wait for a bay to free up; `interference_wait` accumulates the
    waited time (a direct, physically meaningful cost of poor spatial
    coordination).
    """

    def __init__(self, device, x, n_cranes, crane_start_bays=None,
                 t_row=1.2, t_bay=3.5, t_acc=40, t_pd=30):
        self.device = device
        self.batch, self.n_bays, self.n_rows, self.max_tiers = x.size()
        self.max_stacks = self.n_bays * self.n_rows
        self.n_cranes = n_cranes
        self.t_row = t_row
        self.t_bay = t_bay
        self.t_acc = t_acc
        self.t_pd = t_pd

        self.base_env = Env(device, x, max_retrievals=None)
        self.base_env.t_row = t_row
        self.base_env.t_bay = t_bay
        self.base_env.t_acc = t_acc
        self.base_env.t_pd = t_pd

        if crane_start_bays is None:
            crane_start_bays = [
                max(1, min(self.n_bays, c * self.n_bays // n_cranes + 1))
                for c in range(n_cranes)
            ]
        self.crane_start_bays = crane_start_bays

        self._reset_bookkeeping()

    def _reset_bookkeeping(self):
        n = self.n_cranes
        self.crane_bays = torch.full((self.batch, n), -1, device=self.device)
        self.crane_rows = torch.full((self.batch, n), -1, device=self.device)
        self.assigned_counts = torch.zeros(n, device=self.device, dtype=torch.long)
        self.interference_events = torch.zeros((self.batch, n), device=self.device)
        self.interference_wait = 0.0
        self.a7_reassignments = 0
        self.a7_violations = 0
        self.last_crane_id = None
        # Timing state
        self.crane_time = [0.0] * n            # when each crane becomes free
        self.bay_free_time = [0.0] * (self.n_bays + 1)  # 1-indexed bays
        self.last_retrieval_finish = 0.0

    # ------------------------------------------------------------------ #
    # Position and constraint helpers
    # ------------------------------------------------------------------ #

    def _nominal_bay(self, crane_id):
        b = self.crane_bays[0, crane_id].item()
        return self.crane_start_bays[crane_id] if b < 0 else b

    def _a7_compatible(self, crane_id, dest_bay):
        """Would moving `crane_id` to `dest_bay` preserve the left-to-right
        crane order (checked against the other cranes' nominal bays)?"""
        for c in range(self.n_cranes):
            if c == crane_id:
                continue
            other = self._nominal_bay(c)
            if c < crane_id and other > dest_bay:
                return False
            if c > crane_id and other < dest_bay:
                return False
        return True

    def _enforce_a7(self, crane_id, dest_bay):
        """Return an order-compatible crane for `dest_bay`, preferring the
        strategy's choice; count a reassignment when overridden."""
        if self._a7_compatible(crane_id, dest_bay):
            return crane_id
        candidates = [c for c in range(self.n_cranes) if self._a7_compatible(c, dest_bay)]
        if not candidates:
            # Degenerate: e.g. C > B, no crane can serve dest_bay without
            # crossing another. A7 is unavoidably violated this step; count
            # it explicitly rather than silently reporting a7_reassignments
            # unchanged, which would look identical to "no problem occurred".
            self.a7_violations += 1
            return crane_id
        best = min(candidates, key=lambda c: abs(self._nominal_bay(c) - dest_bay))
        self.a7_reassignments += 1
        return best

    def _bay_ready(self, bays):
        return max(self.bay_free_time[b] for b in bays)

    def _occupy_bays(self, bays, until):
        for b in bays:
            self.bay_free_time[b] = max(self.bay_free_time[b], until)

    # ------------------------------------------------------------------ #
    # Episode interface
    # ------------------------------------------------------------------ #

    def clear(self):
        """Pre-episode auto-retrievals, executed by crane 0 by convention
        (leftmost under A7). Binds crane 0's resulting position and clock so
        MCEnv(C=1) stays exactly equivalent to the original Env."""
        bay_before = int(self.base_env.curr_bay[0].item())
        cost = self.base_env.clear()
        d = float(cost[0].item())
        if d > 0:
            bay_after = int(self.base_env.curr_bay[0].item())
            self.crane_bays[0, 0] = bay_after
            self.crane_rows[0, 0] = int(self.base_env.curr_row[0].item())
            self.crane_time[0] = d
            self.last_retrieval_finish = d
            bays = [b for b in {bay_before, bay_after} if b >= 1]
            self._occupy_bays(bays or [bay_after], d)
        return cost

    def step(self, dest_stack, crane_id):
        dest_idx = dest_stack[0, 0].item()
        dest_bay = (dest_idx // self.n_rows) + 1
        source_idx = int(self.base_env.target_stack[0].item())
        source_bay = (source_idx // self.n_rows) + 1

        crane_id = self._enforce_a7(crane_id, dest_bay)
        # The requested crane may have been overridden above (A7); callers
        # that attribute cost/logs to "the crane that moved" must read this,
        # not the crane_id they originally passed in.
        self.last_crane_id = crane_id

        # A crane that has not moved yet keeps Env's -1 convention: the base
        # env charges no approach travel on first use (curr := source stack).
        # This makes MCEnv(C=1) cost-identical to the original Env.
        current_bay = self.crane_bays[0, crane_id].item()
        current_row = self.crane_rows[0, crane_id].item()
        self.base_env.curr_bay = torch.full((self.batch,), current_bay, device=self.device)
        self.base_env.curr_row = torch.full((self.batch,), current_row, device=self.device)

        # Phase 1: relocation (approach + carry + set-down).
        d1_cost = self.base_env.step(dest_stack, no_clear=True)
        d1 = float(d1_cost[0].item())
        reloc_bay_after = int(self.base_env.curr_bay[0].item())

        # Phase 2: auto-retrievals unlocked by this relocation.
        d2_cost = self.base_env.clear()
        d2 = float(d2_cost[0].item())

        # --- schedule phase 1 (A6 delay on the bays the crane works in) ---
        reloc_bays = sorted({b for b in (current_bay, source_bay, dest_bay) if b >= 1})
        bay_ready = self._bay_ready(reloc_bays)
        ready = max(self.crane_time[crane_id], bay_ready)
        if bay_ready > self.crane_time[crane_id]:
            self.interference_events[0, crane_id] += 1
            self.interference_wait += bay_ready - self.crane_time[crane_id]
        f1 = ready + d1
        self._occupy_bays(reloc_bays, f1)

        # --- schedule phase 2 (strict global retrieval order) ---
        if d2 > 0:
            retr_bay = int(self.base_env.curr_bay[0].item())
            retr_bays = sorted({b for b in (reloc_bay_after, retr_bay) if b >= 1})
            start2 = max(f1, self.last_retrieval_finish, self._bay_ready(retr_bays))
            f2 = start2 + d2
            self.last_retrieval_finish = f2
            self._occupy_bays(retr_bays, f2)
            self.crane_time[crane_id] = f2
        else:
            self.crane_time[crane_id] = f1

        # Record where the base env actually left the crane: step() ends at
        # the relocation destination and clear() then moves the crane to the
        # retrieval point of the last retrieved target. Using dest_bay here
        # instead would teleport the crane and break travel-cost continuity.
        self.crane_bays[0, crane_id] = int(self.base_env.curr_bay[0].item())
        self.crane_rows[0, crane_id] = int(self.base_env.curr_row[0].item())
        self.assigned_counts[crane_id] += 1

        total = d1_cost + d2_cost
        return total, self.base_env.x.reshape(self.batch, self.n_bays, self.n_rows, self.max_tiers)

    @property
    def makespan(self):
        return max(max(self.crane_time), self.last_retrieval_finish)

    def get_state(self):
        return self.base_env.x.reshape(self.batch, self.n_bays, self.n_rows, self.max_tiers)

    @property
    def terminated(self):
        return self.base_env.all_terminated()

    def reset(self, x):
        self.base_env = Env(self.device, x, max_retrievals=None)
        self.base_env.t_row = self.t_row
        self.base_env.t_bay = self.t_bay
        self.base_env.t_acc = self.t_acc
        self.base_env.t_pd = self.t_pd
        self._reset_bookkeeping()
