import torch
from env.env import Env


class MCEnv:
    """Multi-crane CRP environment.

    Manages C cranes operating on one shared yard. Crane positions tracked
    independently. Constraints: no two cranes in same bay (A6) and bay
    ordering preserved (A7).

    At each step: (1) validate interference, (2) set base_env.curr_bay/row
    to match assigned crane's position, (3) delegate to base_env.step().
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
            crane_start_bays = [1]
            step = max(1, self.n_bays // n_cranes)
            for c in range(1, n_cranes):
                crane_start_bays.append(min(self.n_bays, crane_start_bays[-1] + step))
        self.crane_start_bays = crane_start_bays

        self.crane_bays = torch.full((self.batch, n_cranes), -1, device=device)
        self.crane_rows = torch.full((self.batch, n_cranes), -1, device=device)
        self.crane_busy = torch.zeros((self.batch, n_cranes), dtype=torch.bool, device=device)
        self.assigned_counts = torch.zeros(n_cranes, device=device, dtype=torch.long)
        self.interference_events = torch.zeros((self.batch, n_cranes), device=device)

    def _validate_interference(self, crane_id, dest_bay):
        if self.crane_busy[0, crane_id]:
            return False
        for c in range(self.n_cranes):
            if c != crane_id and self.crane_bays[0, c] == dest_bay:
                return False
        return True

    def _validate_non_crossing(self, crane_id, dest_bay):
        before_bays = [self.crane_bays[0, c].item() for c in range(self.n_cranes) if self.crane_bays[0, c] >= 0]
        if not before_bays:
            return True
        current_order = sorted(range(self.n_cranes), key=lambda c: self.crane_start_bays[c])
        new_positions = list(self.crane_bays[0].clone())
        new_positions[crane_id] = dest_bay
        new_order = sorted(range(self.n_cranes), key=lambda c: max(new_positions[c].item() if new_positions[c] >= 0 else self.crane_start_bays[c], 0))
        return current_order == new_order

    def _resolve_interference(self, crane_id, dest_bay):
        for c in range(self.n_cranes):
            if c != crane_id and self._validate_interference(c, dest_bay):
                return c
        raise RuntimeError(
            f'Interference at bay {dest_bay} cannot be resolved: '
            f'all {self.n_cranes} cranes are busy or blocked. '
            f'crane_bays={self.crane_bays[0].tolist()}, '
            f'crane_busy={self.crane_busy[0].tolist()}'
        )

    def step(self, dest_stack, crane_id):
        dest_idx = dest_stack[0, 0].item()
        dest_bay = (dest_idx // self.n_rows) + 1

        if not self._validate_interference(crane_id, dest_bay):
            self.interference_events[0, crane_id] += 1
            new_crane = self._resolve_interference(crane_id, dest_bay)
            crane_id = new_crane

        dest_bay = (dest_idx // self.n_rows) + 1

        current_bay = (self.crane_start_bays[crane_id]
                       if self.crane_bays[0, crane_id] < 0
                       else self.crane_bays[0, crane_id].item())
        current_row = (1
                       if self.crane_rows[0, crane_id] < 0
                       else self.crane_rows[0, crane_id].item())

        self.crane_busy[0, crane_id] = True
        self.base_env.curr_bay = torch.full((self.batch,), current_bay, device=self.device)
        self.base_env.curr_row = torch.full((self.batch,), current_row, device=self.device)

        base_cost = self.base_env.step(dest_stack)

        self.crane_bays[0, crane_id] = dest_bay
        self.crane_rows[0, crane_id] = (dest_idx % self.n_rows) + 1
        self.crane_busy[0, crane_id] = False
        self.assigned_counts[crane_id] += 1

        return base_cost, self.base_env.x.reshape(self.batch, self.n_bays, self.n_rows, self.max_tiers)

    def get_state(self):
        return self.base_env.x.reshape(self.batch, self.n_bays, self.n_rows, self.max_tiers)

    @property
    def terminated(self):
        return self.base_env.all_terminated()

    def reset(self, x):
        self.base_env = Env(self.device, x, max_retrievals=None)
        self.crane_bays = torch.full((self.batch, self.n_cranes), -1, device=self.device)
        self.crane_rows = torch.full((self.batch, self.n_cranes), -1, device=self.device)
        self.crane_busy = torch.zeros((self.batch, self.n_cranes), dtype=torch.bool, device=self.device)
        self.assigned_counts = torch.zeros(self.n_cranes, device=self.device, dtype=torch.long)
        self.interference_events = torch.zeros((self.batch, self.n_cranes), device=self.device)
