import torch
from env.env import Env


class MCEnv:
    """Multi-crane CRP environment.

    Manages C cranes operating on one shared yard. Crane positions tracked
    independently. Interference constraints: no two cranes in same bay.

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
        self.assigned_counts = torch.zeros(n_cranes, device=device, dtype=torch.long)
        self.interference_events = torch.zeros((self.batch, n_cranes), device=device)

    def _validate_interference(self, crane_id, dest_bay):
        for c in range(self.n_cranes):
            if c != crane_id and self.crane_bays[0, c] == dest_bay:
                return False
        return True

    def _resolve_interference(self, crane_id, dest_bay):
        for c in range(self.n_cranes):
            if c != crane_id and self._validate_interference(c, dest_bay):
                return c
        return crane_id

    def step(self, dest_stack, crane_id):
        source_idx = self.base_env.target_stack
        dest_idx = dest_stack[0, 0].item()
        dest_bay = (dest_idx // self.n_rows) + 1

        if not self._validate_interference(crane_id, dest_bay):
            self.interference_events[0, crane_id] += 1
            crane_id = self._resolve_interference(crane_id, dest_bay)

        current_bay = (self.crane_start_bays[crane_id]
                       if self.crane_bays[0, crane_id] < 0
                       else self.crane_bays[0, crane_id].item())
        current_row = (1
                       if self.crane_rows[0, crane_id] < 0
                       else self.crane_rows[0, crane_id].item())

        self.base_env.curr_bay = torch.full((self.batch,), current_bay, device=self.device)
        self.base_env.curr_row = torch.full((self.batch,), current_row, device=self.device)

        base_cost = self.base_env.step(dest_stack)

        self.crane_bays[0, crane_id] = dest_bay
        self.crane_rows[0, crane_id] = (dest_idx % self.n_rows) + 1
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
        self.assigned_counts = torch.zeros(self.n_cranes, device=self.device, dtype=torch.long)
        self.interference_events = torch.zeros((self.batch, self.n_cranes), device=self.device)
