import torch
try:
    from env.env import Env
except:
    import os, sys
    current_dir = os.path.dirname(os.path.abspath(__file__))
    top_level_dir = os.path.abspath(os.path.join(current_dir, ".."))
    if top_level_dir not in sys.path:
        sys.path.append(top_level_dir)
    from env.env import Env


class RandomRelocate:
    """Random relocation: pick a random feasible stack."""

    def run(self, x):
        batch, n_bays, n_rows, n_tiers = x.size()
        env = Env('cpu', x, max_retrievals=None)
        env.t_pd = 30; env.t_acc = 40; env.t_bay = 3.5; env.t_row = 1.2
        total_cost = 0

        cost = env.clear()
        if isinstance(cost, torch.Tensor):
            total_cost += cost[0].item() if cost.dim() > 0 else cost.item()

        while not env.all_terminated():
            stacks = env.x[0]
            full_mask = (stacks[:, -1] > 0)
            target_idx = env.target_stack[0].item()
            full_mask[target_idx] = True
            feasible = torch.where(~full_mask)[0]
            if len(feasible) == 0:
                break
            idx = torch.randint(0, len(feasible), (1,))[0].item()
            dest = torch.tensor([[feasible[idx].item()]])
            cost = env.step(dest)
            if isinstance(cost, torch.Tensor):
                total_cost += cost[0].item() if cost.dim() > 0 else cost.item()

        return total_cost, 0


class NearestStack:
    """Nearest stack: relocate to the closest stack with space."""

    def run(self, x):
        batch, n_bays, n_rows, n_tiers = x.size()
        env = Env('cpu', x, max_retrievals=None)
        env.t_pd = 30; env.t_acc = 40; env.t_bay = 3.5; env.t_row = 1.2
        total_cost = 0

        cost = env.clear()
        if isinstance(cost, torch.Tensor):
            total_cost += cost[0].item() if cost.dim() > 0 else cost.item()

        while not env.all_terminated():
            stacks = env.x[0]
            full_mask = (stacks[:, -1] > 0)
            target_idx = env.target_stack[0].item()
            full_mask[target_idx] = True
            feasible = torch.where(~full_mask)[0]
            if len(feasible) == 0:
                break

            target_bay = target_idx // n_rows
            target_row = target_idx % n_rows
            best_dist = float('inf')
            best = feasible[0].item()
            for f in feasible:
                f = f.item()
                d = abs(f // n_rows - target_bay) + abs(f % n_rows - target_row)
                if d < best_dist:
                    best_dist = d
                    best = f
            dest = torch.tensor([[best]])
            cost = env.step(dest)
            if isinstance(cost, torch.Tensor):
                total_cost += cost[0].item() if cost.dim() > 0 else cost.item()

        return total_cost, 0


class LowestHeight:
    """Lowest height: relocate to the shortest stack with space."""

    def run(self, x):
        batch, n_bays, n_rows, n_tiers = x.size()
        env = Env('cpu', x, max_retrievals=None)
        env.t_pd = 30; env.t_acc = 40; env.t_bay = 3.5; env.t_row = 1.2
        total_cost = 0

        cost = env.clear()
        if isinstance(cost, torch.Tensor):
            total_cost += cost[0].item() if cost.dim() > 0 else cost.item()

        while not env.all_terminated():
            stacks = env.x[0]
            full_mask = (stacks[:, -1] > 0)
            target_idx = env.target_stack[0].item()
            full_mask[target_idx] = True
            feasible = torch.where(~full_mask)[0]
            if len(feasible) == 0:
                break

            heights = (stacks > 0).sum(dim=1)
            best_h = float('inf')
            best = feasible[0].item()
            for f in feasible:
                f = f.item()
                h = heights[f].item()
                if h < best_h:
                    best_h = h
                    best = f
            dest = torch.tensor([[best]])
            cost = env.step(dest)
            if isinstance(cost, torch.Tensor):
                total_cost += cost[0].item() if cost.dim() > 0 else cost.item()

        return total_cost, 0
