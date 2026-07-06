"""SOTA baselines for CRP: Rollout + GP and Beam Search."""

import torch
try:
    from env.env import Env
    from baselines.durasevic2025 import Durasevic2025
except:
    import os, sys
    current_dir = os.path.dirname(os.path.abspath(__file__))
    top_level_dir = os.path.abspath(os.path.join(current_dir, ".."))
    if top_level_dir not in sys.path:
        sys.path.append(top_level_dir)
    from env.env import Env
    from baselines.durasevic2025 import Durasevic2025


class RolloutGP:
    """Rollout + GP-evolved rule (Durasevic et al. 2024, PPSN).
    
    1-step lookahead: for each feasible relocation, simulate the immediate step,
    then evaluate the resulting state via full GP run. Pick the move with
    lowest total (immediate + simulated future) cost.
    """

    def __init__(self):
        self._gp = Durasevic2025()

    def run(self, x):
        batch, n_bays, n_rows, n_tiers = x.size()
        device = x.device

        env = Env(device, x, max_retrievals=None)
        env.t_pd = 30; env.t_acc = 40; env.t_bay = 3.5; env.t_row = 1.2
        total_cost = 0.0

        init_cost = env.clear()
        if isinstance(init_cost, torch.Tensor):
            total_cost += init_cost[0].item() if init_cost.dim() > 0 else init_cost.item()

        while not env.all_terminated():
            stacks = env.x[0]
            full_mask = (stacks[:, -1] > 0)
            target_idx = env.target_stack[0].item()
            full_mask[target_idx] = True
            feasible = torch.where(~full_mask)[0]
            if len(feasible) == 0:
                break

            best_total = float('inf')
            best_dest = feasible[0].item()

            for f_idx in feasible:
                f = f_idx.item()
                sim = _clone_env(env)
                step_cost = sim.step(torch.tensor([[f]]))
                if isinstance(step_cost, torch.Tensor):
                    step_cost = step_cost[0].item() if step_cost.dim() > 0 else step_cost.item()

                # GP simulates rest
                sim_x = sim.x.reshape(1, env.n_bays, env.n_rows, env.max_tiers)
                gp_result = self._gp.run(sim_x)
                gp_cost = gp_result[0] if isinstance(gp_result, tuple) else gp_result
                if isinstance(gp_cost, torch.Tensor):
                    gp_cost = gp_cost[0].item() if gp_cost.dim() > 0 else gp_cost.item()

                total = step_cost + gp_cost
                if total < best_total:
                    best_total = total
                    best_dest = f

            dest = torch.tensor([[best_dest]])
            cost = env.step(dest)
            if isinstance(cost, torch.Tensor):
                total_cost += cost[0].item() if cost.dim() > 0 else cost.item()

        return total_cost, 0


class BeamSearchCRP:
    """Beam Search for CRP (Zhang & Zhu 2025, JMSE).
    
    Maintains top-B beam states. Each state = (env, cumulative_cost).
    Expands each state with all feasible relocations, scores by
    cumulative cost + future estimate, keeps top-B.
    """

    def __init__(self, beam_width=5):
        self.beam_width = beam_width

    def run(self, x, beam_width=None):
        if beam_width is not None:
            self.beam_width = beam_width

        class BState:
            __slots__ = ('x', 'n_bays', 'n_rows', 'n_tiers', 'total_cost')
            def __init__(self, x, n_bays, n_rows, n_tiers, cost):
                self.x = x.clone()
                self.n_bays = n_bays
                self.n_rows = n_rows
                self.n_tiers = n_tiers
                self.total_cost = cost

        batch, n_bays, n_rows, n_tiers = x.size()

        init_env = Env('cpu', x, max_retrievals=None)
        init_env.t_pd = 30; init_env.t_acc = 40
        init_env.t_bay = 3.5; init_env.t_row = 1.2
        init_cost = 0.0

        c = init_env.clear()
        if isinstance(c, torch.Tensor):
            init_cost += c[0].item() if c.dim() > 0 else c.item()

        beam = [BState(init_env.x.reshape(batch, n_bays, n_rows, n_tiers),
                       n_bays, n_rows, n_tiers, init_cost)]

        for _ in range(n_bays * n_rows * n_tiers):  # upper bound
            if not beam:
                break
            if all(_is_terminal(s.x, s.n_rows) for s in beam):
                break

            candidates = []
            for state in beam:
                if _is_terminal(state.x, state.n_rows):
                    candidates.append((state.total_cost, state))
                    continue

                stacks = state.x.reshape(-1, n_tiers)
                full_mask = (stacks[:, -1] > 0)
                target_idx = _find_target(state.x.reshape(-1, n_tiers))
                if target_idx is None:
                    candidates.append((state.total_cost, state))
                    continue
                full_mask[target_idx] = True
                feasible = torch.where(~full_mask)[0]

                for f in feasible:
                    f_i = f.item()
                    new_x, step_c = _simulate_step(state.x, n_bays, n_rows, n_tiers,
                                                   target_idx, f_i)
                    total = state.total_cost + step_c
                    future = _estimate_future(new_x.reshape(-1, n_tiers))
                    candidates.append((total + future, BState(new_x, n_bays, n_rows, n_tiers, total)))

            candidates.sort(key=lambda x: x[0])
            beam = [c[1] for c in candidates[:self.beam_width]]

        if beam:
            return min(beam, key=lambda s: s.total_cost).total_cost, 0
        return 0, 0


def _clone_env(env):
    new_env = Env(env.device,
                   env.x.reshape(1, env.n_bays, env.n_rows, env.max_tiers),
                   max_retrievals=None)
    new_env.t_pd = env.t_pd; new_env.t_acc = env.t_acc
    new_env.t_bay = env.t_bay; new_env.t_row = env.t_row
    new_env.x = env.x.clone()
    new_env.target_stack = env.target_stack.clone() if env.target_stack is not None else None
    new_env.empty = env.empty.clone()
    new_env.retrieved = env.retrieved.clone()
    new_env.curr_bay = env.curr_bay.clone()
    new_env.curr_row = env.curr_row.clone()
    new_env.relocations = env.relocations.clone()
    new_env.retrievals = env.retrievals.clone()
    if hasattr(env, 'early_stopped'):
        new_env.early_stopped = env.early_stopped.clone()
    return new_env


def _is_terminal(x_4d, n_rows):
    stacks = x_4d.reshape(-1, x_4d.shape[-1])
    return (stacks == 0).all(dim=1).all().item()


def _find_target(stacks):
    max_val = stacks.shape[0] * stacks.shape[1] + 1
    mins = torch.where(stacks == 0, torch.tensor(float('inf')), stacks).min(dim=1).values
    if (mins == float('inf')).all():
        return None
    return mins.argmin().item()


def _simulate_step(x_4d, n_bays, n_rows, n_tiers, target_idx, dest_idx):
    x = x_4d.clone().reshape(-1, n_tiers)

    # Find top container at target stack
    target = x[target_idx]
    top_tier = (target > 0).sum().int().item() - 1
    if top_tier < 0:
        return x_4d, 0
    top_val = target[top_tier].item()

    # Move to dest
    dest = x[dest_idx]
    dest_top = (dest > 0).sum().int().item()
    if dest_top >= n_tiers:
        return x_4d, 0

    x[target_idx, top_tier] = 0
    x[dest_idx, dest_top] = top_val

    # Cost estimate (approximate)
    src_bay = target_idx // n_rows + 1
    src_row = target_idx % n_rows + 1
    dst_bay = dest_idx // n_rows + 1
    dst_row = dest_idx % n_rows + 1
    cost = 3.5 * abs(src_bay - dst_bay) + 1.2 * abs(src_row - dst_row)
    if src_bay != dst_bay:
        cost += 40
    cost += 30  # handling

    # Check if target is now retrievable
    while True:
        new_target = _find_target(x)
        if new_target is None:
            break
        new_top = (x[new_target] > 0).sum().int().item() - 1
        if new_top < 0:
            break
        if new_top == 0 or x[new_target, new_top - 1].item() != 0:
            break
        # Retrieve
        ret_bay = new_target // n_rows + 1
        ret_row = new_target % n_rows + 1
        cost += 1.2 * (ret_row + ret_row) + 30
        x[new_target, new_top] = 0

    return x.reshape(1, n_bays, n_rows, n_tiers), cost


def _estimate_future(stacks):
    """Count mandatory relocations * min cost."""
    count = 0
    for s in range(stacks.shape[0]):
        stack = stacks[s]
        for t in range(stacks.shape[1]):
            if t == 0:
                continue
            c = stack[t].item()
            if c == 0:
                continue
            below = stack[:t]
            below_v = below[below > 0]
            if len(below_v) > 0 and (below_v < c).any().item():
                count += 1
    return count * (2 * 1.2 + 30)
