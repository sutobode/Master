"""Multi-crane heuristic baselines: per-step destination rules driven through
the SAME multi-crane loop as the DRL policy.

Each adapter implements the destination rule of a published single-crane
heuristic (its `restricted` variant: relocations only from the target stack,
matching M-CRP assumption A3 and the restriction the DRL policy operates
under). A CraneAssignmentStrategy then assigns the executing crane and MCEnv
executes the move — identical protocol to ZeroShot, so the comparison is
move-for-move fair.

The earlier revision of this module ran the ORIGINAL single-crane episode
(`bl.run(x)`) and divided its cost by the multi-crane lower bound, which is
not a multi-crane method at all; do not reintroduce that shortcut.
"""

import sys, os, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def _min_priorities(stacks, big):
    return torch.where(stacks > 0, stacks, torch.full_like(stacks, big)).amin(dim=1)


def _top_priorities(stacks):
    s, t = stacks.shape
    lens = (stacks > 0).sum(dim=1)
    top_idx = torch.clamp(lens - 1, min=0)
    return stacks[torch.arange(s), top_idx]


def _travel_time(a, b, n_rows, t_acc, t_bay, t_row):
    bay_a, row_a = a // n_rows + 1, a % n_rows + 1
    bay_b, row_b = b // n_rows + 1, b % n_rows + 1
    cost = 0.0
    if bay_a != bay_b:
        cost += t_acc + t_bay * abs(bay_a - bay_b)
    return cost + t_row * abs(row_a - row_b)


def _tie_break_by_travel(cands, target_stack, n_rows, t_acc, t_bay, t_row):
    """Break ties randomly among the minimum-travel-time candidates, matching
    the published heuristics' stochastic tie-break (baselines/lin2015.py,
    leveling.py's `choose_stack_by_travel_time` uses torch.randint) rather
    than silently favoring the lowest stack index on every tie."""
    costs = [_travel_time(target_stack, s, n_rows, t_acc, t_bay, t_row) for s in cands]
    min_cost = min(costs)
    best = [s for s, c in zip(cands, costs) if c == min_cost]
    return best[torch.randint(len(best), (1,)).item()]


def _exclude_target(cands, target_stack, rule_name):
    """Filter the target stack out of a candidate set (it can leak back in
    via the fallback branch when every OTHER stack is full and all entries
    tie at the sentinel score) and fail loudly if no destination remains,
    matching the original single-crane heuristics' behavior on a gridlocked
    yard instead of silently relocating a container onto itself."""
    filtered = [c for c in cands if c != target_stack]
    if not filtered:
        raise RuntimeError(
            f'{rule_name}: no valid relocation destination for target stack '
            f'{target_stack} — yard is gridlocked (every other stack is full)'
        )
    return filtered


class LinDest:
    """Lin et al. (2015) SSI destination rule (restricted variant)."""
    name = 'M-Lin2015'

    def __init__(self, pr=30, pb=300):
        self.pr, self.pb = pr, pb

    def select(self, stacks, target_stack, n_rows, t_acc, t_bay, t_row):
        s, t = stacks.shape
        big = s * t + 1
        valid = (stacks[:, -1] == 0)
        valid[target_stack] = False
        min_prios = _min_priorities(stacks, big)
        target_top = _top_priorities(stacks)[target_stack]

        ideal = valid & (min_prios > target_top)
        idx = torch.arange(s)
        if ideal.any():
            target_bay = target_stack // n_rows + 1
            ssi = min_prios + self.pr * (idx % n_rows + 1) + self.pb * (idx // n_rows + 1 - target_bay).abs()
            ssi = torch.where(ideal, ssi, torch.full_like(ssi, float('inf')))
            cands = torch.where(ssi == ssi.min())[0].tolist()
        else:
            mm = torch.where(valid, min_prios, torch.full_like(min_prios, -1.0))
            cands = torch.where(mm == mm.max())[0].tolist()
        cands = _exclude_target(cands, target_stack, self.name)
        return _tie_break_by_travel(cands, target_stack, n_rows, t_acc, t_bay, t_row)


class KimDest:
    """Kim et al. (2016) classification destination rule (restricted variant)."""
    name = 'M-Kim2016'

    def select(self, stacks, target_stack, n_rows, t_acc, t_bay, t_row):
        s, t = stacks.shape
        big = s * t + 1
        valid = (stacks[:, -1] == 0)
        valid[target_stack] = False
        tops = _top_priorities(stacks)
        min_prios = _min_priorities(stacks, big)
        target_top = tops[target_stack]

        non_increasing = (stacks[:, :-1] >= stacks[:, 1:]).all(dim=1)
        ideal = non_increasing & valid

        if ideal.any():
            case3 = ideal & (tops > target_top)
            if case3.any():
                tt = torch.where(case3, tops, torch.full_like(tops, float('inf')))
                cands = torch.where(tt == tt.min())[0].tolist()
            else:
                tt = torch.where(ideal, tops, torch.full_like(tops, -1.0))
                cands = torch.where(tt == tt.max())[0].tolist()
        else:
            mm = torch.where(valid, min_prios, torch.full_like(min_prios, -1.0))
            cands = torch.where(mm == mm.max())[0].tolist()
        cands = _exclude_target(cands, target_stack, self.name)
        return _tie_break_by_travel(cands, target_stack, n_rows, t_acc, t_bay, t_row)


class LevelingDest:
    """Zehendner et al. (2017) leveling rule, same-bay-restricted as in
    Shin et al. (2026)'s multi-bay adaptation."""
    name = 'M-Leveling'

    def select(self, stacks, target_stack, n_rows, t_acc, t_bay, t_row):
        lens = (stacks > 0).sum(dim=1)
        valid = (stacks[:, -1] == 0)
        valid[target_stack] = False

        target_bay = target_stack // n_rows
        bay_mask = (torch.arange(stacks.shape[0]) // n_rows) == target_bay
        same_bay = valid & bay_mask
        pool = same_bay if same_bay.any() else valid

        ll = torch.where(pool, lens, torch.full_like(lens, 10 ** 6))
        cands = torch.where(ll == ll.min())[0].tolist()
        cands = _exclude_target(cands, target_stack, self.name)
        return _tie_break_by_travel(cands, target_stack, n_rows, t_acc, t_bay, t_row)


def run_mc_heuristic_episode(rule, env, strategy, n_bays, n_rows, n_tiers, max_steps=None):
    """Drive a destination rule through MCEnv — mirror of run_mcrp_episode."""
    if max_steps is None:
        max_steps = max(2000, n_bays * n_rows * n_tiers * 2)
    total_cost = 0.0
    step = 0

    initial_cost = env.clear()
    total_cost += initial_cost.sum().item()

    while not env.terminated and step < max_steps:
        stacks = env.base_env.x[0]
        target_stack_idx = env.base_env.target_stack[0].item()

        dest = rule.select(stacks, target_stack_idx, n_rows,
                           env.t_acc, env.t_bay, env.t_row)
        crane_id = strategy.assign(env, target_stack_idx, dest)

        cost, _ = env.step(dest_stack=torch.tensor([[dest]]), crane_id=crane_id)
        total_cost += cost[0].item()
        step += 1

    return {
        'total_cost': total_cost,
        'makespan': env.makespan,
        'n_steps': step,
        'n_interference': env.interference_events[0, :].sum().item(),
        'interference_wait': env.interference_wait,
        'a7_reassignments': env.a7_reassignments,
        'a7_violations': env.a7_violations,
    }
