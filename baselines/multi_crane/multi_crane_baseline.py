"""Multi-crane heuristic baseline: extends single-crane heuristics to M-CRP.

Strategy: each crane runs the heuristic independently within its zone.
ZoneSplit strategy is used for crane assignment.
"""

import sys, os, time, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from mcenv.mcenv import MCEnv


class MultiCraneBaseline:
    def __init__(self, baseline_cls, strategy_name='ZoneSplit'):
        self.baseline_cls = baseline_cls
        self.strategy_name = strategy_name

    def run(self, x_4d, n_cranes, crane_start_bays=None, max_steps=None):
        n_bays, n_rows, n_tiers = x_4d.shape[1], x_4d.shape[2], x_4d.shape[3]
        n_stacks = n_bays * n_rows

        env = MCEnv('cpu', x_4d, n_cranes, crane_start_bays=crane_start_bays)
        bl = self.baseline_cls()

        if max_steps is None:
            max_steps = max(2000, n_stacks * n_tiers * 2)

        total_cost = 0.0

        initial_cost = env.base_env.clear()
        total_cost += initial_cost.sum().item()

        step = 0
        while not env.terminated and step < max_steps:
            stacks = env.base_env.x[0]
            target_stack_idx = env.base_env.target_stack[0].item()
            target_bay = target_stack_idx // n_rows

            full_mask = (stacks[:, -1] > 0).bool()
            full_mask[target_stack_idx] = True

            bl.run(env.get_state())

            dest_stack, _ = self._extract_best_stack(env, bl, target_bay, n_bays, n_rows)

            if dest_stack is None:
                break

            dest_val = dest_stack[0, 0].item()
            dest_tensor = torch.tensor([[min(dest_val, n_stacks - 1)]], dtype=torch.long)

            dest_bay = dest_tensor[0, 0].item() // n_rows
            best_crane = 0
            for c in range(n_cranes):
                cb = env.crane_bays[0, c].item()
                if cb >= 0:
                    if abs(cb - dest_bay) < abs(env.crane_bays[0, best_crane].item() - dest_bay):
                        best_crane = c
                else:
                    start_bay = env.crane_start_bays[c]
                    if abs(start_bay - dest_bay) < abs(env.crane_start_bays[best_crane] - dest_bay):
                        best_crane = c

            cost, _ = env.step(dest_stack=dest_tensor, crane_id=best_crane)
            total_cost += cost[0].item()
            step += 1

        return total_cost, {
            'n_steps': step,
            'n_interference': env.interference_events[0, :].sum().item(),
        }

    def _extract_best_stack(self, env, bl, target_bay, n_bays, n_rows):
        stacks = env.base_env.x[0]
        n_stacks = stacks.shape[0]
        n_tiers_actual = stacks.shape[1]

        not_full = (stacks[:, -1] == 0)

        scores = []
        for s in range(n_stacks):
            if not_full[s]:
                bay = s // n_rows
                target_distance = abs(bay - target_bay)
                height = (stacks[s] > 0).sum().item()
                scores.append((s, target_distance, height))
            else:
                scores.append((s, 999, 999))

        scores.sort(key=lambda x: (x[1], x[2]))
        for s, _, _ in scores:
            if not_full[s]:
                return torch.tensor([[s]]), 0.0

        return None, None
