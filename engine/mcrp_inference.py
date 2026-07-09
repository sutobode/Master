import torch


def run_mcrp_episode(policy, env, strategy, n_bays, n_rows, n_tiers, max_steps=None):
    if max_steps is None:
        max_steps = max(2000, n_bays * n_rows * n_tiers * 2)
    total_cost = 0.0
    per_crane_cost = torch.zeros(env.n_cranes)
    trajectory = []
    step = 0

    # Initialize: retrieve all immediately accessible containers.
    # env.clear() binds the resulting crane position (crane 0 by convention)
    # so the first relocation pays approach travel from the correct location.
    initial_cost = env.clear()
    total_cost += initial_cost.sum().item()

    while not env.terminated and step < max_steps:
        state = env.get_state()

        stacks = env.base_env.x[0]
        full_mask = (stacks[:, -1] > 0).bool()
        target_stack_idx = env.base_env.target_stack[0].item()
        full_mask[target_stack_idx] = True

        dest_stack = policy.get_action(
            state, n_bays, n_rows, n_tiers,
            target_stack=target_stack_idx,
            invalid_mask=full_mask.unsqueeze(0),
            t_acc=env.t_acc, t_bay=env.t_bay, t_row=env.t_row, t_pd=env.t_pd
        )

        requested_crane = strategy.assign(env, target_stack_idx, dest_stack[0, 0].item())

        cost, _ = env.step(dest_stack=dest_stack, crane_id=requested_crane)
        # env.step() may override the strategy's choice for A7 compliance;
        # attribute cost/logs to the crane that actually executed the move.
        crane_id = env.last_crane_id

        cost_val = cost[0].item()
        total_cost += cost_val
        per_crane_cost[crane_id] += cost_val
        trajectory.append({
            'step': step, 'crane': crane_id,
            'dest_bay': dest_stack[0, 0].item() // n_rows,
            'cost': cost_val
        })
        step += 1

    return {
        'total_cost': total_cost,
        'makespan': env.makespan,
        'n_steps': step,
        'n_interference': env.interference_events[0, :].sum().item(),
        'interference_wait': env.interference_wait,
        'a7_reassignments': env.a7_reassignments,
        'a7_violations': env.a7_violations,
        'per_crane_cost': per_crane_cost.tolist(),
        'trajectory': trajectory
    }
