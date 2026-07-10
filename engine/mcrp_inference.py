import torch
from mcenv.mcenv import MCEnv


def record_zeroshot_trajectory(policy, x, n_bays, n_rows, n_tiers, max_steps=None):
    """Run the ZeroShot policy ONCE against a reference n_cranes=1 MCEnv and
    record the ordered (target_stack, dest_stack) decision sequence.

    This sequence is PROVEN independent of n_cranes and crane-assignment
    strategy: MCEnv.step()'s `crane_id` only ever feeds into per-crane travel
    cost and timing bookkeeping (`crane_bays`/`crane_time`), never into
    `base_env.x` or `base_env.target_stack` -- the state transition a policy
    decision produces is identical no matter which crane "executes" it, and
    env.clear()'s auto-retrieval cascade is likewise state-driven, not
    crane-driven. Verified empirically: the recorded sequence is byte-for-byte
    identical across (n_cranes, strategy) combinations for the same instance
    (docs/superpowers/plans/2026-07-10-script-consolidation-handoff.md).

    Replaying this sequence via replay_zeroshot_episode() for every
    (n_cranes, strategy) combination -- instead of calling run_mcrp_episode()
    (which re-invokes the policy, i.e. re-runs the encoder, at every step) --
    eliminates redundant DRL inference: the expensive part (~90% of a
    ZeroShot run's wall-clock time) is computed once per instance instead of
    once per (n_cranes x strategy) combination.
    """
    if max_steps is None:
        max_steps = max(2000, n_bays * n_rows * n_tiers * 2)
    env = MCEnv('cpu', x, n_cranes=1)
    env.clear()
    dest_sequence = []
    step = 0
    while not env.terminated and step < max_steps:
        stacks = env.base_env.x[0]
        target_stack_idx = env.base_env.target_stack[0].item()
        full_mask = (stacks[:, -1] > 0).bool()
        full_mask[target_stack_idx] = True

        dest_stack = policy.get_action(
            env.get_state(), n_bays, n_rows, n_tiers,
            target_stack=target_stack_idx,
            invalid_mask=full_mask.unsqueeze(0),
            t_acc=env.t_acc, t_bay=env.t_bay, t_row=env.t_row, t_pd=env.t_pd
        )
        dest_idx = dest_stack[0, 0].item()
        dest_sequence.append(dest_idx)
        env.step(dest_stack=dest_stack, crane_id=0)
        step += 1
    return dest_sequence


def replay_zeroshot_episode(dest_sequence, env, strategy):
    """Replay a trajectory recorded by record_zeroshot_trajectory() through a
    fresh (n_cranes, strategy) MCEnv -- same bookkeeping as run_mcrp_episode(),
    without calling the policy again. `env` must wrap the SAME instance the
    trajectory was recorded from; behavior is undefined otherwise (no
    validation is done here -- callers control both, this is an internal
    optimization, not a public API meant to accept arbitrary trajectories)."""
    total_cost = 0.0
    per_crane_cost = torch.zeros(env.n_cranes)
    step = 0

    initial_cost = env.clear()
    total_cost += initial_cost.sum().item()

    for dest_idx in dest_sequence:
        target_stack_idx = env.base_env.target_stack[0].item()
        dest_stack = torch.tensor([[dest_idx]])

        requested_crane = strategy.assign(env, target_stack_idx, dest_idx)
        cost, _ = env.step(dest_stack=dest_stack, crane_id=requested_crane)
        crane_id = env.last_crane_id

        cost_val = cost[0].item()
        total_cost += cost_val
        per_crane_cost[crane_id] += cost_val
        step += 1

    assert env.terminated, (
        f'replay ended with env.terminated=False after {step} steps -- the '
        f'trajectory was recorded from a different instance than `env` wraps'
    )

    return {
        'total_cost': total_cost,
        'makespan': env.makespan,
        'n_steps': step,
        'n_interference': env.interference_events[0, :].sum().item(),
        'interference_wait': env.interference_wait,
        'a7_reassignments': env.a7_reassignments,
        'a7_violations': env.a7_violations,
        'per_crane_cost': per_crane_cost.tolist(),
    }


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
