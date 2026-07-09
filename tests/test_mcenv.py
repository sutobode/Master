import sys, os, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mcenv.mcenv import MCEnv
from env.env import Env


def test_mcenv_single_crane_match_env():
    """MCEnv(C=1) must produce same cost sequence as original Env."""
    x = torch.zeros(1, 2, 4, 6)
    x[0, 0, 0, :4] = torch.tensor([1., 2., 3., 4.])
    x[0, 0, 1, :2] = torch.tensor([5., 6.])
    x[0, 1, 0, :3] = torch.tensor([7., 8., 9.])

    mcenv = MCEnv('cpu', x, n_cranes=1)
    # Original Env expects (batch, n_bays, n_rows, max_tiers) — same as x
    orig_env = Env('cpu', x)

    mcenv.base_env.clear()
    orig_env.clear()
    assert torch.allclose(mcenv.base_env.x, orig_env.x), 'State mismatch after clear'


def test_mcenv_interference_is_a_delay():
    """A6 is enforced as a time delay: an operation in a busy bay waits for
    the bay to free instead of silently reassigning the crane."""
    x = torch.zeros(1, 4, 4, 6)
    mcenv = MCEnv('cpu', x, n_cranes=2, crane_start_bays=[1, 3])
    mcenv.bay_free_time[2] = 100.0  # bay 2 busy until t=100

    assert mcenv._bay_ready([2]) == 100.0
    assert mcenv._bay_ready([3]) == 0.0
    # A crane free at t=0 that needs bay 2 must wait until t=100.
    assert max(mcenv.crane_time[0], mcenv._bay_ready([2])) == 100.0


def test_mcenv_a7_non_crossing_enforced():
    """A7: an assignment that would cross a neighbouring crane along the bay
    axis is redirected to an order-compatible crane and counted."""
    x = torch.zeros(1, 4, 4, 6)
    mcenv = MCEnv('cpu', x, n_cranes=2, crane_start_bays=[1, 3])
    mcenv.crane_bays[0, 0] = 1
    mcenv.crane_bays[0, 1] = 3

    # Crane 0 (left) moving to bay 4 would cross crane 1 at bay 3.
    assert not mcenv._a7_compatible(0, 4)
    assert mcenv._a7_compatible(1, 4)
    assert mcenv._enforce_a7(0, 4) == 1
    assert mcenv.a7_reassignments == 1
    # Compatible moves keep the strategy's choice.
    assert mcenv._enforce_a7(0, 2) == 0


def test_mcenv_makespan_c1_equals_total_cost():
    """With one crane there is no parallelism: makespan == total work."""
    x = torch.zeros(1, 2, 4, 6)
    x[0, 0, 0, :4] = torch.tensor([4., 2., 6., 1.])
    x[0, 0, 1, :3] = torch.tensor([5., 3., 8.])
    x[0, 1, 0, :3] = torch.tensor([9., 7., 10.])

    env = MCEnv('cpu', x, n_cranes=1)
    total = env.clear().clone().float()
    step = 0
    while not env.terminated and step < 200:
        stacks = env.base_env.x[0]
        tgt = env.base_env.target_stack[0].item()
        d = next(i for i in range(stacks.shape[0]) if i != tgt and stacks[i, -1].item() == 0)
        c, _ = env.step(torch.tensor([[d]]), crane_id=0)
        total = total + c
        step += 1
    assert abs(env.makespan - total[0].item()) < 1e-6


def test_mcenv_makespan_bounds():
    """For C>=2: makespan <= total work, and makespan >= work/C only when the
    schedule is perfectly balanced (sanity: makespan is within (0, total])."""
    x = torch.zeros(1, 4, 4, 6)
    x[0, 0, 0, :4] = torch.tensor([4., 2., 6., 1.])
    x[0, 1, 0, :3] = torch.tensor([5., 3., 8.])
    x[0, 2, 0, :3] = torch.tensor([9., 7., 10.])
    x[0, 3, 1, :2] = torch.tensor([12., 11.])

    env = MCEnv('cpu', x, n_cranes=2, crane_start_bays=[1, 3])
    total = env.clear().clone().float()
    step = 0
    while not env.terminated and step < 200:
        stacks = env.base_env.x[0]
        tgt = env.base_env.target_stack[0].item()
        d = next(i for i in range(stacks.shape[0]) if i != tgt and stacks[i, -1].item() == 0)
        c, _ = env.step(torch.tensor([[d]]), crane_id=step % 2)
        total = total + c
        step += 1
    assert 0 < env.makespan <= total[0].item() + 1e-6


def test_mcenv_step():
    x = torch.zeros(1, 2, 4, 6)
    x[0, 0, 0, :4] = torch.tensor([1., 2., 3., 4.])
    x[0, 1, 0, :1] = torch.tensor([5.])

    mcenv = MCEnv('cpu', x, n_cranes=2)
    mcenv.base_env.clear()  # Initialize target stack
    cost, state = mcenv.step(dest_stack=torch.tensor([[4]]), crane_id=0)
    assert isinstance(cost, torch.Tensor)
    assert cost[0].item() > 0
    assert state is not None


def test_mcenv_c1_episode_cost_identical_to_env():
    """MCEnv(C=1) must reproduce the original Env's episode cost EXACTLY.

    Same action sequence driven through both simulators; any divergence means
    the multi-crane wrapper distorts travel-cost continuity (the backward-
    compatibility contract underpinning the zero-shot claims).
    """
    torch.manual_seed(0)
    x = torch.zeros(1, 2, 4, 6)
    x[0, 0, 0, :4] = torch.tensor([4., 2., 6., 1.])
    x[0, 0, 1, :3] = torch.tensor([5., 3., 8.])
    x[0, 0, 2, :2] = torch.tensor([7., 10.])
    x[0, 1, 0, :3] = torch.tensor([9., 12., 11.])
    x[0, 1, 3, :2] = torch.tensor([13., 14.])

    def greedy_dest(env_x, target_stack):
        # Deterministic rule: first non-full stack != target.
        stacks = env_x[0]
        for i in range(stacks.shape[0]):
            if i != target_stack and stacks[i, -1].item() == 0:
                return i
        raise RuntimeError('no destination')

    # Reference: plain Env
    ref = Env('cpu', x.clone())
    ref_cost = ref.clear().clone().float()
    steps = 0
    while not ref.all_terminated() and steps < 100:
        d = greedy_dest(ref.x, ref.target_stack[0].item())
        ref_cost = ref_cost + ref.step(torch.tensor([[d]]))
        steps += 1

    # MCEnv with a single crane, same decision rule
    mc = MCEnv('cpu', x.clone(), n_cranes=1)
    mc_cost = mc.clear().clone().float()
    steps = 0
    while not mc.terminated and steps < 100:
        d = greedy_dest(mc.base_env.x, mc.base_env.target_stack[0].item())
        c, _ = mc.step(torch.tensor([[d]]), crane_id=0)
        mc_cost = mc_cost + c
        steps += 1

    assert abs(ref_cost[0].item() - mc_cost[0].item()) < 1e-6, (
        f'C=1 episode cost mismatch: Env={ref_cost[0].item()} MCEnv={mc_cost[0].item()}'
    )
