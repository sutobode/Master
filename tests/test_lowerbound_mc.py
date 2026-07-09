import sys, os, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bounds.lowerbound_mc import compute_lb_mc, _count_mandatory_relocations
from baselines.lowerbound import get_wt_lb
from mcenv.mcenv import MCEnv


def _greedy_dest(env_x, target_stack):
    stacks = env_x[0]
    for i in range(stacks.shape[0]):
        if i != target_stack and stacks[i, -1].item() == 0:
            return i
    raise RuntimeError('no destination')


def _episode_cost(x, n_cranes):
    """Total work of a full episode under a simple feasible policy."""
    env = MCEnv('cpu', x.clone(), n_cranes=n_cranes)
    cost = env.clear().clone().float()
    step = 0
    while not env.terminated and step < 500:
        d = _greedy_dest(env.base_env.x, env.base_env.target_stack[0].item())
        c, _ = env.step(torch.tensor([[d]]), crane_id=step % n_cranes)
        cost = cost + c
        step += 1
    return cost[0].item()


def _instances():
    x1 = torch.zeros(1, 1, 16, 6)
    x1[0, 0, 0, :5] = torch.tensor([5., 4., 3., 2., 1.])

    x2 = torch.zeros(1, 2, 4, 6)
    x2[0, 0, 0, :4] = torch.tensor([1., 4., 2., 6.])
    x2[0, 0, 2, :3] = torch.tensor([5., 3., 8.])
    x2[0, 1, 0, :3] = torch.tensor([9., 7., 10.])
    x2[0, 1, 3, :2] = torch.tensor([12., 11.])

    x3 = torch.zeros(1, 4, 4, 6)
    vals = list(range(1, 21))
    k = 0
    for b in range(4):
        for r in range(4):
            if k < len(vals):
                x3[0, b, r, 0] = float(vals[len(vals) - 1 - k])
                k += 1
    return [(x1, 1, 16, 6), (x2, 2, 4, 6), (x3, 4, 4, 6)]


def test_lb_mc_one_crane_matches_theorem2():
    """At C=1 the M-CRP bound must equal Shin et al.'s Theorem 2 bound."""
    x = torch.zeros(1, 1, 16, 6)
    x[0, 0, 0, :5] = torch.tensor([5., 4., 3., 2., 1.])

    lb = compute_lb_mc(x, 1, 16, 6, n_cranes=1)
    lb_t2 = get_wt_lb(x.reshape(1, 1, 16, 6))
    assert abs(lb['work'][0].item() - lb_t2) / lb_t2 < 1e-5
    assert abs(lb['makespan'][0].item() - lb_t2) / lb_t2 < 1e-5


def test_lb_mc_valid_no_negative_gap():
    """A valid lower bound can never exceed the cost of a feasible schedule."""
    for x, B, R, T in _instances():
        for C in (1, 2, 3):
            if C > B * R:
                continue
            lb = compute_lb_mc(x, B, R, T, n_cranes=C)
            cost = _episode_cost(x, C)
            assert lb['work'][0].item() <= cost * (1 + 1e-5), (
                f'LB_work={lb["work"][0].item():.1f} > feasible cost={cost:.1f} '
                f'(B={B}, C={C}) — bound is invalid'
            )
            assert lb['makespan'][0].item() <= lb['work'][0].item() + 1e-6


def test_lb_mc_decreases_with_more_cranes():
    x = torch.zeros(1, 2, 4, 6)
    x[0, 0, 0, :3] = torch.tensor([3., 2., 1.])
    x[0, 1, 1, :2] = torch.tensor([4., 5.])

    lb1 = compute_lb_mc(x, 2, 4, 6, n_cranes=1)
    lb2 = compute_lb_mc(x, 2, 4, 6, n_cranes=2)
    assert lb2['work'][0] <= lb1['work'][0] + 1e-6
    assert lb2['makespan'][0] <= lb1['makespan'][0] + 1e-6


def test_lb_mc_positive():
    x = torch.zeros(1, 1, 4, 4)
    x[0, 0, 0, :3] = torch.tensor([3., 2., 1.])
    lb = compute_lb_mc(x, 1, 4, 4, n_cranes=2)
    assert lb['work'][0].item() > 0
    assert lb['makespan'][0].item() > 0


def test_lb_makespan_bay_workload_term():
    """A bay whose fixed workload exceeds work/C must drive the makespan LB."""
    x = torch.zeros(1, 2, 4, 6)
    # Bay 1 heavy (deep well-ordered stack: no mandatory relocations),
    # bay 2 nearly empty.
    x[0, 0, 0, :6] = torch.tensor([6., 5., 4., 3., 2., 1.])
    x[0, 0, 1, :6] = torch.tensor([12., 11., 10., 9., 8., 7.])
    x[0, 1, 0, :1] = torch.tensor([13.])

    lb = compute_lb_mc(x, 2, 4, 6, n_cranes=3)
    assert lb['makespan'][0].item() >= lb['work'][0].item() / 3 - 1e-6


def test_count_mandatory_relocations():
    x = torch.zeros(1, 4, 6)
    x[0, 0, :3] = torch.tensor([3., 2., 1.])  # no disorder (1 on top, retrieved first)
    assert _count_mandatory_relocations(x) == 0

    x2 = torch.zeros(1, 4, 6)
    x2[0, 0, :3] = torch.tensor([1., 2., 3.])  # 1 at bottom, 2 and 3 block it
    assert _count_mandatory_relocations(x2) == 2  # 2 and 3 must move
