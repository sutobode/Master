import sys, os, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bounds.lowerbound_mc import compute_lb_mc, _count_mandatory_relocations
from baselines.lowerbound import get_wt_lb


def test_lb_mc_one_crane_matches_theorem2():
    x = torch.zeros(1, 1, 16, 6)
    x[0, 0, 0, :5] = torch.tensor([5., 4., 3., 2., 1.])

    lb_mc = compute_lb_mc(x, 1, 16, 6, n_cranes=1)
    lb_t2 = get_wt_lb(x.reshape(1, -1, 6))
    assert abs(lb_mc.item() - lb_t2) / lb_t2 < 0.1, \
        f'LB_MCRP={lb_mc:.1f}, LB_T2={lb_t2:.1f}'


def test_lb_mc_decreases_with_more_cranes():
    x = torch.zeros(1, 2, 4, 6)
    x[0, 0, 0, :3] = torch.tensor([3., 2., 1.])

    lb_1 = compute_lb_mc(x, 2, 4, 6, n_cranes=1)
    lb_2 = compute_lb_mc(x, 2, 4, 6, n_cranes=2)
    assert lb_2 <= lb_1 + 1e-6, f'LB_2={lb_2:.1f} > LB_1={lb_1:.1f}'


def test_lb_mc_positive():
    x = torch.zeros(1, 1, 4, 4)
    x[0, 0, 0, :3] = torch.tensor([3., 2., 1.])
    lb = compute_lb_mc(x, 1, 4, 4, n_cranes=2)
    assert lb.item() > 0


def test_lb_mc_interference_term():
    x_low = torch.zeros(1, 2, 4, 6)
    x_low[0, 0, 0, :3] = torch.tensor([1., 3., 4.])
    x_low[0, 1, 0, :3] = torch.tensor([2., 5., 6.])

    x_high = torch.zeros(1, 2, 4, 6)
    x_high[0, 0, 0, :6] = torch.tensor([1., 5., 6., 7., 8., 9.])
    x_high[0, 1, 0, :3] = torch.tensor([2., 3., 4.])

    lb_low = compute_lb_mc(x_low, 2, 4, 6, n_cranes=2)
    lb_high = compute_lb_mc(x_high, 2, 4, 6, n_cranes=2)
    print(f'LB_low={lb_low.item():.1f}, LB_high={lb_high.item():.1f}')
    assert lb_high > lb_low, f'High contention ({lb_high:.1f}) should be > low ({lb_low:.1f})'


def test_count_mandatory_relocations():
    x = torch.zeros(1, 4, 6)
    x[0, 0, :3] = torch.tensor([3., 2., 1.])  # no disorder (1 on top, retrieved first)
    assert _count_mandatory_relocations(x) == 0

    x2 = torch.zeros(1, 4, 6)
    x2[0, 0, :3] = torch.tensor([1., 2., 3.])  # 1 at bottom, 2 and 3 block it
    assert _count_mandatory_relocations(x2) == 2  # 2 and 3 must move
