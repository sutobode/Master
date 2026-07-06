import sys, os, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from policy.zero_shot import ZeroShotPolicy
from mcenv.mcenv import MCEnv
from engine.mcrp_inference import run_mcrp_episode
from strategies import RoundRobin
from bounds.lowerbound_mc import compute_lb_mc


def test_end_to_end_single_instance():
    policy = ZeroShotPolicy()
    x = torch.zeros(1, 2, 4, 4)
    x[0, 0, 0, :4] = torch.tensor([4., 3., 2., 1.])
    x[0, 0, 1, :3] = torch.tensor([5., 6., 7.])
    x[0, 1, 0, :2] = torch.tensor([8., 9.])

    env = MCEnv('cpu', x, n_cranes=2)
    strategy = RoundRobin(2, 2, 4)
    result = run_mcrp_episode(policy, env, strategy, 2, 4, 4)
    lb = compute_lb_mc(x, 2, 4, 4, n_cranes=2).item()

    assert result['total_cost'] > 0
    assert lb > 0
    gap = 100 * (result['total_cost'] - lb) / lb
    assert -50 < gap < 500, f'Gap {gap:.1f}% outside sanity range'


def test_end_to_end_parse_file():
    import glob
    from experiment import parse_instance_file, load_instance_tensor

    files = glob.glob('benchmarks/mc_instances/lee_mc/*.txt')
    if not files:
        return

    data_lines, n_cranes, crane_starts = parse_instance_file(files[0])
    assert n_cranes in [2, 3]
    assert len(crane_starts) == n_cranes
    assert len(data_lines) > 0

    dims = os.path.basename(files[0]).replace('.txt', '').split('_')[1][1:]
    n_bays = int(dims[0:2])
    n_rows = int(dims[2:4])
    n_tiers = int(dims[4:6])
    tensor = load_instance_tensor(data_lines, n_bays, n_rows, n_tiers)
    assert tensor.shape == (1, n_bays, n_rows, n_tiers)
