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
    lb = compute_lb_mc(x, 2, 4, 4, n_cranes=2)
    lb_work = lb['work'][0].item()
    lb_makespan = lb['makespan'][0].item()

    assert result['total_cost'] > 0
    assert 0 < lb_work
    assert 0 < lb_makespan <= lb_work
    # Valid bounds: no negative gaps, and makespan cannot exceed total work.
    assert result['total_cost'] >= lb_work * (1 - 1e-9)
    assert result['makespan'] >= lb_makespan * (1 - 1e-9)
    assert result['makespan'] <= result['total_cost'] + 1e-6


def test_end_to_end_parse_file():
    import glob
    from experiment import parse_instance_file, load_instance_tensor

    files = sorted(glob.glob('benchmarks/mc_instances/lee_mc/*.txt'))
    if not files:
        return

    data_lines, crane_starts = parse_instance_file(files[0])
    assert set(crane_starts) == {2, 3}
    assert len(crane_starts[2]) == 2 and len(crane_starts[3]) == 3
    assert len(data_lines) > 0

    dims = os.path.basename(files[0]).replace('.txt', '').split('_')[1][1:]
    n_bays = int(dims[0:2])
    n_rows = int(dims[2:4])
    n_tiers = int(dims[4:6])
    tensor = load_instance_tensor(data_lines, n_bays, n_rows, n_tiers)
    assert tensor.shape == (1, n_bays, n_rows, n_tiers)
