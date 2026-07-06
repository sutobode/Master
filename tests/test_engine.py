import sys, os, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.mcrp_inference import run_mcrp_episode
from policy.zero_shot import ZeroShotPolicy
from mcenv.mcenv import MCEnv
from strategies import RoundRobin, ZoneSplit


def test_engine_basic_run():
    policy = ZeroShotPolicy()
    x = torch.zeros(1, 2, 4, 4)
    x[0, 0, 0, :3] = torch.tensor([3., 2., 1.])
    x[0, 0, 1, :2] = torch.tensor([5., 4.])
    x[0, 1, 0, :2] = torch.tensor([6., 7.])

    env = MCEnv('cpu', x, n_cranes=2)
    strategy = RoundRobin(2, 2, 4)
    result = run_mcrp_episode(policy, env, strategy, 2, 4, 4)

    assert result['total_cost'] > 0
    assert result['n_steps'] >= 1
    assert len(result['per_crane_cost']) == 2


def test_engine_tracks_interference():
    policy = ZeroShotPolicy()
    x = torch.zeros(1, 2, 4, 4)
    x[0, 0, 0, :4] = torch.tensor([4., 3., 2., 1.])
    x[0, 0, 1, :3] = torch.tensor([5., 6., 7.])

    env = MCEnv('cpu', x, n_cranes=2)
    strategy = RoundRobin(2, 2, 4)
    result = run_mcrp_episode(policy, env, strategy, 2, 4, 4)

    assert 'n_interference' in result
    assert result['n_interference'] >= 0


def test_engine_all_strategies_run_without_error():
    """All 4 strategies produce valid output on same instance."""
    policy = ZeroShotPolicy()
    x = torch.zeros(1, 2, 4, 4)
    x[0, 0, 0, :3] = torch.tensor([1., 3., 5.])
    x[0, 1, 0, :2] = torch.tensor([2., 4.])

    from strategies import RoundRobin, ZoneSplit, LoadBalance, GreedyOptimal
    for Strategy in [RoundRobin, ZoneSplit, LoadBalance, GreedyOptimal]:
        env = MCEnv('cpu', x, n_cranes=2)
        strategy = Strategy(2, 2, 4)
        result = run_mcrp_episode(policy, env, strategy, 2, 4, 4)
        assert result['total_cost'] > 0
        assert result['n_steps'] >= 1
        assert len(result['per_crane_cost']) == 2
