import sys, os, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from strategies import RoundRobin, ZoneSplit, LoadBalance, GreedyOptimal


def make_mock_env(n_cranes=2, n_bays=4):
    class MockEnv:
        pass
    env = MockEnv()
    env.n_cranes = n_cranes
    env.n_bays = n_bays
    env.n_rows = 4
    env.crane_bays = torch.full((1, n_cranes), -1)
    env.crane_rows = torch.full((1, n_cranes), -1)
    env.crane_start_bays = [1, 3]
    class BaseEnv:
        t_acc = 40
        t_bay = 3.5
        t_row = 1.2
        t_pd = 30
    env.base_env = BaseEnv()
    return env


def test_round_robin_cycles():
    s = RoundRobin(3, 4, 4)
    env = make_mock_env(3, 4)
    assert s.assign(env, 0, 0) == 0
    assert s.assign(env, 0, 0) == 1
    assert s.assign(env, 0, 0) == 2
    assert s.assign(env, 0, 0) == 0


def test_zone_split_by_target_bay():
    s = ZoneSplit(2, 4, 4)
    env = make_mock_env(2, 4)
    assert s.assign(env, target_stack=0, dest_stack=5) == 0
    assert s.assign(env, target_stack=10, dest_stack=5) == 1


def test_load_balance_distributes():
    s = LoadBalance(2, 4, 4)
    env = make_mock_env(2, 4)
    ids = [s.assign(env, 0, 0) for _ in range(10)]
    assert abs(ids.count(0) - ids.count(1)) <= 1


def test_greedy_optimal_returns_valid_crane():
    s = GreedyOptimal(2, 4, 4)
    env = make_mock_env(2, 4)
    result = s.assign(env, target_stack=0, dest_stack=5)
    assert 0 <= result < 2


def test_strategies_share_interface():
    env = make_mock_env(2, 4)
    for Strategy in [RoundRobin, ZoneSplit, LoadBalance, GreedyOptimal]:
        s = Strategy(2, 4, 4)
        cid = s.assign(env, target_stack=0, dest_stack=5)
        assert 0 <= cid < 2, f'{s.name} returned invalid crane {cid}'
