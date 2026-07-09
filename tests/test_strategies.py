import sys, os, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from strategies import RoundRobin, ZoneSplit, LoadBalance, GreedyOptimal


def make_mock_env(n_cranes=2, n_bays=4):
    class MockEnv:
        t_acc = 40
        t_bay = 3.5
        t_row = 1.2
        t_pd = 30

        def _a7_compatible(self, crane_id, dest_bay):
            for c in range(self.n_cranes):
                if c == crane_id:
                    continue
                other = self._nominal_bay(c)
                if c < crane_id and other > dest_bay:
                    return False
                if c > crane_id and other < dest_bay:
                    return False
            return True

        def _nominal_bay(self, c):
            b = self.crane_bays[0, c].item()
            return self.crane_start_bays[c] if b < 0 else b

        def _bay_ready(self, bays):
            return max(self.bay_free_time[b] for b in bays)

    env = MockEnv()
    env.n_cranes = n_cranes
    env.n_bays = n_bays
    env.n_rows = 4
    env.crane_bays = torch.full((1, n_cranes), -1)
    env.crane_rows = torch.full((1, n_cranes), -1)
    env.crane_start_bays = [1 + c * max(1, n_bays // n_cranes) for c in range(n_cranes)]
    env.crane_time = [0.0] * n_cranes
    env.bay_free_time = [0.0] * (n_bays + 1)
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


def test_load_balance_picks_least_busy_crane():
    s = LoadBalance(2, 4, 4)
    env = make_mock_env(2, 4)
    env.crane_time = [100.0, 5.0]
    assert s.assign(env, 0, 0) == 1
    env.crane_time = [5.0, 100.0]
    assert s.assign(env, 0, 0) == 0


def test_load_balance_differs_from_round_robin():
    """With uneven busy times, LoadBalance must NOT cycle like RoundRobin
    (the earlier count-based version degenerated to RoundRobin exactly)."""
    s = LoadBalance(2, 4, 4)
    env = make_mock_env(2, 4)
    env.crane_time = [0.0, 50.0]
    picks = [s.assign(env, 0, 0) for _ in range(4)]
    assert picks == [0, 0, 0, 0]


def test_greedy_optimal_returns_valid_crane():
    s = GreedyOptimal(2, 4, 4)
    env = make_mock_env(2, 4)
    result = s.assign(env, target_stack=0, dest_stack=5)
    assert 0 <= result < 2


def test_greedy_optimal_prefers_earliest_finish():
    s = GreedyOptimal(2, 4, 4)
    env = make_mock_env(2, 4)
    # Crane 0 heavily loaded -> crane 1 finishes earlier despite travel.
    env.crane_time = [1000.0, 0.0]
    env.crane_bays[0, 0] = 1
    env.crane_rows[0, 0] = 1
    env.crane_bays[0, 1] = 3
    env.crane_rows[0, 1] = 1
    # target in bay 2 (stack 4..7), dest in bay 2 as well
    assert s.assign(env, target_stack=4, dest_stack=5) == 1


def test_strategies_share_interface():
    env = make_mock_env(2, 4)
    for Strategy in [RoundRobin, ZoneSplit, LoadBalance, GreedyOptimal]:
        s = Strategy(2, 4, 4)
        cid = s.assign(env, target_stack=0, dest_stack=5)
        assert 0 <= cid < 2, f'{s.name} returned invalid crane {cid}'
