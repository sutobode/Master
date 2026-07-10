import sys, os, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.mcrp_inference import run_mcrp_episode, record_zeroshot_trajectory, replay_zeroshot_episode
from policy.zero_shot import ZeroShotPolicy
from mcenv.mcenv import MCEnv
from strategies import RoundRobin, ZoneSplit, LoadBalance, GreedyOptimal


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


def test_replay_matches_run_mcrp_episode_exactly():
    """record_zeroshot_trajectory + replay_zeroshot_episode (the optimization
    that avoids re-running the DRL policy per n_cranes/strategy combo) must
    reproduce run_mcrp_episode()'s numbers EXACTLY -- this is the correctness
    contract the speedup depends on. Checked across several (n_cranes,
    strategy) combinations on one small instance."""
    from benchmarks.benchmarks import find_and_process_file
    policy = ZeroShotPolicy()
    x, _ = find_and_process_file('benchmarks/Lee_instances', 'random', 1, 16, 6, 1, no_print=True)

    dest_sequence = record_zeroshot_trajectory(policy, x, 1, 16, 6)
    assert len(dest_sequence) >= 1

    for n_cranes, Strategy in [(2, RoundRobin), (3, ZoneSplit), (2, LoadBalance), (3, GreedyOptimal)]:
        env_direct = MCEnv('cpu', x, n_cranes=n_cranes)
        strategy_direct = Strategy(n_cranes, 1, 16)
        direct = run_mcrp_episode(policy, env_direct, strategy_direct, 1, 16, 6)

        env_replay = MCEnv('cpu', x, n_cranes=n_cranes)
        strategy_replay = Strategy(n_cranes, 1, 16)
        replayed = replay_zeroshot_episode(dest_sequence, env_replay, strategy_replay)

        assert replayed['total_cost'] == direct['total_cost'], f'{n_cranes},{Strategy}: total_cost mismatch'
        assert replayed['makespan'] == direct['makespan'], f'{n_cranes},{Strategy}: makespan mismatch'
        assert replayed['n_steps'] == direct['n_steps']
        assert replayed['interference_wait'] == direct['interference_wait']
        assert replayed['a7_reassignments'] == direct['a7_reassignments']
        assert replayed['per_crane_cost'] == direct['per_crane_cost']
