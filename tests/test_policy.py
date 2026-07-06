import sys, os, argparse, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from policy.zero_shot import ZeroShotPolicy
from benchmarks.benchmarks import find_and_process_file
from model.model import Model


MODEL_ARGS = argparse.Namespace(
    device=torch.device('cpu'), embed_dim=128, n_encode_layers=3, n_heads=8,
    ff_hidden=512, tanh_c=10, lstm=True, bay_embedding=True,
    online=False, online_known_num=None
)


def test_policy_loads():
    p = ZeroShotPolicy()
    assert p.encoder is not None
    assert hasattr(p, 'W_target')


def test_policy_get_scores():
    p = ZeroShotPolicy()
    x, _ = find_and_process_file('benchmarks/Lee_instances', 'random', 1, 16, 6, 1, no_print=True)
    scores = p.get_scores(x, 1, 16, 6, target_stack=0)
    assert scores.shape == (1, 16)
    assert torch.allclose(scores.exp().sum(dim=1), torch.tensor([1.0]), atol=1e-5)


def test_policy_get_action():
    p = ZeroShotPolicy()
    x, _ = find_and_process_file('benchmarks/Lee_instances', 'random', 1, 16, 6, 1, no_print=True)
    action = p.get_action(x, 1, 16, 6, target_stack=3)
    assert action.shape == (1, 1)
    assert 0 <= action[0, 0].item() < 16


def test_policy_action_with_mask():
    p = ZeroShotPolicy()
    x, _ = find_and_process_file('benchmarks/Lee_instances', 'random', 1, 16, 6, 1, no_print=True)
    mask = torch.zeros(1, 16)
    mask[0, :8] = 1.0
    action = p.get_action(x, 1, 16, 6, target_stack=10, invalid_mask=mask)
    assert action[0, 0].item() >= 8


def test_policy_matches_original_model_cost():
    """CRITICAL: ZeroShotPolicy + MCEnv(C=1) must produce same cost as original model.

    This verifies the scorer extraction is correct. If this test fails,
    the entire zero-shot experiment is invalid.
    """
    from mcenv.mcenv import MCEnv
    from engine.mcrp_inference import run_mcrp_episode
    from strategies import RoundRobin

    p = ZeroShotPolicy()
    x, _ = find_and_process_file('benchmarks/Lee_instances', 'random', 1, 16, 6, 1, no_print=True)

    orig_model = Model(MODEL_ARGS)
    orig_model.load_state_dict(
        torch.load('baselines/models/proposed/epoch(100).pt', map_location='cpu')
    )
    orig_model.eval()
    orig_model.decoder.set_sampler('greedy')

    with torch.no_grad():
        wt_orig, _ = orig_model(x, None)

    env = MCEnv('cpu', x, n_cranes=1, crane_start_bays=[1])
    strategy = RoundRobin(1, 1, 16)
    result = run_mcrp_episode(p, env, strategy, 1, 16, 6)

    cost_diff_pct = 100 * abs(result['total_cost'] - wt_orig[0].item()) / wt_orig[0].item()
    print(f'Original cost: {wt_orig[0].item():.1f}, Zero-shot cost: {result["total_cost"]:.1f}, '
          f'diff: {cost_diff_pct:.2f}%')
    assert cost_diff_pct < 2.0, (
        f'Zero-shot cost ({result["total_cost"]:.1f}) differs from original '
        f'({wt_orig[0].item():.1f}) by {cost_diff_pct:.2f}%'
    )
