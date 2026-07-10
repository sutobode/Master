"""ZeroShot-only multi-crane sweep (no heuristic baselines) -- kept for
`--quick` smoke-testing (see CLAUDE.md) and as the import source for
parse_instance_file/load_instance_tensor/verify_backward_compatibility,
reused by analysis/run_multi_crane_full.py. For the full method x strategy
comparison (ZeroShot + M-Lin2015/M-Kim2016/M-Leveling, all 4 strategies, both
crane counts, in ONE csv), use:
    python -m analysis.run_multi_crane_full --dataset small
    python -m analysis.run_multi_crane_full --dataset large
"""

import os, sys, time, glob, argparse
from datetime import datetime
import torch
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from policy.zero_shot import ZeroShotPolicy
from mcenv.mcenv import MCEnv
from strategies import RoundRobin, ZoneSplit, LoadBalance, GreedyOptimal
from bounds.lowerbound_mc import compute_lb_mc
from engine.mcrp_inference import run_mcrp_episode

STRATEGY_MAP = {
    'S1': RoundRobin, 'S2': ZoneSplit,
    'S3': LoadBalance, 'S4': GreedyOptimal
}


def parse_args():
    p = argparse.ArgumentParser(description='M-CRP Zero-shot Transfer Analysis')
    p.add_argument('--instance_dir', default='benchmarks/mc_instances/lee_mc',
                   help='Directory containing M-CRP instance files')
    p.add_argument('--cranes', type=int, nargs='+', default=[2, 3],
                   help='Number of cranes to evaluate (e.g., 2 3)')
    p.add_argument('--strategies', nargs='+', default=['S1', 'S2', 'S3', 'S4'],
                   help='Strategies: S1=RoundRobin S2=ZoneSplit S3=LoadBalance S4=GreedyOptimal')
    p.add_argument('--max_instances', type=int, default=None,
                   help='Limit number of instances (for quick testing)')
    p.add_argument('--seed', type=int, default=1234)
    p.add_argument('--output', default=None,
                   help='Output CSV path (default: results/mcrp_experiment_TIMESTAMP.csv)')
    p.add_argument('--quick', action='store_true',
                   help='Quick test: 3 instances x 2 cranes x 2 strategies (12 runs)')
    return p.parse_args()


def parse_instance_file(path):
    """Parse an M-CRP layout file.

    Returns (data_lines, crane_starts) where crane_starts maps a crane
    count C -> list of start bays. One file per unique layout; the crane
    count is an experiment parameter (see benchmarks/generate_mc_instances.py).
    """
    with open(path, 'r') as f:
        lines = f.readlines()

    crane_starts = {}
    data_lines = []

    for line in lines:
        if line.startswith('# crane_start_bays_c'):
            key, _, val = line.partition('=')
            c = int(key.strip().rsplit('_c', 1)[1])
            crane_starts[c] = eval(val.strip())
        elif line.strip() and not line.startswith('#'):
            data_lines.append(line)

    return data_lines, crane_starts


def load_instance_tensor(data_lines, n_bays, n_rows, n_tiers):
    import numpy as np
    matrix = np.zeros((n_bays * n_rows, n_tiers), dtype=int)
    for line in data_lines[1:]:
        vals = list(map(int, line.split()))
        bay, stack, num_tiers = vals[:3]
        containers = vals[3:]
        unique = list(dict.fromkeys(containers))
        padded = unique + [0] * (n_tiers - len(unique))
        idx = (bay - 1) * n_rows + (stack - 1)
        matrix[idx] = padded
    tensor = torch.tensor(matrix).float().reshape(1, n_bays, n_rows, n_tiers)
    return tensor


def verify_backward_compatibility(policy):
    """Sanity check: ZeroShotPolicy(C=1) must match original model cost within 2%."""
    from model.model import Model
    from mcenv.mcenv import MCEnv
    from engine.mcrp_inference import run_mcrp_episode
    from strategies import RoundRobin
    from benchmarks.benchmarks import find_and_process_file

    try:
        x, _ = find_and_process_file('benchmarks/Lee_instances', 'random', 1, 16, 6, 1, no_print=True)
    except (FileNotFoundError, ValueError):
        print('  [SKIP] backward compat: benchmark files not found')
        return

    model_args = argparse.Namespace(
        device=torch.device('cpu'), embed_dim=128, n_encode_layers=3, n_heads=8,
        ff_hidden=512, tanh_c=10, lstm=True, bay_embedding=True,
        online=False, online_known_num=None
    )
    orig_model = Model(model_args)
    orig_model.load_state_dict(
        torch.load('baselines/models/proposed/epoch(100).pt', map_location='cpu')
    )
    orig_model.eval()
    orig_model.decoder.set_sampler('greedy')

    with torch.no_grad():
        wt_orig, _ = orig_model(x, None)

    env = MCEnv('cpu', x, n_cranes=1, crane_start_bays=[1])
    strategy = RoundRobin(1, x.shape[1], x.shape[2])
    result = run_mcrp_episode(policy, env, strategy, x.shape[1], x.shape[2], x.shape[3])

    cost_diff_pct = 100 * abs(result['total_cost'] - wt_orig[0].item()) / wt_orig[0].item()
    print(f'  Backward compat C=1: original={wt_orig[0].item():.1f}, zero-shot={result["total_cost"]:.1f}, diff={cost_diff_pct:.2f}%')
    assert cost_diff_pct < 0.01, (
        f'Backward compatibility FAILED: zero-shot cost ({result["total_cost"]:.1f}) '
        f'differs from original ({wt_orig[0].item():.1f}) by {cost_diff_pct:.2f}% '
        f'(C=1 must be numerically identical to the original Env)'
    )
    print(f'  [PASS] Backward compatibility verified (diff={cost_diff_pct:.2f}%)')


def run_experiment(args):
    torch.manual_seed(args.seed)
    policy = ZeroShotPolicy(device=torch.device('cpu'))
    print('=== Verifying backward compatibility (C=1) ===')
    verify_backward_compatibility(policy)
    print()

    files = sorted(glob.glob(os.path.join(args.instance_dir, '*.txt')))
    if args.max_instances:
        files = files[:args.max_instances]

    results = []
    start_time = time.time()
    n_total = len(files) * len(args.cranes) * len(args.strategies)
    n_done = 0

    for fpath in files:
        fname = os.path.basename(fpath)
        data_lines, crane_starts = parse_instance_file(fpath)

        parts = fname.replace('.txt', '').split('_')
        # parts: ['mc', 'R021606', '001']
        dims = parts[1][1:]
        n_bays = int(dims[0:2])
        n_rows = int(dims[2:4])
        n_tiers = int(dims[4:6])

        x = load_instance_tensor(data_lines, n_bays, n_rows, n_tiers)

        for n_cranes in args.cranes:
            # The lower bound depends only on (x, n_cranes), not on strategy;
            # compute it once per crane count instead of once per strategy.
            lb = compute_lb_mc(x, n_bays, n_rows, n_tiers, n_cranes)
            lb_work = lb['work'][0].item()
            lb_makespan = lb['makespan'][0].item()

            for sname in args.strategies:
                StrategyCls = STRATEGY_MAP[sname]
                strategy = StrategyCls(n_cranes, n_bays, n_rows)
                env = MCEnv(
                    'cpu', x, n_cranes,
                    crane_start_bays=crane_starts.get(n_cranes)
                )

                t0 = time.time()
                result = run_mcrp_episode(policy, env, strategy, n_bays, n_rows, n_tiers)
                elapsed = time.time() - t0

                gap_work = 100 * (result['total_cost'] - lb_work) / lb_work if lb_work > 0 else 0.0
                makespan = result.get('makespan')
                gap_makespan = (
                    100 * (makespan - lb_makespan) / lb_makespan
                    if makespan is not None and lb_makespan > 0 else None
                )
                # Relative epsilon: chained float/tensor accumulation across
                # env.py/mcenv.py/lowerbound_mc.py can leave a schedule that
                # exactly meets the bound a few ULPs below it; only a real
                # (relatively-large) shortfall indicates an invalid bound.
                eps = 1e-6
                assert gap_work >= -eps * max(1.0, lb_work), (
                    f'Negative work gap on {fname} C={n_cranes} {sname}: '
                    f'cost={result["total_cost"]:.1f} < LB={lb_work:.1f} — bound invalid'
                )
                if gap_makespan is not None:
                    assert gap_makespan >= -eps * max(1.0, lb_makespan), (
                        f'Negative makespan gap on {fname} C={n_cranes} {sname}: '
                        f'makespan={makespan:.1f} < LB={lb_makespan:.1f} — bound invalid'
                    )

                results.append({
                    'instance': fname,
                    'n_cranes': n_cranes,
                    'strategy': sname,
                    'total_cost': result['total_cost'],
                    'makespan': makespan,
                    'lb_work': lb_work,
                    'lb_makespan': lb_makespan,
                    'gap_work': round(max(gap_work, 0.0), 2),
                    'gap_makespan': round(max(gap_makespan, 0.0), 2) if gap_makespan is not None else None,
                    'n_steps': result['n_steps'],
                    'interference': result['n_interference'],
                    'interference_wait': round(result.get('interference_wait', 0.0), 2),
                    'a7_reassignments': result.get('a7_reassignments', 0),
                    'a7_violations': result.get('a7_violations', 0),
                    'time_s': round(elapsed, 3),
                    'per_crane_cost': str(result['per_crane_cost']),
                })

                n_done += 1
                if n_done % 10 == 0:
                    elapsed_total = time.time() - start_time
                    rate = n_done / elapsed_total
                    remaining = (n_total - n_done) / rate if rate > 0 else 0
                    print(
                        f'  [{n_done}/{n_total}] {fname} C={n_cranes} {sname} '
                        f'gap_w={gap_work:.1f}% | {remaining:.0f}s remaining'
                    )

    total = time.time() - start_time
    print(f'\nTotal: {len(results)} runs in {total:.1f}s')
    return pd.DataFrame(results)


if __name__ == '__main__':
    args = parse_args()

    if args.quick:
        print('=== QUICK MODE: 3 instances, 2 cranes, 2 strategies ===')
        args.max_instances = 3
        args.cranes = [2]
        args.strategies = ['S1', 'S2']

    os.makedirs('results', exist_ok=True)
    df = run_experiment(args)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    path = args.output or f'results/mcrp_experiment_{timestamp}.csv'
    df.to_csv(path, index=False)
    print(f'Saved: {path}')

    cols = ['gap_work'] + (['gap_makespan'] if df['gap_makespan'].notna().any() else [])
    summary = df.groupby(['n_cranes', 'strategy'])[cols].agg(['mean', 'std', 'min', 'max'])
    print('\n=== Summary ===')
    print(summary.to_string())
