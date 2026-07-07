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
    with open(path, 'r') as f:
        lines = f.readlines()

    n_cranes = 2
    crane_starts = [1, 2]
    data_lines = []

    for line in lines:
        if line.startswith('# n_cranes'):
            n_cranes = int(line.split('=')[1].strip())
        elif line.startswith('# crane_start'):
            starts_str = line.split('=')[1].strip()
            crane_starts = eval(starts_str)
        elif line.strip() and not line.startswith('#'):
            data_lines.append(line)

    return data_lines, n_cranes, crane_starts


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
    assert cost_diff_pct < 2.0, (
        f'Backward compatibility FAILED: zero-shot cost ({result["total_cost"]:.1f}) '
        f'differs from original ({wt_orig[0].item():.1f}) by {cost_diff_pct:.2f}%'
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
        data_lines, file_cranes, crane_starts = parse_instance_file(fpath)

        parts = fname.replace('.txt', '').split('_')
        # parts: ['mc', 'R021606', '001', 'c2']
        dims = parts[1][1:]
        n_bays = int(dims[0:2])
        n_rows = int(dims[2:4])
        n_tiers = int(dims[4:6])

        x = load_instance_tensor(data_lines, n_bays, n_rows, n_tiers)

        for n_cranes in args.cranes:
            for sname in args.strategies:
                StrategyCls = STRATEGY_MAP[sname]
                strategy = StrategyCls(n_cranes, n_bays, n_rows)
                env = MCEnv(
                    'cpu', x, n_cranes,
                    crane_start_bays=crane_starts if n_cranes == file_cranes else None
                )

                t0 = time.time()
                result = run_mcrp_episode(policy, env, strategy, n_bays, n_rows, n_tiers)
                elapsed = time.time() - t0

                lb = compute_lb_mc(x, n_bays, n_rows, n_tiers, n_cranes).item()
                gap = 100 * (result['total_cost'] - lb) / lb if lb > 0 else 0.0

                results.append({
                    'instance': fname,
                    'n_cranes': n_cranes,
                    'strategy': sname,
                    'total_cost': result['total_cost'],
                    'lb_mc': lb,
                    'gap': round(gap, 2),
                    'n_steps': result['n_steps'],
                    'interference': result['n_interference'],
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
                        f'gap={gap:.1f}% | {remaining:.0f}s remaining'
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

    summary = df.groupby(['n_cranes', 'strategy'])['gap'].agg(['mean', 'std', 'min', 'max'])
    print('\n=== Summary ===')
    print(summary.to_string())
