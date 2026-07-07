"""Run full experiment in batches, saving intermediate results.

Usage:
    python run_full_experiment.py
"""

import os, sys, time, glob, torch, argparse
import pandas as pd
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from policy.zero_shot import ZeroShotPolicy
from mcenv.mcenv import MCEnv
from engine.mcrp_inference import run_mcrp_episode
from bounds.lowerbound_mc import compute_lb_mc
from strategies import RoundRobin, ZoneSplit, LoadBalance, GreedyOptimal
from experiment import parse_instance_file, load_instance_tensor, verify_backward_compatibility

STRATEGY_MAP = {
    'S1': RoundRobin, 'S2': ZoneSplit,
    'S3': LoadBalance, 'S4': GreedyOptimal
}


def run_batch(instance_files, policy, args, output_csv, batch_id, n_batches):
    results = []
    n_total = len(instance_files) * len(args.cranes) * len(args.strategies)
    n_done = 0

    for fpath in instance_files:
        fname = os.path.basename(fpath)
        try:
            data_lines, file_cranes, crane_starts = parse_instance_file(fpath)
        except Exception:
            continue

        parts = fname.replace('.txt', '').split('_')
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
                    'cost': round(result['total_cost'], 1),
                    'lb_mc': round(lb, 1),
                    'gap': round(gap, 2),
                    'n_steps': result['n_steps'],
                    'interference': result['n_interference'],
                    'time_s': round(elapsed, 3),
                })
                n_done += 1

                if n_done % 10 == 0:
                    elapsed_total = time.time() - args.start_time
                    rate = n_done / elapsed_total
                    remaining = (n_total - n_done) / rate if rate > 0 else 0
                    pct = 100 * n_done / n_total
                    print(
                        f'  [{n_done}/{n_total}] ({pct:.0f}%) '
                        f'{fname} C={n_cranes} {sname} '
                        f'gap={gap:.1f}% | {remaining:.0f}s remaining'
                    )

    return results


def main():
    torch.manual_seed(1234)

    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_dir', default='benchmarks/mc_instances/lee_mc')
    parser.add_argument('--cranes', type=int, nargs='+', default=[2, 3])
    parser.add_argument('--strategies', nargs='+', default=['S1', 'S2', 'S3', 'S4'])
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--batch_size', type=int, default=20,
                        help='Number of instances per batch (saves after each batch)')
    args = parser.parse_args()

    args.start_time = time.time()

    policy = ZeroShotPolicy(device=torch.device('cpu'))

    print('=== Verifying backward compatibility (C=1) ===')
    verify_backward_compatibility(policy)
    print()

    files = sorted(glob.glob(os.path.join(args.instance_dir, '*.txt')))
    print(f'Total instance files: {len(files)}')
    print(f'Configs: {len(args.cranes)} crane counts × {len(args.strategies)} strategies')
    print(f'Total runs: {len(files) * len(args.cranes) * len(args.strategies)}')
    print(f'Batch size: {args.batch_size} instances per save')
    print()

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_csv = f'results/mcrp_experiment_{timestamp}.csv'
    os.makedirs('results', exist_ok=True)

    all_results = []

    for batch_start in range(0, len(files), args.batch_size):
        batch_files = files[batch_start:batch_start + args.batch_size]
        batch_id = batch_start // args.batch_size + 1
        n_batches = (len(files) + args.batch_size - 1) // args.batch_size
        batch_bays = [os.path.basename(f).split('_')[1][1:3] for f in batch_files]

        print(f'\n=== Batch {batch_id}/{n_batches} '
              f'(instances {batch_start+1}-{batch_start+len(batch_files)}, '
              f'bays {min(batch_bays)}-{max(batch_bays)}) ===')

        batch_results = run_batch(
            batch_files, policy, args, output_csv,
            batch_id, n_batches
        )
        all_results.extend(batch_results)

        df = pd.DataFrame(all_results)
        df.to_csv(output_csv, index=False)
        elapsed = time.time() - args.start_time
        print(f'  [BATCH SAVED] {len(all_results)} runs to {output_csv} '
              f'({elapsed:.0f}s elapsed)')

    total_time = time.time() - args.start_time
    print(f'\n=== COMPLETE ===')
    print(f'Total: {len(all_results)} runs in {total_time:.1f}s')
    print(f'Results saved: {output_csv}')

    summary = df.groupby(['n_cranes', 'strategy'])['gap'].agg(['mean', 'std', 'min', 'max'])
    print('\n=== Summary ===')
    print(summary.to_string())


if __name__ == '__main__':
    main()
