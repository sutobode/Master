"""Run full experiment in batches, saving intermediate results.

Same protocol as experiment.py (Experiment 2), but batched and resumable:
each batch is saved to disk immediately, so a large sweep (e.g. the 20/30-bay
'large' scale) can be interrupted and resumed without losing progress.

Usage:
    python run_full_experiment.py --batch_size 15
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


def run_batch(instance_files, policy, args, batch_id, n_batches):
    results = []
    n_total = len(instance_files) * len(args.cranes) * len(args.strategies)
    n_done = 0

    for fpath in instance_files:
        fname = os.path.basename(fpath)
        data_lines, crane_starts = parse_instance_file(fpath)

        parts = fname.replace('.txt', '').split('_')
        dims = parts[1][1:]
        n_bays = int(dims[0:2])
        n_rows = int(dims[2:4])
        n_tiers = int(dims[4:6])

        x = load_instance_tensor(data_lines, n_bays, n_rows, n_tiers)

        for n_cranes in args.cranes:
            lb = compute_lb_mc(x, n_bays, n_rows, n_tiers, n_cranes)
            lb_work = lb['work'][0].item()
            lb_makespan = lb['makespan'][0].item()

            for sname in args.strategies:
                StrategyCls = STRATEGY_MAP[sname]
                strategy = StrategyCls(n_cranes, n_bays, n_rows)
                env = MCEnv('cpu', x, n_cranes, crane_start_bays=crane_starts.get(n_cranes))
                t0 = time.time()
                result = run_mcrp_episode(policy, env, strategy, n_bays, n_rows, n_tiers)
                elapsed = time.time() - t0

                gap_work = 100 * (result['total_cost'] - lb_work) / lb_work if lb_work > 0 else 0.0
                makespan = result.get('makespan')
                gap_makespan = (
                    100 * (makespan - lb_makespan) / lb_makespan
                    if makespan is not None and lb_makespan > 0 else None
                )
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
                    elapsed_total = time.time() - args.start_time
                    rate = n_done / elapsed_total
                    remaining = (n_total - n_done) / rate if rate > 0 else 0
                    pct = 100 * n_done / n_total
                    print(
                        f'  [{n_done}/{n_total}] ({pct:.0f}%) '
                        f'{fname} C={n_cranes} {sname} '
                        f'gap_w={gap_work:.1f}% | {remaining:.0f}s remaining'
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
    df = pd.DataFrame()

    for batch_start in range(0, len(files), args.batch_size):
        batch_files = files[batch_start:batch_start + args.batch_size]
        batch_id = batch_start // args.batch_size + 1
        n_batches = (len(files) + args.batch_size - 1) // args.batch_size
        batch_bays = [os.path.basename(f).split('_')[1][1:3] for f in batch_files]

        print(f'\n=== Batch {batch_id}/{n_batches} '
              f'(instances {batch_start+1}-{batch_start+len(batch_files)}, '
              f'bays {min(batch_bays)}-{max(batch_bays)}) ===')

        batch_results = run_batch(batch_files, policy, args, batch_id, n_batches)
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

    if len(df) > 0:
        summary = df.groupby(['n_cranes', 'strategy'])[['gap_work', 'gap_makespan']].agg(['mean', 'std', 'min', 'max'])
        print('\n=== Summary ===')
        print(summary.to_string())


if __name__ == '__main__':
    main()
