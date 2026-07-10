"""Setting B (multi-crane, C in {2,3}): unified full comparison across BOTH
layout scales, method x strategy in ONE table.

Replaces experiment.py's sweep (ZeroShot x S1-S4 only) + run_mc_baselines_v2.py
(3 heuristics x S1-S4 only) + run_mc_large.py (ZeroShot only, S1/S2 only, no
heuristics). Those three scripts covered the SAME protocol (MCEnv + strategy +
timing model) but wrote to separate CSVs that had to be manually joined, and
diverged in scope at large scale. This script always runs, per (instance,
n_cranes, strategy):
  ZeroShot   - proposed method (the frozen DRL policy's destination choice)
  M-Lin2015 / M-Kim2016 / M-Leveling - published heuristics' destination rules,
             extended to multi-crane (baselines/multi_crane/multi_crane_baseline.py)
all driven through the SAME 4 crane-assignment strategies (S1 RoundRobin, S2
ZoneSplit, S3 LoadBalance, S4 GreedyOptimal). There is no "Original Model /
SOTA" row here: Shin et al.'s single-crane model has no multi-crane variant to
run -- that gap is exactly what this project's ZeroShot transfer fills, so
the 3 heuristic baselines are the only valid prior-art comparators at C>1.

Two independent speedups over a naive per-(instance,n_cranes,strategy,method)
loop:
  1. ZeroShot's (target_stack, dest_stack) decision sequence is provably
     independent of n_cranes/strategy (crane_id only affects per-crane
     cost/timing bookkeeping in MCEnv.step, never the state transition --
     verified in tests/test_engine.py::test_replay_matches_run_mcrp_episode_exactly).
     record_zeroshot_trajectory() runs the policy ONCE per instance;
     replay_zeroshot_episode() replays it per (n_cranes, strategy) combo
     instead of re-running the DRL policy 8x -- ~4x speedup on ZeroShot rows.
  2. Instances (files) are fully independent of each other, so they're
     processed in PARALLEL worker processes (each pinned to 1 thread via
     torch.set_num_threads(1) to avoid N workers each spawning their own
     thread pool and fighting over the same cores).

RESUME: the output CSV is written incrementally -- as soon as a file's full
(n_cranes x strategy x method) row set is computed, it's appended and fsync'd
to disk immediately. If interrupted (power loss, crash, Ctrl+C), re-run the
SAME command (same --dataset/--cranes/--strategies): a file is considered
already done if it already has exactly len(cranes)*len(strategies)*4 rows in
the output CSV, and is skipped. Changing --cranes/--strategies between an
interrupted run and its resume is NOT supported (the row-count check would
misdetect completeness) -- use --fresh in that case. Pass --fresh to ignore
any existing output and start over.

Usage:
  python -m analysis.run_multi_crane_full --dataset small
      -> benchmarks/mc_instances/lee_mc/ (70 layouts, generated already).
         70 x 2 crane-counts x 4 strategies x 4 methods = 2,240 runs.
  python -m analysis.run_multi_crane_full --dataset large --max_instances 20
      -> benchmarks/mc_instances/lee_mc_large/ (up to 160 layouts). Generate
         first if missing:
           python -c "from benchmarks.generate_mc_instances import generate_large; generate_large()"
         Same 4 methods x 4 strategies x 2 crane-counts per instance --
         each instance is far slower (1440-2880 containers); start smaller
         (--max_instances) and scale up. --workers controls parallelism.
Output: results/multi_crane_small.csv or results/multi_crane_large.csv
"""

import sys, os, time, glob, argparse, multiprocessing, csv
import pandas as pd
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcenv.mcenv import MCEnv
from bounds.lowerbound_mc import compute_lb_mc
from strategies import RoundRobin, ZoneSplit, LoadBalance, GreedyOptimal
from engine.mcrp_inference import record_zeroshot_trajectory, replay_zeroshot_episode
from baselines.multi_crane.multi_crane_baseline import (
    LinDest, KimDest, LevelingDest, run_mc_heuristic_episode
)
from experiment import parse_instance_file, load_instance_tensor, verify_backward_compatibility
from policy.zero_shot import ZeroShotPolicy

STRATEGY_MAP = {'S1': RoundRobin, 'S2': ZoneSplit, 'S3': LoadBalance, 'S4': GreedyOptimal}
RULES = [LinDest(), KimDest(), LevelingDest()]

DATASET_DIRS = {
    'small': 'benchmarks/mc_instances/lee_mc',
    'large': 'benchmarks/mc_instances/lee_mc_large',
}

FIELDNAMES = [
    'instance', 'n_cranes', 'method', 'strategy', 'total_cost', 'makespan',
    'lb_work', 'lb_makespan', 'gap_work', 'gap_makespan', 'interference',
    'interference_wait', 'a7_reassignments', 'a7_violations', 'n_steps', 'time_s',
]

# Populated once per worker process by _init_worker().
_worker_policy = None


def _init_worker():
    global _worker_policy
    torch.set_num_threads(1)
    _worker_policy = ZeroShotPolicy(device=torch.device('cpu'))


def _process_file(task):
    """Runs in a worker process: full (n_cranes x strategy x method) sweep
    for ONE instance file. Returns a list of row dicts."""
    fpath, cranes, strategy_names = task
    fname = os.path.basename(fpath)
    data_lines, crane_starts = parse_instance_file(fpath)
    dims = fname.split('_')[1][1:]
    n_bays, n_rows, n_tiers = int(dims[0:2]), int(dims[2:4]), int(dims[4:6])
    x = load_instance_tensor(data_lines, n_bays, n_rows, n_tiers)

    zs_trajectory = record_zeroshot_trajectory(_worker_policy, x, n_bays, n_rows, n_tiers)

    rows = []
    for n_cranes in cranes:
        lb = compute_lb_mc(x, n_bays, n_rows, n_tiers, n_cranes)
        lb_work, lb_makespan = lb['work'][0].item(), lb['makespan'][0].item()

        for sname in strategy_names:
            StrategyCls = STRATEGY_MAP[sname]
            methods = [('ZeroShot', None)] + [(rule.name, rule) for rule in RULES]
            for method_name, rule in methods:
                strategy = StrategyCls(n_cranes, n_bays, n_rows)
                env = MCEnv('cpu', x, n_cranes, crane_start_bays=crane_starts.get(n_cranes))

                bt0 = time.time()
                if rule is None:
                    res = replay_zeroshot_episode(zs_trajectory, env, strategy)
                else:
                    res = run_mc_heuristic_episode(rule, env, strategy, n_bays, n_rows, n_tiers)
                bt = time.time() - bt0

                gap_work = 100 * (res['total_cost'] - lb_work) / lb_work if lb_work > 0 else 0.0
                gap_makespan = 100 * (res['makespan'] - lb_makespan) / lb_makespan if lb_makespan > 0 else 0.0
                eps = 1e-6
                assert gap_work >= -eps * max(1.0, lb_work) and gap_makespan >= -eps * max(1.0, lb_makespan), (
                    f'negative gap: {fname} C={n_cranes} {method_name} {sname} '
                    f'gap_work={gap_work:.4f} gap_makespan={gap_makespan:.4f}'
                )

                rows.append({
                    'instance': fname, 'n_cranes': n_cranes, 'method': method_name, 'strategy': sname,
                    'total_cost': round(res['total_cost'], 1), 'makespan': round(res['makespan'], 1),
                    'lb_work': round(lb_work, 1), 'lb_makespan': round(lb_makespan, 1),
                    'gap_work': round(max(gap_work, 0.0), 2), 'gap_makespan': round(max(gap_makespan, 0.0), 2),
                    'interference': res.get('n_interference', 0),
                    'interference_wait': round(res.get('interference_wait', 0.0), 2),
                    'a7_reassignments': res.get('a7_reassignments', 0),
                    'a7_violations': res.get('a7_violations', 0),
                    'n_steps': res['n_steps'], 'time_s': round(bt, 3),
                })
    return fname, rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', choices=['small', 'large'], required=True)
    p.add_argument('--max_instances', type=int, default=None,
                    help='Limit number of layouts (default: all available)')
    p.add_argument('--cranes', type=int, nargs='+', default=[2, 3])
    p.add_argument('--strategies', nargs='+', default=['S1', 'S2', 'S3', 'S4'])
    p.add_argument('--workers', type=int, default=None,
                    help='Number of parallel worker processes (each instance file is fully independent). '
                         'Default: cpu_count()-2. Pass 1 for the old fully-sequential behavior.')
    p.add_argument('--fresh', action='store_true',
                    help='Ignore any existing output CSV and start over from scratch, instead of '
                         'resuming (skipping files already fully present in it).')
    args = p.parse_args()

    instance_dir = DATASET_DIRS[args.dataset]
    files = sorted(glob.glob(f'{instance_dir}/*.txt'))
    if not files:
        raise SystemExit(
            f'{instance_dir}/ is empty.' + (
                '\nGenerate it first:\n'
                '  python -c "from benchmarks.generate_mc_instances import generate_large; generate_large()"'
                if args.dataset == 'large' else
                '\nGenerate it first:\n  python -m benchmarks.generate_mc_instances'
            )
        )
    if args.max_instances:
        files = files[:args.max_instances]

    policy = ZeroShotPolicy(device=torch.device('cpu'))
    print('=== Verifying backward compatibility (C=1) ===')
    verify_backward_compatibility(policy)
    print()

    out = f'results/multi_crane_{args.dataset}.csv'
    n_per_file = len(args.cranes) * len(args.strategies) * (1 + len(RULES))

    rows = []
    if args.fresh and os.path.exists(out):
        os.remove(out)
        print(f'--fresh: removed existing {out}')
    if not args.fresh and os.path.exists(out):
        existing_df = pd.read_csv(out)
        rows = existing_df.to_dict('records')
        counts = existing_df.groupby('instance').size()
        done_instances = set(counts[counts == n_per_file].index)
        n_partial = (counts != n_per_file).sum()
        print(f'Resuming from {out}: {len(done_instances)} files already done ({len(rows)} rows).')
        if n_partial:
            print(f'  NOTE: {n_partial} file(s) have a different row count than expected for the '
                  f'current --cranes/--strategies ({n_per_file}/file) -- treating as NOT done and '
                  f're-running them; this WILL duplicate their old (incompatible) rows in {out}. If '
                  f'you changed --cranes/--strategies since the interrupted run, use --fresh instead.')
    else:
        done_instances = set()

    files = [f for f in files if os.path.basename(f) not in done_instances]

    n_workers = args.workers or max(1, (os.cpu_count() or 4) - 2)
    n_files_total = len(files)

    if n_files_total == 0:
        print('Nothing left to do -- all files already completed.')
    else:
        print(f'Using {n_workers} worker process(es) for {n_files_total} remaining files', flush=True)

        file_is_new = not os.path.exists(out)
        os.makedirs('results', exist_ok=True)
        csv_file = open(out, 'a', newline='', encoding='utf-8')
        writer = csv.DictWriter(csv_file, fieldnames=FIELDNAMES)
        if file_is_new:
            writer.writeheader()
            csv_file.flush()

        t0 = time.time()
        n_files_done = 0
        n_total_remaining = n_files_total * n_per_file
        tasks = [(fpath, args.cranes, args.strategies) for fpath in files]

        try:
            with multiprocessing.Pool(n_workers, initializer=_init_worker) as pool:
                for fname, file_rows in pool.imap_unordered(_process_file, tasks):
                    for row in file_rows:
                        writer.writerow(row)
                    csv_file.flush()
                    os.fsync(csv_file.fileno())
                    rows.extend(file_rows)
                    n_files_done += 1

                    elapsed = time.time() - t0
                    rate = elapsed / n_files_done
                    remaining = rate * (n_files_total - n_files_done)
                    print(f'  [{n_files_done}/{n_files_total} files, '
                          f'{n_files_done * n_per_file}/{n_total_remaining} runs this session] {fname} done '
                          f'({elapsed:.0f}s elapsed) | ~{remaining:.0f}s ({remaining / 60:.1f} min) remaining',
                          flush=True)
        finally:
            csv_file.close()

    df = pd.DataFrame(rows)
    print(f'\nTotal: {len(df)} rows in {out}')

    for gap_col in ('gap_work', 'gap_makespan'):
        print(f'\n=== {gap_col} (headline: method x S2 only) ===')
        s2 = df[df['strategy'] == 'S2']
        print(s2.groupby(['n_cranes', 'method'])[gap_col].agg(['mean', 'std']).round(2).to_string())
        print(f'\n=== {gap_col} (full method x strategy matrix) ===')
        print(df.groupby(['n_cranes', 'method', 'strategy'])[gap_col].mean().round(2).unstack().to_string())


if __name__ == '__main__':
    main()
