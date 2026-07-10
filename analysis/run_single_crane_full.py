"""Setting A (single-crane, C=1): unified full comparison across BOTH dataset
scales, with the SAME 6 methods on each.

Replaces run_single_crane_v2.py (Lee only) + run_single_crane_large.py (Shin,
only Lin2015 as baseline). Those two scripts diverged in method coverage --
Shin only got 1 of 4 heuristic baselines -- which made "Setting A" look like
two different, incomparable experiments instead of one comparison run twice
at different scale. This script always runs the full 6 rows per instance:
  OriginalModel  - Shin et al. (2026) SOTA, greedy decode
  ZeroShot       - proposed method's actual code path (MCEnv n_cranes=1);
                   backward-compat contract requires this to numerically
                   match OriginalModel to <0.01% on every instance
  Lin2015 / Kim2016 / Leveling / Durasevic2025 - published heuristics

Instances are processed in PARALLEL across worker processes (each instance's
6-method computation is fully independent of every other instance -- no
shared state, no cross-instance randomness dependency). Each worker loads its
own copy of the model/policy once (via the pool initializer) and pins itself
to a single thread (torch.set_num_threads(1)) so N worker processes actually
use N cores instead of each spawning its own internal thread pool and
fighting over the same cores. Default worker count leaves 2 threads free for
the OS; override with --workers if you see thermal throttling on a long run.

RESUME: the output CSV is written incrementally -- as soon as an instance's
6 rows are computed, they're appended and fsync'd to disk immediately (not
buffered until the end). If the run is interrupted (power loss, crash,
Ctrl+C), just re-run the SAME command: instances already present in the
output CSV are detected and skipped, and only the remaining ones run. Each
instance's write is atomic (all 6 rows appended together after the worker
returns), so the output CSV never contains a partially-written instance to
worry about. Pass --fresh to ignore any existing output and start over.

Usage:
  python -m analysis.run_single_crane_full --dataset lee
      -> benchmarks/Lee_instances/, 70 instances, minutes.
  python -m analysis.run_single_crane_full --dataset shin --max_per_scale 20
      -> benchmarks/Shin_instances/, up to 160 instances (8 groups x 20),
         each instance ~20-100s+ per method x 6 methods -- can take hours,
         parallelized across --workers processes.
         Reduce --max_per_scale for a faster partial run.
Output: results/single_crane_lee.csv or results/single_crane_shin.csv
"""

import sys, os, time, argparse, itertools, multiprocessing, csv
import pandas as pd
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.model import Model
from policy.zero_shot import ZeroShotPolicy
from mcenv.mcenv import MCEnv
from engine.mcrp_inference import run_mcrp_episode
from strategies import RoundRobin
from baselines.lin2015 import Lin2015
from baselines.kim2016 import Kim2016
from baselines.leveling import Leveling
from baselines.durasevic2025 import Durasevic2025
from baselines.lowerbound import get_wt_lb
from benchmarks.benchmarks import find_and_process_file

HEURISTICS = {
    'Lin2015': Lin2015, 'Kim2016': Kim2016,
    'Leveling': Leveling, 'Durasevic2025': Durasevic2025,
}
FIELDNAMES = ['instance', 'type', 'n_bays', 'n_tiers', 'method', 'cost', 'lb', 'gap']

LEE_BAYS6 = [1, 2, 4, 6, 8, 10]
LEE_BAYS8 = [1, 2, 4, 6]
SHIN_BAYS = [20, 30]
SHIN_TIERS = [6, 8]

# Populated once per worker process by _init_worker(), so the (slow-to-load)
# model/policy are only constructed once per process, not once per instance.
_worker_model = None
_worker_policy = None


def lee_instances():
    for tier, bays in ((6, LEE_BAYS6), (8, LEE_BAYS8)):
        for bay in bays:
            for inst_type, idxs in (('random', range(1, 6)), ('upsidedown', range(1, 3))):
                for idx in idxs:
                    yield inst_type, bay, 16, tier, idx


def shin_instances(max_per_scale):
    for tier in SHIN_TIERS:
        for bay in SHIN_BAYS:
            for inst_type in ('random', 'upsidedown'):
                for idx in range(1, max_per_scale + 1):
                    yield inst_type, bay, 16, tier, idx


def load_original_model():
    args = argparse.Namespace(
        device=torch.device('cpu'), embed_dim=128, n_encode_layers=3, n_heads=8,
        ff_hidden=512, tanh_c=10, lstm=True, bay_embedding=True,
        online=False, online_known_num=None
    )
    model = Model(args)
    model.load_state_dict(torch.load('baselines/models/proposed/epoch(100).pt', map_location='cpu'))
    model.eval()
    model.decoder.set_sampler('greedy')
    return model


def _init_worker():
    global _worker_model, _worker_policy
    torch.set_num_threads(1)
    _worker_model = load_original_model()
    _worker_policy = ZeroShotPolicy(device=torch.device('cpu'))


def _process_instance(task):
    """Runs in a worker process. Returns None if the instance file doesn't
    exist (caller counts it as skipped), else {'name', 'rows', 'diff_pct'}."""
    src_dir, inst_type, bay, row, tier, idx = task
    try:
        x, name = find_and_process_file(src_dir, inst_type, bay, row, tier, idx, no_print=True)
    except (FileNotFoundError, ValueError):
        return None

    lb = float(get_wt_lb(x))
    type_label = 'R' if inst_type == 'random' else 'U'
    rows = []

    with torch.no_grad():
        wt, _ = _worker_model(x, None)
    cost = float(wt[0].item())
    rows.append({'instance': name, 'type': type_label, 'n_bays': bay, 'n_tiers': tier,
                 'method': 'OriginalModel', 'cost': round(cost, 1), 'lb': round(lb, 1),
                 'gap': round(100 * (cost - lb) / lb, 2)})

    env = MCEnv('cpu', x, n_cranes=1)
    zs_result = run_mcrp_episode(_worker_policy, env, RoundRobin(1, bay, row), bay, row, tier)
    cost_zs = zs_result['total_cost']
    diff_pct = 100 * abs(cost_zs - cost) / cost if cost > 0 else 0.0
    rows.append({'instance': name, 'type': type_label, 'n_bays': bay, 'n_tiers': tier,
                 'method': 'ZeroShot', 'cost': round(cost_zs, 1), 'lb': round(lb, 1),
                 'gap': round(100 * (cost_zs - lb) / lb, 2)})

    for mname, MCls in HEURISTICS.items():
        torch.manual_seed(1234)
        try:
            cost_h, _ = MCls().run(x)
        except Exception as e:
            print(f'  ERROR {mname} {name}: {e}')
            continue
        rows.append({'instance': name, 'type': type_label, 'n_bays': bay, 'n_tiers': tier,
                     'method': mname, 'cost': round(float(cost_h), 1), 'lb': round(lb, 1),
                     'gap': round(100 * (float(cost_h) - lb) / lb, 2)})

    return {'name': name, 'rows': rows, 'diff_pct': diff_pct}


def _resolve_task_names(src_dir, items):
    """Cheap file lookup (no model inference) to learn each task's resulting
    instance filename upfront, so already-completed instances (matched by
    that filename in an existing output CSV) can be filtered out before
    submitting work to the pool. Missing files are dropped here too, same as
    _process_instance would (but without paying for the pool round-trip)."""
    resolved = []
    for inst_type, bay, row, tier, idx in items:
        try:
            _, name = find_and_process_file(src_dir, inst_type, bay, row, tier, idx, no_print=True)
        except (FileNotFoundError, ValueError):
            continue
        resolved.append((name, (src_dir, inst_type, bay, row, tier, idx)))
    return resolved


def _backward_compat_diff_from_rows(rows):
    """Recompute max |ZeroShot - OriginalModel| cost diff from a list of row
    dicts (used to fold already-resumed rows into the assert below)."""
    by_instance = {}
    for r in rows:
        if r['method'] in ('OriginalModel', 'ZeroShot'):
            by_instance.setdefault(r['instance'], {})[r['method']] = r['cost']
    max_diff = 0.0
    for costs in by_instance.values():
        if 'OriginalModel' in costs and 'ZeroShot' in costs and costs['OriginalModel'] > 0:
            diff = 100 * abs(costs['ZeroShot'] - costs['OriginalModel']) / costs['OriginalModel']
            max_diff = max(max_diff, diff)
    return max_diff


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', choices=['lee', 'shin'], required=True)
    p.add_argument('--max_per_scale', type=int, default=20,
                    help='Shin only: instances per (bay,tier,type) group, max 20 available. '
                         'Lee always uses its fixed standard set (70 instances).')
    p.add_argument('--max_instances', type=int, default=None,
                    help='Cap total (instance, method-row) iterations for a quick smoke test on a small '
                         'subset (e.g. --max_instances 7 covers bay=1 fully: 5 random + 2 upside-down for '
                         'Lee). Default (unset) = no cap, i.e. the full dataset.')
    p.add_argument('--workers', type=int, default=None,
                    help='Number of parallel worker processes (each instance is fully independent, so '
                         'this parallelizes across instances). Default: cpu_count()-2 (leaves headroom '
                         'for the OS). Pass 1 for the old fully-sequential behavior.')
    p.add_argument('--fresh', action='store_true',
                    help='Ignore any existing output CSV and start over from scratch, instead of '
                         'resuming (skipping instances already present in it).')
    args = p.parse_args()

    if args.dataset == 'lee':
        src_dir = 'benchmarks/Lee_instances'
        gen = lee_instances()
        out = 'results/single_crane_lee.csv'
    else:
        src_dir = 'benchmarks/Shin_instances'
        gen = shin_instances(args.max_per_scale)
        out = 'results/single_crane_shin.csv'

    if args.max_instances is not None:
        gen = itertools.islice(gen, args.max_instances)
    resolved = _resolve_task_names(src_dir, gen)
    n_found = len(resolved)

    rows = []
    if args.fresh and os.path.exists(out):
        os.remove(out)
        print(f'--fresh: removed existing {out}')
    if not args.fresh and os.path.exists(out):
        existing_df = pd.read_csv(out)
        rows = existing_df.to_dict('records')
        done_instances = set(existing_df['instance'].unique())
        print(f'Resuming from {out}: {len(done_instances)} instances already done '
              f'({len(rows)} rows) -- skipping those.')
    else:
        done_instances = set()

    tasks = [task for name, task in resolved if name not in done_instances]
    n_instances_total = len(tasks)
    n_already_done = n_found - n_instances_total
    max_zs_diff_pct = _backward_compat_diff_from_rows(rows)

    if n_instances_total == 0:
        print('Nothing left to do -- all instances already completed.')
    else:
        n_workers = args.workers or max(1, (os.cpu_count() or 4) - 2)
        print(f'Using {n_workers} worker process(es) for {n_instances_total} remaining instances '
              f'({n_already_done} already done)', flush=True)

        file_is_new = not os.path.exists(out)
        os.makedirs('results', exist_ok=True)
        csv_file = open(out, 'a', newline='', encoding='utf-8')
        writer = csv.DictWriter(csv_file, fieldnames=FIELDNAMES)
        if file_is_new:
            writer.writeheader()
            csv_file.flush()

        t0 = time.time()
        n_skipped = 0
        n_done = 0

        try:
            with multiprocessing.Pool(n_workers, initializer=_init_worker) as pool:
                for result in pool.imap_unordered(_process_instance, tasks):
                    n_done += 1
                    if result is None:
                        n_skipped += 1
                        continue
                    for row in result['rows']:
                        writer.writerow(row)
                    csv_file.flush()
                    os.fsync(csv_file.fileno())
                    rows.extend(result['rows'])
                    max_zs_diff_pct = max(max_zs_diff_pct, result['diff_pct'])

                    elapsed = time.time() - t0
                    rate = elapsed / n_done
                    remaining = rate * (n_instances_total - n_done)
                    print(f'  [{n_done}/{n_instances_total}] {result["name"]} done ({elapsed:.0f}s elapsed, '
                          f'{len(rows)} total rows) | ~{remaining:.0f}s ({remaining / 60:.1f} min) remaining',
                          flush=True)
        finally:
            csv_file.close()

        if n_skipped:
            print(f'Skipped {n_skipped} missing instances')

    assert max_zs_diff_pct < 0.01, (
        f'ZeroShot(C=1) diverged from OriginalModel by up to {max_zs_diff_pct:.4f}% '
        f'-- backward-compatibility contract broken'
    )
    print(f'\nBackward-compat check: max |ZeroShot - OriginalModel| = {max_zs_diff_pct:.4f}%')

    df = pd.DataFrame(rows)
    print(f'\nTotal: {len(df)} rows in {out}')

    by_method = df.groupby('method')['gap'].agg(['mean', 'std', 'min', 'max']).round(2).sort_values('mean')
    by_type = df.groupby(['method', 'type'])['gap'].agg(['mean', 'std', 'count']).round(2)
    by_type_unstacked = df.groupby(['method', 'type'])['gap'].mean().round(2).unstack()

    print(f'\n=== Gap by method ({args.dataset}) ===')
    print(by_method.to_string())
    print('\n=== By type (Random vs Upside-down) ===')
    print(by_type_unstacked.to_string())

    report_path = f'results/single_crane_{args.dataset}_report.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f'Setting A (single-crane, C=1) -- dataset={args.dataset}\n')
        f.write(f'Total runs: {len(df)}\n')
        f.write(f'Backward-compat (max |ZeroShot - OriginalModel|): {max_zs_diff_pct:.4f}%\n')
        f.write('\n--- Gap by method ---\n')
        f.write(by_method.to_string())
        f.write('\n\n--- Gap by method x type (Random vs Upside-down), mean/std/count ---\n')
        f.write(by_type.to_string())
        f.write('\n')
    print(f'Report saved: {report_path}')


if __name__ == '__main__':
    main()
