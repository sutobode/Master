"""Behavioral analysis: measures per-step destination locality for ZeroShot
vs Leveling, providing direct quantitative evidence for the mechanism behind
the scale-boundary finding (docs/superpowers/plans/2026-07-11-master-plan.md
Phase 1.1): ZeroShot's destination choice is an unconstrained global argmax
over every stack in the yard (policy/zero_shot.py::get_scores), while
Leveling explicitly restricts to same-bay candidates first, falling back to
the whole yard only when none are free (baselines/multi_crane/
multi_crane_baseline.py::LevelingDest). This script records, per step, the
bay-distance between the target (source) stack and the chosen destination
for both methods, on the same Lee/Shin instances used in Setting A.

RESUME: output CSV is written incrementally per completed instance (same
pattern as analysis/run_single_crane_full.py); re-run the same command to
continue an interrupted run. Pass --fresh to start over.

Usage:
  python -m analysis.behavioral_analysis --dataset lee
  python -m analysis.behavioral_analysis --dataset shin --max_per_scale 20 --workers 6
"""
import sys, os, argparse, csv, multiprocessing
import torch
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcenv.mcenv import MCEnv
from policy.zero_shot import ZeroShotPolicy
from baselines.multi_crane.multi_crane_baseline import LevelingDest
from benchmarks.benchmarks import find_and_process_file

LEE_BAYS6 = [1, 2, 4, 6, 8, 10]
LEE_BAYS8 = [1, 2, 4, 6]
SHIN_BAYS = [20, 30]
SHIN_TIERS = [6, 8]
FIELDNAMES = ['instance', 'type', 'n_bays', 'n_tiers',
              'zs_n_steps', 'zs_mean_bay_dist', 'zs_same_bay_rate',
              'lv_n_steps', 'lv_mean_bay_dist', 'lv_same_bay_rate']

_worker_policy = None


def lee_instances():
    for tier, bays in ((6, LEE_BAYS6), (8, LEE_BAYS8)):
        for bay in bays:
            for inst_type, idxs in (('random', range(1, 6)), ('upsidedown', range(1, 3))):
                for idx in idxs:
                    yield 'benchmarks/Lee_instances', inst_type, bay, 16, tier, idx


def shin_instances(max_per_scale):
    for tier in SHIN_TIERS:
        for bay in SHIN_BAYS:
            for inst_type in ('random', 'upsidedown'):
                for idx in range(1, max_per_scale + 1):
                    yield 'benchmarks/Shin_instances', inst_type, bay, 16, tier, idx


def _init_worker():
    global _worker_policy
    torch.set_num_threads(1)
    _worker_policy = ZeroShotPolicy(device=torch.device('cpu'))


def zeroshot_step_distances(policy, x, n_bays, n_rows, n_tiers, max_steps=None):
    if max_steps is None:
        max_steps = max(2000, n_bays * n_rows * n_tiers * 2)
    env = MCEnv('cpu', x, n_cranes=1)
    env.clear()
    distances = []
    step = 0
    while not env.terminated and step < max_steps:
        stacks = env.base_env.x[0]
        target_stack_idx = env.base_env.target_stack[0].item()
        full_mask = (stacks[:, -1] > 0).bool()
        full_mask[target_stack_idx] = True
        dest_stack = policy.get_action(
            env.get_state(), n_bays, n_rows, n_tiers,
            target_stack=target_stack_idx, invalid_mask=full_mask.unsqueeze(0),
            t_acc=env.t_acc, t_bay=env.t_bay, t_row=env.t_row, t_pd=env.t_pd
        )
        dest_idx = dest_stack[0, 0].item()
        distances.append(abs(dest_idx // n_rows - target_stack_idx // n_rows))
        env.step(dest_stack=dest_stack, crane_id=0)
        step += 1
    return distances


def leveling_step_distances(x, n_bays, n_rows, n_tiers, max_steps=None):
    if max_steps is None:
        max_steps = max(2000, n_bays * n_rows * n_tiers * 2)
    env = MCEnv('cpu', x, n_cranes=1)
    env.clear()
    rule = LevelingDest()
    distances = []
    step = 0
    while not env.terminated and step < max_steps:
        stacks = env.base_env.x[0]
        target_stack_idx = env.base_env.target_stack[0].item()
        dest_idx = rule.select(stacks, target_stack_idx, n_rows, env.t_acc, env.t_bay, env.t_row)
        distances.append(abs(dest_idx // n_rows - target_stack_idx // n_rows))
        env.step(dest_stack=torch.tensor([[dest_idx]]), crane_id=0)
        step += 1
    return distances


def _process(task):
    src_dir, inst_type, bay, row, tier, idx = task
    try:
        x, name = find_and_process_file(src_dir, inst_type, bay, row, tier, idx, no_print=True)
    except (FileNotFoundError, ValueError):
        return None
    zs_d = zeroshot_step_distances(_worker_policy, x, bay, row, tier)
    lv_d = leveling_step_distances(x, bay, row, tier)
    type_label = 'R' if inst_type == 'random' else 'U'
    return {
        'instance': name, 'type': type_label, 'n_bays': bay, 'n_tiers': tier,
        'zs_n_steps': len(zs_d),
        'zs_mean_bay_dist': round(sum(zs_d) / len(zs_d), 4) if zs_d else 0,
        'zs_same_bay_rate': round(sum(1 for d in zs_d if d == 0) / len(zs_d), 4) if zs_d else 0,
        'lv_n_steps': len(lv_d),
        'lv_mean_bay_dist': round(sum(lv_d) / len(lv_d), 4) if lv_d else 0,
        'lv_same_bay_rate': round(sum(1 for d in lv_d if d == 0) / len(lv_d), 4) if lv_d else 0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', choices=['lee', 'shin', 'both'], default='both')
    ap.add_argument('--max_per_scale', type=int, default=20)
    ap.add_argument('--workers', type=int, default=max(1, multiprocessing.cpu_count() - 2))
    ap.add_argument('--out', default='results/behavioral_analysis.csv')
    ap.add_argument('--fresh', action='store_true')
    args = ap.parse_args()

    tasks = []
    if args.dataset in ('lee', 'both'):
        tasks += list(lee_instances())
    if args.dataset in ('shin', 'both'):
        tasks += list(shin_instances(args.max_per_scale))

    done_names = set()
    if os.path.exists(args.out) and not args.fresh:
        existing = pd.read_csv(args.out)
        done_names = set(existing['instance'])
        print(f'Resuming: {len(done_names)} instances already done -- skipping those.')

    mode = 'w' if (args.fresh or not os.path.exists(args.out)) else 'a'
    with open(args.out, mode, newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if mode == 'w':
            writer.writeheader()
            f.flush()

        remaining = []
        for t in tasks:
            src_dir, inst_type, bay, row, tier, idx = t
            try:
                _, name = find_and_process_file(src_dir, inst_type, bay, row, tier, idx, no_print=True)
            except (FileNotFoundError, ValueError):
                continue
            if name not in done_names:
                remaining.append(t)
        print(f'Total tasks: {len(tasks)}, remaining: {len(remaining)}, workers={args.workers}')

        with multiprocessing.Pool(args.workers, initializer=_init_worker) as pool:
            n_done = 0
            for r in pool.imap_unordered(_process, remaining):
                if r is None:
                    continue
                if r['instance'] in done_names:
                    continue
                writer.writerow(r)
                f.flush()
                os.fsync(f.fileno())
                n_done += 1
                print(f'[{n_done}] {r["instance"]}: ZS same-bay={r["zs_same_bay_rate"]:.2f} '
                      f'Leveling same-bay={r["lv_same_bay_rate"]:.2f}')

    print(f'Done -> {args.out}')


if __name__ == '__main__':
    main()
