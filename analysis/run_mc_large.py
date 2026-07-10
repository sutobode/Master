"""Experiment 4: multi-crane scale generalization on large-scale layouts.

Experiments 2/3 only cover 70 layouts derived from Lee_instances (<=10 bays,
<=720 containers). But multi-crane operation is motivated by LARGE yards
(the paper's own Introduction cites Rotterdam/Singapore/Busan running
multiple cranes to meet throughput on big blocks) -- a small yard barely
needs a second crane. This experiment checks whether the S1-vs-S2 pattern
found at small/medium scale (spatial awareness reduces interference, some
makespan speedup from C=2 to C=3) holds, strengthens, or breaks down on the
20/30-bay, 1440-2880-container layouts that most resemble a real block.

Deliberately narrow scope to keep runtime bounded (each instance already
takes ~20-100s+ for the DRL forward pass alone at this scale, per Shin et
al.; multi-crane adds per-step environment overhead on top):
  - only S1 (RoundRobin, non-spatial baseline) and S2 (ZoneSplit, the best
    strategy from Experiment 2) -- not the full S1-S4
  - only C=2 -- not both C=2 and C=3
  - a small subset of instances via --n_instances (default 3, max ~10 per
    generate_large()'s 160 available layouts)

Requires benchmarks/mc_instances/lee_mc_large/ to exist first:
    python -c "from benchmarks.generate_mc_instances import generate_large; generate_large()"

Usage: python -m analysis.run_mc_large [--n_instances N]
Output: results/mc_large_v2.csv
"""

import sys, os, time, glob, argparse
import pandas as pd
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from policy.zero_shot import ZeroShotPolicy
from mcenv.mcenv import MCEnv
from engine.mcrp_inference import run_mcrp_episode
from bounds.lowerbound_mc import compute_lb_mc
from strategies import RoundRobin, ZoneSplit
from experiment import parse_instance_file, load_instance_tensor

OUT = 'results/mc_large_v2.csv'
LARGE_DIR = 'benchmarks/mc_instances/lee_mc_large'
STRATEGY_MAP = {'S1': RoundRobin, 'S2': ZoneSplit}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--n_instances', type=int, default=3,
                    help='How many large-scale layouts to test (of up to 160 available)')
    args = p.parse_args()

    if not os.path.isdir(LARGE_DIR) or not glob.glob(f'{LARGE_DIR}/*.txt'):
        raise SystemExit(
            f'{LARGE_DIR}/ is empty. Generate it first:\n'
            f'  python -c "from benchmarks.generate_mc_instances import generate_large; generate_large()"'
        )

    files = sorted(glob.glob(f'{LARGE_DIR}/*.txt'))[:args.n_instances]
    policy = ZeroShotPolicy(device=torch.device('cpu'))

    rows = []
    t0 = time.time()
    n_total = len(files) * len(STRATEGY_MAP)
    for fpath in files:
        fname = os.path.basename(fpath)
        data_lines, crane_starts = parse_instance_file(fpath)
        dims = fname.split('_')[1][1:]
        n_bays, n_rows, n_tiers = int(dims[0:2]), int(dims[2:4]), int(dims[4:6])
        x = load_instance_tensor(data_lines, n_bays, n_rows, n_tiers)

        n_cranes = 2
        lb = compute_lb_mc(x, n_bays, n_rows, n_tiers, n_cranes)
        lb_work, lb_makespan = lb['work'][0].item(), lb['makespan'][0].item()

        for sname, StrategyCls in STRATEGY_MAP.items():
            strategy = StrategyCls(n_cranes, n_bays, n_rows)
            env = MCEnv('cpu', x, n_cranes, crane_start_bays=crane_starts.get(n_cranes))
            bt0 = time.time()
            res = run_mcrp_episode(policy, env, strategy, n_bays, n_rows, n_tiers)
            bt = time.time() - bt0

            gap_work = 100 * (res['total_cost'] - lb_work) / lb_work
            gap_makespan = 100 * (res['makespan'] - lb_makespan) / lb_makespan
            rows.append({
                'instance': fname, 'n_bays': n_bays, 'n_cranes': n_cranes, 'strategy': sname,
                'total_cost': round(res['total_cost'], 1), 'makespan': round(res['makespan'], 1),
                'gap_work': round(max(gap_work, 0.0), 2), 'gap_makespan': round(max(gap_makespan, 0.0), 2),
                'interference_wait': round(res['interference_wait'], 1),
                'a7_reassignments': res['a7_reassignments'], 'time_s': round(bt, 1),
            })
            print(f'  [{len(rows)}/{n_total}] {fname} {sname}: gap_makespan={gap_makespan:.1f}% ({bt:.0f}s)')

    df = pd.DataFrame(rows)
    os.makedirs('results', exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f'\nTotal: {len(df)} runs in {time.time() - t0:.0f}s -> {OUT}')
    print(df.groupby(['n_bays', 'strategy'])[['gap_makespan', 'interference_wait']].mean().round(2).to_string())


if __name__ == '__main__':
    main()
