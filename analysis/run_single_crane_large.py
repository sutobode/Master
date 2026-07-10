"""Experiment 1b: single-crane scale generalization on Shin_instances.

Mirrors Shin et al. (2026)'s own Q4 scalability check: 20/30-bay layouts,
1440-2880 containers, comparing the Original Model (+ ZeroShot, which the
70-instance Experiment 1 already showed is numerically identical at C=1)
against Lin2015 (the strongest single-crane heuristic in Experiment 1).
Only Lin2015 is included as a heuristic comparator -- matching Shin et al.'s
own choice for this scale (their Table 5 also compares only GP and Lin) --
to keep runtime bounded: Kim2016/Durasevic2025 are already 3-8x worse than
Lin2015 at small scale and only degrade further with yard size.

This dataset (benchmarks/Shin_instances/) is NOT used by the default
70-instance Experiment 1, nor by the 70-layout multi-crane Experiments 2/3
-- it is wired in ONLY here, to answer "does the zero-shot pipeline still
work at the scale that motivates multi-crane operation in the first place?"

Usage: python -m analysis.run_single_crane_large [--max_per_scale N]
Output: results/single_crane_large.csv
"""

import sys, os, time, argparse
import pandas as pd
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.run_single_crane_v2 import load_original_model
from policy.zero_shot import ZeroShotPolicy
from mcenv.mcenv import MCEnv
from engine.mcrp_inference import run_mcrp_episode
from strategies import RoundRobin
from baselines.lin2015 import Lin2015
from baselines.lowerbound import get_wt_lb
from benchmarks.benchmarks import find_and_process_file

OUT = 'results/single_crane_large.csv'
BAYS = [20, 30]
TIERS = [6, 8]


def large_instances(max_per_scale=None):
    for tier in TIERS:
        for bay in BAYS:
            for inst_type in ('random', 'upsidedown'):
                idxs = range(1, (max_per_scale or 20) + 1)
                for idx in idxs:
                    yield inst_type, bay, 16, tier, idx


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--max_per_scale', type=int, default=3,
                    help='Instances per (bay,tier,type) group (max 20 available). '
                         'Default 3 keeps runtime bounded: each large instance takes '
                         '~20-100s+ per method (Shin et al. reported comparable times '
                         'for the DRL forward pass alone at this scale). Pass 20 for '
                         'the full benchmark if you have time (8 groups x 20 x 3 '
                         'methods = 480 runs).')
    args = p.parse_args()

    model = load_original_model()
    policy = ZeroShotPolicy(device=torch.device('cpu'))

    rows = []
    t0 = time.time()
    n_skipped = 0
    for inst_type, bay, row, tier, idx in large_instances(args.max_per_scale):
        try:
            x, name = find_and_process_file('benchmarks/Shin_instances', inst_type, bay, row, tier, idx, no_print=True)
        except (FileNotFoundError, ValueError):
            n_skipped += 1
            continue
        lb = float(get_wt_lb(x))

        bt0 = time.time()
        with torch.no_grad():
            wt, _ = model(x, None)
        cost = float(wt[0].item())
        rows.append({'instance': name, 'type': 'R' if inst_type == 'random' else 'U',
                     'n_bays': bay, 'n_tiers': tier, 'method': 'OriginalModel',
                     'cost': round(cost, 1), 'lb': round(lb, 1),
                     'gap': round(100 * (cost - lb) / lb, 2), 'time_s': round(time.time() - bt0, 2)})

        bt0 = time.time()
        env = MCEnv('cpu', x, n_cranes=1)
        zs_result = run_mcrp_episode(policy, env, RoundRobin(1, bay, row), bay, row, tier)
        cost_zs = zs_result['total_cost']
        rows.append({'instance': name, 'type': 'R' if inst_type == 'random' else 'U',
                     'n_bays': bay, 'n_tiers': tier, 'method': 'ZeroShot',
                     'cost': round(cost_zs, 1), 'lb': round(lb, 1),
                     'gap': round(100 * (cost_zs - lb) / lb, 2), 'time_s': round(time.time() - bt0, 2)})

        bt0 = time.time()
        torch.manual_seed(1234)
        try:
            cost_h, _ = Lin2015().run(x)
            rows.append({'instance': name, 'type': 'R' if inst_type == 'random' else 'U',
                         'n_bays': bay, 'n_tiers': tier, 'method': 'Lin2015',
                         'cost': round(float(cost_h), 1), 'lb': round(lb, 1),
                         'gap': round(100 * (float(cost_h) - lb) / lb, 2), 'time_s': round(time.time() - bt0, 2)})
        except Exception as e:
            print(f'  ERROR Lin2015 {name}: {e}')

        print(f'  {name} done ({time.time() - t0:.0f}s elapsed)')

    if n_skipped:
        print(f'Skipped {n_skipped} missing instances (reduce --max_per_scale or check benchmarks/Shin_instances/)')

    df = pd.DataFrame(rows)
    os.makedirs('results', exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f'\nTotal: {len(df)} runs in {time.time() - t0:.0f}s -> {OUT}')
    print('\n=== Gap by method (large-scale Shin instances) ===')
    print(df.groupby('method')['gap'].agg(['mean', 'std', 'min', 'max']).round(2).sort_values('mean').to_string())
    print('\n=== Gap by scale ===')
    print(df.groupby(['method', 'n_bays'])['gap'].mean().round(2).unstack().to_string())


if __name__ == '__main__':
    main()
