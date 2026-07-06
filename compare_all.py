"""So sánh tất cả baselines + zero-shot DRL trên Lee benchmark (small subset)."""

import sys, os, time, torch
import pandas as pd
from baselines.lowerbound import get_wt_lb
from benchmarks.benchmarks import find_and_process_file

from baselines.lin2015 import Lin2015
from baselines.kim2016 import Kim2016
from baselines.leveling import Leveling
from baselines.durasevic2025 import Durasevic2025
from baselines.simple_baselines import RandomRelocate, NearestStack, LowestHeight

from policy.zero_shot import ZeroShotPolicy
from mcenv.mcenv import MCEnv
from engine.mcrp_inference import run_mcrp_episode
from strategies import RoundRobin

BASELINES = {
    'Lin2015': Lin2015,
    'Kim2016': Kim2016,
    'Leveling': Leveling,
    'Durasevic2025': Durasevic2025,
    'Random': RandomRelocate,
    'NearestStack': NearestStack,
    'LowestHeight': LowestHeight,
}

CONFIGS = [(1, 16, 6), (2, 16, 6)]


def run_baseline(baseline_cls, x_4d):
    bl = baseline_cls()
    t0 = time.time()
    result = bl.run(x_4d)
    elapsed = time.time() - t0
    if isinstance(result, tuple):
        cost = result[0]
    else:
        cost = result
    if isinstance(cost, torch.Tensor):
        cost = cost[0].item() if cost.dim() > 0 else cost.item()
    return float(cost), elapsed


def run_zero_shot(x_4d):
    policy = ZeroShotPolicy()
    env = MCEnv('cpu', x_4d, n_cranes=1)
    strategy = RoundRobin(1, x_4d.shape[1], x_4d.shape[2])
    t0 = time.time()
    result = run_mcrp_episode(policy, env, strategy, x_4d.shape[1], x_4d.shape[2], x_4d.shape[3])
    elapsed = time.time() - t0
    return result['total_cost'], elapsed


def main():
    torch.manual_seed(1234)
    os.makedirs('results', exist_ok=True)
    all_rows = []

    for bay, row, tier in CONFIGS:
        print(f'\n=== {bay}bay x {row}row x {tier}tier ===')
        for idx in range(1, 6):
            x, name = find_and_process_file(
                'benchmarks/Lee_instances', 'random', bay, row, tier, idx, no_print=True
            )
            lb = float(get_wt_lb(x))

            row_data = {'instance': name.replace('.txt', ''), 'LB': round(lb, 1)}

            for bname, bcls in BASELINES.items():
                try:
                    cost, t = run_baseline(bcls, x)
                    gap = 100 * (cost - lb) / lb
                    row_data[f'{bname}_gap'] = round(gap, 2)
                    row_data[f'{bname}_time'] = round(t, 3)
                except Exception as e:
                    row_data[f'{bname}_gap'] = None
                    row_data[f'{bname}_time'] = None
                    print(f'  {name}: {bname} ERROR: {e}')

            try:
                cost, t = run_zero_shot(x)
                gap = 100 * (cost - lb) / lb
                row_data[f'ZeroShot_gap'] = round(gap, 2)
                row_data[f'ZeroShot_time'] = round(t, 3)
            except Exception as e:
                row_data[f'ZeroShot_gap'] = None
                row_data[f'ZeroShot_time'] = None
                print(f'  {name}: ZeroShot ERROR: {e}')

            all_rows.append(row_data)

            # Print progress
            gaps = []
            for bname in list(BASELINES.keys()) + ['ZeroShot']:
                g = row_data.get(f'{bname}_gap')
                if g is not None:
                    gaps.append(f'{bname}={g}')
            print(f'  {name}: {", ".join(gaps)}')

    df = pd.DataFrame(all_rows)
    print('\n\n========== AVERAGE GAP ACROSS ALL INSTANCES ==========')
    for col in sorted(df.columns):
        if col.endswith('_gap') and df[col].notna().any():
            vals = df[col].dropna()
            print(f'  {col:25s}: mean={vals.mean():.2f}%  min={vals.min():.2f}%  max={vals.max():.2f}%')

    path = 'results/baseline_comparison.csv'
    df.to_csv(path, index=False)
    print(f'\nSaved: {path}')
    print('\n--- Best method per instance ---')
    for _, row in df.iterrows():
        best, best_g = None, float('inf')
        for bname in list(BASELINES.keys()) + ['ZeroShot']:
            g = row.get(f'{bname}_gap')
            if g is not None and g < best_g:
                best_g, best = g, bname
        print(f'  {row["instance"]:30s}: {best:20s} (gap={best_g:.2f}%)')


if __name__ == '__main__':
    main()
