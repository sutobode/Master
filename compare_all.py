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
LOG_FILE = 'results/compare_log.txt'


def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(msg + '\n')
    print(msg)


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
    if os.path.exists(LOG_FILE):
        os.remove(LOG_FILE)
    all_rows = []

    log('=' * 100)
    log('COMPARE ALL BASELINES + ZEROSHOT')
    log(f'Configs: {CONFIGS}')
    log('=' * 100)

    for bay, row, tier in CONFIGS:
        for idx in range(1, 3):  # Only 2 instances per config for speed
            x, name = find_and_process_file(
                'benchmarks/Lee_instances', 'random', bay, row, tier, idx, no_print=True
            )
            lb = float(get_wt_lb(x))
            inst_name = name.replace('.txt', '')

            log(f'\n--- {inst_name} (LB={lb:.1f}) ---')
            row_data = {'instance': inst_name, 'LB': round(lb, 1)}

            # Fast baselines
            for bname, bcls in BASELINES.items():
                try:
                    cost, t = run_baseline(bcls, x)
                    gap = 100 * (cost - lb) / lb
                    row_data[f'{bname}_gap'] = round(gap, 2)
                    row_data[f'{bname}_time'] = round(t, 3)
                    log(f'  {bname:20s}: cost={cost:.1f} gap={gap:.2f}% time={t:.2f}s')
                except Exception as e:
                    row_data[f'{bname}_gap'] = None
                    row_data[f'{bname}_time'] = None
                    log(f'  {bname:20s}: ERROR {e}')

            # ZeroShot
            try:
                cost, t = run_zero_shot(x)
                gap = 100 * (cost - lb) / lb
                row_data['ZeroShot_gap'] = round(gap, 2)
                row_data['ZeroShot_time'] = round(t, 3)
                log(f'  ZS (ZeroShot)     : cost={cost:.1f} gap={gap:.2f}% time={t:.2f}s')
            except Exception as e:
                row_data['ZeroShot_gap'] = None
                row_data['ZeroShot_time'] = None
                log(f'  ZS (ZeroShot)     : ERROR {e}')

            all_rows.append(row_data)

    df = pd.DataFrame(all_rows)
    log('\n\n' + '=' * 80)
    log('AVERAGE GAP ACROSS ALL INSTANCES')
    log('=' * 80)
    log('  Method                      | Mean Gap  | Min      | Max      | Avg Time')
    log('-' * 65)
    for col in sorted(df.columns):
        if col.endswith('_gap') and df[col].notna().any():
            vals = df[col].dropna()
            time_col = col.replace('_gap', '_time')
            avg_t = df[time_col].dropna().mean() if time_col in df else 0
            name = col.replace('_gap', '')
            log(f'  {name:>25s} | {vals.mean():>7.2f}% | {vals.min():>7.2f}% | {vals.max():>7.2f}% | {avg_t:>7.3f}s')
    log('-' * 65)

    path = 'results/baseline_comparison.csv'
    df.to_csv(path, index=False)
    log(f'\nResults saved: {path}')
    log('\n--- Best method per instance ---')
    for _, row in df.iterrows():
        best, best_g = None, float('inf')
        for bname in list(BASELINES.keys()) + ['ZeroShot']:
            g = row.get(f'{bname}_gap')
            if g is not None and g < best_g:
                best_g, best = g, bname
        inst_name = row['instance']
        log(f'  {inst_name:30s}: {best:20s} (gap={best_g:.2f}%)')


if __name__ == '__main__':
    main()
