"""Run ALL remaining experiments efficiently without re-running completed ones.

Strategy:
  - Single-crane: run 7 baselines on 50 Lee instances (missing, ~2 min)
  - Multi-crane: run heuristic baselines on 140 M-CRP instances (missing, ~10 min)
  - Combine all results into comprehensive report
"""

import sys, os, time, torch, glob, json
import pandas as pd
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from policy.zero_shot import ZeroShotPolicy
from mcenv.mcenv import MCEnv
from engine.mcrp_inference import run_mcrp_episode
from bounds.lowerbound_mc import compute_lb_mc
from baselines.lowerbound import get_wt_lb
from strategies import ZoneSplit, RoundRobin
from baselines.lin2015 import Lin2015
from baselines.kim2016 import Kim2016
from baselines.leveling import Leveling
from baselines.durasevic2025 import Durasevic2025
from baselines.simple_baselines import RandomRelocate, NearestStack, LowestHeight
from benchmarks.benchmarks import find_and_process_file
from experiment import parse_instance_file, load_instance_tensor

OUT = 'results/final_comprehensive'
os.makedirs(OUT, exist_ok=True)

def log(msg):
    print(msg)
    with open(f'{OUT}/log.txt', 'a', encoding='utf-8') as f:
        f.write(msg + '\n')

SINGLE_BASELINES = {
    'Random': RandomRelocate, 'NearestStack': NearestStack,
    'LowestHeight': LowestHeight, 'Leveling': Leveling,
    'Lin2015': Lin2015, 'Kim2016': Kim2016, 'Durasevic2025': Durasevic2025,
}

# ============================================================
# PART 1: Single-crane baselines on 50 Lee instances
# ============================================================
def run_single_crane_baselines():
    out_csv = f'{OUT}/single_crane_all_baselines.csv'
    if os.path.exists(out_csv):
        log(f'SKIP: {out_csv} already exists')
        return pd.read_csv(out_csv)

    log('\n=== PART 1: Running 7 baselines on 50 Lee instances ===')
    
    # Load existing ZeroShot results
    zs_df = pd.read_csv('results/critical_fixes/full_lee_benchmark.csv')
    
    all_rows = []
    configs = []
    for bay in [1, 2, 4, 6, 8, 10]:
        for tier in [6, 8]:
            if tier == 8 and bay in [8, 10]:
                continue
            configs.append((bay, 16, tier))

    n_total = len(configs) * 5  # 5 instances per config
    n_done = 0
    t0 = time.time()

    for bay, rows, tier in configs:
        for idx in range(1, 6):
            try:
                x, name = find_and_process_file(
                    'benchmarks/Lee_instances', 'random', bay, rows, tier, idx, no_print=True
                )
            except (FileNotFoundError, ValueError):
                continue

            inst_name = name.replace('.txt', '')
            lb = float(get_wt_lb(x))

            # Get ZeroShot result from existing data
            zs_row = zs_df[(zs_df['instance'] == inst_name)]
            zs_gap = zs_row['zero_shot_gap'].values[0] if len(zs_row) > 0 else None

            row_data = {'instance': inst_name, 'bays': bay, 'tiers': tier, 'id': idx, 'lb': round(lb, 1)}

            for bname, BCls in SINGLE_BASELINES.items():
                try:
                    bl = BCls()
                    bl_t0 = time.time()
                    result = bl.run(x)
                    bl_elapsed = time.time() - bl_t0
                    if isinstance(result, tuple):
                        cost = result[0]
                    else:
                        cost = result
                    if isinstance(cost, torch.Tensor):
                        cost = cost[0].item() if cost.dim() > 0 else cost.item()
                    cost = float(cost)
                    gap = 100 * (cost - lb) / lb if lb > 0 else 0
                    row_data[f'{bname}_cost'] = round(cost, 1)
                    row_data[f'{bname}_gap'] = round(gap, 2)
                    row_data[f'{bname}_time'] = round(bl_elapsed, 3)
                except Exception as e:
                    row_data[f'{bname}_gap'] = None

            if zs_gap is not None:
                zs_row_data = zs_df[zs_df['instance'] == inst_name].iloc[0]
                row_data['ZeroShot_cost'] = zs_row_data['zero_shot_cost']
                row_data['ZeroShot_gap'] = zs_row_data['zero_shot_gap']

            all_rows.append(row_data)
            n_done += 1
            if n_done % 10 == 0:
                elapsed = time.time() - t0
                rate = n_done / elapsed
                remaining = (n_total - n_done) / rate if rate > 0 else 0
                log(f'  [{n_done}/{n_total}] {inst_name} | {remaining:.0f}s remaining')

    df = pd.DataFrame(all_rows)
    df.to_csv(out_csv, index=False)
    log(f'\nDone: {len(df)} instances in {time.time()-t0:.0f}s')
    return df

# ============================================================
# PART 2: Multi-crane heuristic baselines on ALL 140 instances
# ============================================================
def run_multi_crane_baselines():
    out_csv = f'{OUT}/multi_crane_all_baselines.csv'
    if os.path.exists(out_csv):
        log(f'SKIP: {out_csv} already exists')
        return pd.read_csv(out_csv)

    log('\n=== PART 2: Running multi-crane baselines on 140 M-CRP instances ===')

    policy = ZeroShotPolicy()
    results = []
    
    files = sorted(glob.glob('benchmarks/mc_instances/lee_mc/*.txt'))
    n_total = len(files)
    n_done = 0
    t0 = time.time()

    for fpath in files:
        fname = os.path.basename(fpath)
        try:
            data_lines, file_cranes, crane_starts = parse_instance_file(fpath)
        except Exception:
            continue

        parts = fname.replace('.txt', '').split('_')
        n_bays = int(parts[1][1:3])
        n_rows = int(parts[1][3:5])
        n_tiers = int(parts[1][5:7])
        n_cranes_cfg = 2 if '_c2.' in fname else 3

        x = load_instance_tensor(data_lines, n_bays, n_rows, n_tiers)
        lb_mc = compute_lb_mc(x, n_bays, n_rows, n_tiers, n_cranes_cfg).item()

        # ZS+S2 (already done for C=2, need for C=3 and all files)
        for strat_name, StratCls in [('ZS+S2', ZoneSplit)]:
            strategy = StratCls(n_cranes_cfg, n_bays, n_rows)
            env = MCEnv('cpu', x, n_cranes=n_cranes_cfg, crane_start_bays=crane_starts)
            result = run_mcrp_episode(policy, env, strategy, n_bays, n_rows, n_tiers)
            gap = 100 * (result['total_cost'] - lb_mc) / lb_mc if lb_mc > 0 else 0
            results.append({
                'instance': fname, 'n_cranes': n_cranes_cfg, 'method': strat_name,
                'cost': round(result['total_cost'], 1), 'lb_mc': round(lb_mc, 1),
                'gap': round(gap, 2), 'interference': result['n_interference'],
                'n_steps': result['n_steps'], 'time_s': round(result.get('time_s', 0), 3),
            })

        # M-Lin2015 (heuristic baseline)
        try:
            bl = Lin2015()
            bl_t0 = time.time()
            bl_result = bl.run(x)
            bl_time = time.time() - bl_t0
            if isinstance(bl_result, tuple):
                bl_cost = bl_result[0]
            else:
                bl_cost = bl_result
            if isinstance(bl_cost, torch.Tensor):
                bl_cost = bl_cost[0].item() if bl_cost.dim() > 0 else bl_cost.item()
            bl_cost = float(bl_cost)
            bl_gap = 100 * (bl_cost - lb_mc) / lb_mc if lb_mc > 0 else 0
            results.append({
                'instance': fname, 'n_cranes': n_cranes_cfg, 'method': 'M-Lin2015',
                'cost': round(bl_cost, 1), 'lb_mc': round(lb_mc, 1),
                'gap': round(bl_gap, 2), 'interference': 0,
                'n_steps': 0, 'time_s': round(bl_time, 3),
            })
        except Exception as e:
            log(f'  ERROR M-Lin2015 {fname}: {e}')

        n_done += 1
        if n_done % 10 == 0:
            elapsed = time.time() - t0
            rate = n_done / elapsed
            remaining = (n_total - n_done) / rate if rate > 0 else 0
            log(f'  [{n_done}/{n_total}] {fname} | {remaining:.0f}s remaining')

    df = pd.DataFrame(results)
    df.to_csv(out_csv, index=False)
    log(f'\nDone: {len(df)} runs in {time.time()-t0:.0f}s')
    return df

# ============================================================
# PART 3: Generate comprehensive summary
# ============================================================
def generate_summary(single_df, multi_df):
    log('\n=== PART 3: Comprehensive Summary ===')

    # Single-crane summary
    log('\n--- SINGLE-CRANE (50 Lee instances) ---')
    gap_cols = [c for c in single_df.columns if c.endswith('_gap')]
    summary = {}
    for col in sorted(gap_cols):
        vals = single_df[col].dropna()
        if len(vals) > 0:
            name = col.replace('_gap', '')
            summary[name] = {
                'mean_gap': round(vals.mean(), 2),
                'std_gap': round(vals.std(), 2),
                'min_gap': round(vals.min(), 2),
                'max_gap': round(vals.max(), 2),
                'n': len(vals),
            }
            log(f'  {name:20s}: mean={vals.mean():7.2f}% std={vals.std():5.2f} min={vals.min():7.2f}% max={vals.max():7.2f}% (n={len(vals)})')

    # Multi-crane summary
    log('\n--- MULTI-CRANE (140 instances × 2-3 cranes) ---')
    for nc in [2, 3]:
        sub = multi_df[multi_df['n_cranes'] == nc]
        log(f'\n  C={nc}:')
        for method in sorted(sub['method'].unique()):
            m_sub = sub[sub['method'] == method]
            log(f'    {method:12s}: gap={m_sub["gap"].mean():6.2f}±{m_sub["gap"].std():.2f}% intf={m_sub["interference"].mean():.1f} (n={len(m_sub)})')

    # Save JSON summary
    summary_data = {
        'single_crane': {k: v for k, v in summary.items()},
        'multi_crane': {
            f'C={nc}': {
                m: {
                    'gap_mean': round(sub[sub['method'] == m]['gap'].mean(), 2),
                    'gap_std': round(sub[sub['method'] == m]['gap'].std(), 2),
                    'intf_mean': round(sub[sub['method'] == m]['interference'].mean(), 1),
                }
                for m in sub['method'].unique()
            }
            for nc, sub in multi_df.groupby('n_cranes')
        }
    }
    with open(f'{OUT}/summary.json', 'w') as f:
        json.dump(summary_data, f, indent=2)
    log(f'\nSaved: {OUT}/summary.json')

if __name__ == '__main__':
    t_start = time.time()
    
    single_df = run_single_crane_baselines()
    multi_df = run_multi_crane_baselines()
    generate_summary(single_df, multi_df)

    log(f'\n{"="*60}')
    log(f'TOTAL TIME: {time.time()-t_start:.0f}s')
    log(f'All results saved to {OUT}/')
