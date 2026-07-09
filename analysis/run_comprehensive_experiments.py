"""Comprehensive M-CRP Experiment: ALL methods on ALL datasets with FULL logging.

Run: PYTHONUNBUFFERED=1 python analysis/run_comprehensive_experiments.py

This will:
  Phase A: Run ALL methods on ALL single-crane datasets
  Phase B: Compare Original Model vs ZeroShot
  Phase C: Generate summary tables
"""

import sys, os, time, torch, json, glob, argparse
import pandas as pd
import numpy as np
from datetime import datetime
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.model import Model
from policy.zero_shot import ZeroShotPolicy
from mcenv.mcenv import MCEnv
from engine.mcrp_inference import run_mcrp_episode
from bounds.lowerbound_mc import compute_lb_mc
from baselines.lowerbound import get_wt_lb
from strategies import ZoneSplit
from baselines.lin2015 import Lin2015
from baselines.kim2016 import Kim2016
from baselines.leveling import Leveling
from baselines.durasevic2025 import Durasevic2025
from baselines.simple_baselines import NearestStack, LowestHeight
from benchmarks.benchmarks import find_and_process_file
from experiment import parse_instance_file, load_instance_tensor

# Config
OUT_DIR = 'results/comprehensive'
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs('logs', exist_ok=True)

DEVICE = torch.device('cpu')

# Load models once globally
MODEL_ARGS = argparse.Namespace(device=DEVICE, embed_dim=128, n_encode_layers=3, n_heads=8, ff_hidden=512, tanh_c=10, lstm=True, bay_embedding=True, online=False, online_known_num=None)

_orig_model = None
_zs_policy = None

def get_original_model():
    global _orig_model
    if _orig_model is None:
        _orig_model = Model(MODEL_ARGS)
        _orig_model.load_state_dict(torch.load('baselines/models/proposed/epoch(100).pt', map_location=DEVICE))
        _orig_model.eval()
        _orig_model.decoder.set_sampler('greedy')
    return _orig_model

def get_zeroshot():
    global _zs_policy
    if _zs_policy is None:
        _zs_policy = ZeroShotPolicy(device=DEVICE)
    return _zs_policy

SINGLE_METHODS_FAST = {
    'OriginalModel': get_original_model,
    'ZeroShot': get_zeroshot,
    'Lin2015': Lin2015, 'Kim2016': Kim2016, 'Leveling': Leveling,
    'Durasevic2025': Durasevic2025, 'NearestStack': NearestStack, 'LowestHeight': LowestHeight,
}

DATASETS = {
    'Lee_random': ('benchmarks/Lee_instances', 'random'),
    'Lee_upsidedown': ('benchmarks/Lee_instances', 'upsidedown'),
    'Shin_random': ('benchmarks/Shin_instances', 'random'),
    'Shin_upsidedown': ('benchmarks/Shin_instances', 'upsidedown'),
}

_log_file = [None]
def log(msg, also_print=True):
    ts = datetime.now().strftime('%H:%M:%S')
    full = f'[{ts}] {msg}'
    if also_print: print(full)
    if _log_file[0]:
        with open(_log_file[0], 'a', encoding='utf-8') as f:
            f.write(full + '\n')

def get_instance_configs(base_path, inst_type_str):
    """Get list of (bay, row, tier, id, fname) for all instances in a dataset."""
    configs = []
    files = []
    for root, dirs, fs in os.walk(base_path):
        for f in fs:
            if f.endswith('.txt'):
                files.append(os.path.join(root, f))
    
    for fpath in sorted(files):
        fname = os.path.basename(fpath)
        # Parse from filename like R011606_0070_001.txt or R201606_1440_001.txt
        try:
            parts = fname.replace('.txt', '').split('_')
            # Find the part with R or U
            dim_part = None
            id_part = None
            for p in parts:
                if p.startswith('R') or p.startswith('U'):
                    dim_part = p
                elif p.isdigit() and len(p) <= 5:
                    id_part = int(p)
            if dim_part is None:
                continue
            dims = dim_part[1:]  # Remove R or U prefix
            n_bays = int(dims[0:2])
            n_rows = int(dims[2:4]) if len(dims) >= 4 else 16
            n_tiers = int(dims[4:6]) if len(dims) >= 6 else 6
            configs.append((n_bays, n_rows, n_tiers, id_part or 1, fname))
        except (ValueError, IndexError):
            continue
    return configs

def load_instance(base_path, inst_type_str, bay, rows, tier, idx, no_print=True):
    """Load instance using the working benchmark parser."""
    try:
        x, _ = find_and_process_file(base_path, inst_type_str, bay, rows, tier, idx, no_print=no_print)
        return x
    except (FileNotFoundError, ValueError) as e:
        log(f'  SKIP: bay={bay} tier={tier} idx={idx} - {e}')
        return None

def run_single_method(method_name, method_obj, x_4d):
    """Run a single method on a 4D tensor and return (cost, gap, time_s)."""
    lb = float(get_wt_lb(x_4d))
    t0 = time.time()
    
    if method_name == 'OriginalModel':
        m = method_obj()
        with torch.no_grad():
            wt, _ = m(x_4d, None)
        cost = wt[0].item()
    elif method_name == 'ZeroShot':
        p = method_obj()
        env = MCEnv(DEVICE, x_4d, n_cranes=1, crane_start_bays=[1])
        result = run_mcrp_episode(p, env, ZoneSplit(1, x_4d.shape[1], x_4d.shape[2]),
                                  x_4d.shape[1], x_4d.shape[2], x_4d.shape[3])
        cost = result['total_cost']
    else:
        bl = method_obj()
        result = bl.run(x_4d)
        if isinstance(result, tuple):
            cost = float(result[0])
        elif isinstance(result, torch.Tensor):
            cost = result[0].item() if result.dim() > 0 else result.item()
        else:
            cost = float(result)
    
    elapsed = time.time() - t0
    gap = 100 * (cost - lb) / lb if lb > 0 else 0
    return round(cost, 1), round(gap, 2), round(elapsed, 3)

def phase_A():
    """Run ALL methods on ALL single-crane datasets."""
    all_results = []
    total_runs = 0
    t_start = time.time()
    
    for ds_name, (base_path, inst_type_str) in DATASETS.items():
        log(f'\n{"="*70}')
        log(f'DATASET: {ds_name} ({base_path}, {inst_type_str})')
        log(f'{"="*70}')
        
        configs = get_instance_configs(base_path, inst_type_str)
        log(f'Found {len(configs)} instance configurations')
        
        ds_results = []
        n_inst = len(configs)
        
        for inst_idx, (bay, rows, tier, idx, fname) in enumerate(configs):
            x = load_instance(base_path, inst_type_str, bay, rows, tier, idx)
            if x is None:
                continue
            if x.shape[0] != 1:
                x = x[:1]
            
            lb = float(get_wt_lb(x))
            
            for mname, mfactory in SINGLE_METHODS_FAST.items():
                try:
                    cost, gap, elapsed = run_single_method(mname, mfactory, x)
                    ds_results.append({
                        'dataset': ds_name, 'instance': fname,
                        'method': mname, 'cost': cost, 'lb': round(lb, 1),
                        'gap': gap, 'time_s': elapsed,
                    })
                    total_runs += 1
                except Exception as e:
                    log(f'  ERROR {mname} on {fname}: {e}')
            
            if (inst_idx + 1) % 5 == 0 or inst_idx == n_inst - 1:
                elapsed = time.time() - t_start
                rate = total_runs / elapsed if elapsed > 0 else 0
                log(f'  [{inst_idx+1}/{n_inst}] {fname} | {total_runs} runs | {elapsed:.0f}s elapsed | {rate:.1f} runs/s')
        
        # Save intermediate results per dataset
        df_ds = pd.DataFrame(ds_results)
        df_ds.to_csv(f'{OUT_DIR}/{ds_name}.csv', index=False)
        log(f'  Saved: {OUT_DIR}/{ds_name}.csv ({len(ds_results)} runs)')
        all_results.extend(ds_results)
    
    df_all = pd.DataFrame(all_results)
    df_all.to_csv(f'{OUT_DIR}/all_single_crane.csv', index=False)
    log(f'\nPhase A COMPLETE: {total_runs} runs in {time.time()-t_start:.0f}s')
    log(f'Saved: {OUT_DIR}/all_single_crane.csv')
    return df_all

def phase_B(df):
    """Compare Original Model vs ZeroShot."""
    log(f'\n{"="*70}')
    log(f'PHASE B: Original Model vs ZeroShot Comparison')
    log(f'{"="*70}')
    
    orig = df[df['method'] == 'OriginalModel'][['dataset', 'instance', 'cost', 'gap']].rename(
        columns={'cost': 'orig_cost', 'gap': 'orig_gap'})
    zs = df[df['method'] == 'ZeroShot'][['dataset', 'instance', 'cost', 'gap']].rename(
        columns={'cost': 'zs_cost', 'gap': 'zs_gap'})
    
    merged = orig.merge(zs, on=['dataset', 'instance'])
    merged['diff_pct'] = 100 * abs(merged['zs_cost'] - merged['orig_cost']) / merged['orig_cost']
    merged['gap_diff'] = merged['zs_gap'] - merged['orig_gap']
    
    log(f'Total paired comparisons: {len(merged)}')
    log(f'Average cost diff: {merged["diff_pct"].mean():.2f}%')
    log(f'Max cost diff: {merged["diff_pct"].max():.2f}%')
    log(f'Instances with diff > 2%: {(merged["diff_pct"] > 2).sum()}/{len(merged)}')
    
    merged.to_csv(f'{OUT_DIR}/original_vs_zeroshot.csv', index=False)
    log(f'Saved: {OUT_DIR}/original_vs_zeroshot.csv')
    return merged

def phase_C(df, orig_vs_zs):
    """Generate comprehensive summary."""
    log(f'\n{"="*70}')
    log(f'PHASE C: Comprehensive Summary')
    log(f'{"="*70}')
    
    # Table 1: Per-dataset summary
    log(f'\n--- GAP(%) BY DATASET AND METHOD ---')
    pivot = df.pivot_table(index='dataset', columns='method', values='gap', aggfunc=['mean', 'std'])
    log(pivot.to_string())
    
    # Table 2: Overall ranking
    log(f'\n--- OVERALL RANKING (ALL DATASETS) ---')
    ranking = df.groupby('method')['gap'].agg(['mean', 'std', 'min', 'max']).sort_values('mean')
    log(ranking.to_string())
    
    # Table 3: Original vs ZeroShot details
    log(f'\n--- ORIGINAL MODEL VS ZEROSHOT ---')
    for ds in orig_vs_zs['dataset'].unique():
        sub = orig_vs_zs[orig_vs_zs['dataset'] == ds]
        log(f'  {ds}: mean diff={sub["diff_pct"].mean():.2f}%, max diff={sub["diff_pct"].max():.2f}%')
    
    # Save summary
    summary = {
        'total_runs': len(df),
        'total_datasets': df['dataset'].nunique(),
        'total_instances': df['instance'].nunique(),
        'avg_gap_per_method': {m: round(g, 2) for m, g in df.groupby('method')['gap'].mean().items()},
        'orig_vs_zs_avg_diff': round(orig_vs_zs['diff_pct'].mean(), 2),
        'orig_vs_zs_max_diff': round(orig_vs_zs['diff_pct'].max(), 2),
    }
    with open(f'{OUT_DIR}/summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    log(f'\nSaved: {OUT_DIR}/summary.json')
    
    return summary

if __name__ == '__main__':
    _log_file[0] = f'logs/comprehensive_experiment_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'
    
    log(f'Starting comprehensive experiment...')
    log(f'Datasets: {list(DATASETS.keys())}')
    log(f'Methods: {list(SINGLE_METHODS_FAST.keys())}')
    log(f'Log file: {_log_file[0]}')
    
    # Phase A
    df = phase_A()
    
    # Phase B
    orig_vs_zs = phase_B(df)
    
    # Phase C
    summary = phase_C(df, orig_vs_zs)
    
    log(f'\n{"="*70}')
    log(f'ALL COMPLETE!')
    log(f'Total: {len(df)} runs across {df["dataset"].nunique()} datasets')
    log(f'Results: {OUT_DIR}/')
    log(f'Summary: {OUT_DIR}/summary.json')
    log(f'{"="*70}')
