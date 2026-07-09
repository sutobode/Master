"""Fix critical issues: full Lee benchmark, multi-bay case study, backward compat verification.

Run: python analysis/fix_critical_issues.py
"""

import sys, os, time, torch, json, glob, argparse
import pandas as pd
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.model import Model
from policy.zero_shot import ZeroShotPolicy
from mcenv.mcenv import MCEnv
from engine.mcrp_inference import run_mcrp_episode
from bounds.lowerbound_mc import compute_lb_mc
from baselines.lowerbound import get_wt_lb
from strategies import ZoneSplit, RoundRobin
from benchmarks.benchmarks import find_and_process_file
from experiment import parse_instance_file, load_instance_tensor

OUTPUT_DIR = 'results/critical_fixes'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def log(msg):
    print(msg)
    with open(f'{OUTPUT_DIR}/log.txt', 'a', encoding='utf-8') as f:
        f.write(msg + '\n')

DEVICE = torch.device('cpu')

# ===========================================================
# FIX 1: Full Single-Crane Lee Benchmark (36 instances)
# ===========================================================
def run_full_single_crane_benchmark():
    log('\n' + '='*70)
    log('FIX 1: Full Single-Crane Lee Benchmark (36 instances)')
    log('='*70)

    policy = ZeroShotPolicy(device=DEVICE)

    results = []
    configs = []
    for bay in [1, 2, 4, 6, 8, 10]:
        for tier in [6, 8]:
            if tier == 8 and bay in [8, 10]:
                continue
            configs.append((bay, 16, tier))

    for bay, rows, tier in configs:
        for idx in range(1, 6):
            try:
                x, name = find_and_process_file(
                    'benchmarks/Lee_instances', 'random', bay, rows, tier, idx, no_print=True
                )
            except (FileNotFoundError, ValueError):
                continue

            lb = float(get_wt_lb(x))
            lb_single = lb

            # ZeroShot (C=1)
            env = MCEnv(DEVICE, x, n_cranes=1, crane_start_bays=[1])
            zs_result = run_mcrp_episode(policy, env, RoundRobin(1, bay, rows), bay, rows, tier)
            zs_gap = 100 * (zs_result['total_cost'] - lb) / lb if lb > 0 else 0

            results.append({
                'instance': name.replace('.txt', ''),
                'bays': bay, 'tiers': tier, 'id': idx,
                'lb': round(lb, 1),
                'zero_shot_cost': round(zs_result['total_cost'], 1),
                'zero_shot_gap': round(zs_gap, 2),
                'n_steps': zs_result['n_steps'],
            })

    df = pd.DataFrame(results)
    df.to_csv(f'{OUTPUT_DIR}/full_lee_benchmark.csv', index=False)

    log(f'\nTotal instances: {len(df)}')
    avg_gap = df['zero_shot_gap'].mean()
    log(f'Average ZeroShot gap: {avg_gap:.2f}%')
    log(f'Min gap: {df["zero_shot_gap"].min():.2f}%')
    log(f'Max gap: {df["zero_shot_gap"].max():.2f}%')

    log(f'\nBy number of bays:')
    for bay in sorted(df['bays'].unique()):
        sub = df[df['bays'] == bay]
        log(f'  B={bay}: mean gap={sub["zero_shot_gap"].mean():.2f}% (n={len(sub)})')

    log(f'\nSaved: {OUTPUT_DIR}/full_lee_benchmark.csv')
    return df

# ===========================================================
# FIX 2: Multi-bay Case Study (6-bay instance)
# ===========================================================
def run_multi_bay_case_study():
    log('\n' + '='*70)
    log('FIX 2: Multi-bay Case Study (detecting S1 vs S2 difference)')
    log('='*70)

    policy = ZeroShotPolicy(device=DEVICE)

    # Test multi-bay instances to find where S1 and S2 differ
    test_files = sorted(glob.glob('benchmarks/mc_instances/lee_mc/*R06*_c2.txt'))[:3]
    test_files += sorted(glob.glob('benchmarks/mc_instances/lee_mc/*R04*_c2.txt'))[:3]

    best_diff = 0
    best_case = None

    for fpath in test_files:
        fname = os.path.basename(fpath)
        data_lines, file_cranes, crane_starts = parse_instance_file(fpath)
        parts = fname.replace('.txt', '').split('_')
        dims = parts[1][1:]
        n_bays = int(dims[0:2])
        n_rows = int(dims[2:4])
        n_tiers = int(dims[4:6])
        x = load_instance_tensor(data_lines, n_bays, n_rows, n_tiers)

        lb = compute_lb_mc(x, n_bays, n_rows, n_tiers, 2).item()
        n_containers = int((x > 0).sum().item())

        # Run S1
        env = MCEnv(DEVICE, x, n_cranes=2, crane_start_bays=crane_starts)
        r1 = run_mcrp_episode(policy, env, RoundRobin(2, n_bays, n_rows), n_bays, n_rows, n_tiers)

        # Run S2
        env = MCEnv(DEVICE, x, n_cranes=2, crane_start_bays=crane_starts)
        r2 = run_mcrp_episode(policy, env, ZoneSplit(2, n_bays, n_rows), n_bays, n_rows, n_tiers)

        diff = abs(r1['total_cost'] - r2['total_cost'])
        g1 = 100*(r1['total_cost']-lb)/lb
        g2 = 100*(r2['total_cost']-lb)/lb

        log(f'\n{fname} (B={n_bays}, N={n_containers}):'
            f'\n  S1: cost={r1["total_cost"]:.0f} gap={g1:.1f}% intf={r1["n_interference"]}'
            f'\n  S2: cost={r2["total_cost"]:.0f} gap={g2:.1f}% intf={r2["n_interference"]}'
            f'\n  Cost diff: {diff:.0f} ({100*diff/r2["total_cost"]:.1f}%)')

        if diff > best_diff and n_bays >= 4:
            best_diff = diff
            best_case = {
                'fname': fname, 'n_bays': n_bays, 'n_rows': n_rows, 'n_tiers': n_tiers,
                'n_containers': n_containers, 'lb': round(lb, 1),
                's1_cost': round(r1['total_cost'], 1), 's1_gap': round(g1, 2), 's1_intf': r1['n_interference'],
                's2_cost': round(r2['total_cost'], 1), 's2_gap': round(g2, 2), 's2_intf': r2['n_interference'],
                'diff_cost': round(diff, 1), 'diff_pct': round(100*diff/r2['total_cost'], 1),
            }

    if best_case:
        log(f'\n=== BEST CASE STUDY INSTANCE ===')
        log(json.dumps(best_case, indent=2))
        with open(f'{OUTPUT_DIR}/case_study.json', 'w') as f:
            json.dump(best_case, f, indent=2)
        log(f'\nSaved: {OUTPUT_DIR}/case_study.json')
    else:
        log('\nWARNING: No instance found where S1 and S2 differ.')

# ===========================================================
# FIX 3: Backward Compatibility Verification (5+ instances)
# ===========================================================
def verify_backward_compatibility():
    log('\n' + '='*70)
    log('FIX 3: Backward Compatibility Verification (5 instances)')
    log('='*70)

    policy = ZeroShotPolicy(device=DEVICE)

    MODEL_ARGS = argparse.Namespace(
        device=DEVICE, embed_dim=128, n_encode_layers=3, n_heads=8,
        ff_hidden=512, tanh_c=10, lstm=True, bay_embedding=True,
        online=False, online_known_num=None
    )

    orig_model = Model(MODEL_ARGS)
    orig_model.load_state_dict(torch.load('baselines/models/proposed/epoch(100).pt', map_location=DEVICE))
    orig_model.eval()
    orig_model.decoder.set_sampler('greedy')

    results = []
    test_configs = [(1,16,6,1), (1,16,6,3), (2,16,6,1), (4,16,6,1), (6,16,6,1)]

    for bay, rows, tier, idx in test_configs:
        try:
            x, name = find_and_process_file(
                'benchmarks/Lee_instances', 'random', bay, rows, tier, idx, no_print=True
            )
        except (FileNotFoundError, ValueError):
            continue

        lb = float(get_wt_lb(x))

        # Original model
        with torch.no_grad():
            wt_orig, _ = orig_model(x, None)
        orig_cost = wt_orig[0].item()

        # ZeroShot (C=1)
        env = MCEnv(DEVICE, x, n_cranes=1, crane_start_bays=[1])
        zs_result = run_mcrp_episode(policy, env, RoundRobin(1, bay, rows), bay, rows, tier)

        diff_pct = 100 * abs(zs_result['total_cost'] - orig_cost) / orig_cost
        zs_gap = 100 * (zs_result['total_cost'] - lb) / lb if lb > 0 else 0
        orig_gap = 100 * (orig_cost - lb) / lb if lb > 0 else 0

        results.append({
            'instance': name.replace('.txt', ''),
            'bays': bay, 'tiers': tier,
            'lb': round(lb, 1),
            'original_cost': round(orig_cost, 1), 'original_gap': round(orig_gap, 2),
            'zeroshot_cost': round(zs_result['total_cost'], 1), 'zeroshot_gap': round(zs_gap, 2),
            'diff_pct': round(diff_pct, 2),
        })

        status = 'PASS' if diff_pct < 2.0 else 'FAIL'
        log(f'  {name}: orig={orig_cost:.0f} zs={zs_result["total_cost"]:.0f} diff={diff_pct:.2f}% [{status}]')

    df = pd.DataFrame(results)
    df.to_csv(f'{OUTPUT_DIR}/backward_compat.csv', index=False)

    avg_diff = df['diff_pct'].mean()
    max_diff = df['diff_pct'].max()
    n_pass = (df['diff_pct'] < 2.0).sum()
    n_total = len(df)

    log(f'\nSummary: {n_pass}/{n_total} passed (threshold 2%)')
    log(f'Average diff: {avg_diff:.2f}%, Max diff: {max_diff:.2f}%')
    log(f'Saved: {OUTPUT_DIR}/backward_compat.csv')


# ===========================================================
# MAIN
# ===========================================================
if __name__ == '__main__':
    sys.exit(
        'DEPRECATED: uses the pre-revision parse_instance_file()/compute_lb_mc() '
        'API and old _c2/_c3-suffixed instance filenames, neither of which exist '
        'anymore. Use `analysis.run_single_crane_v2` and '
        '`analysis.run_mc_baselines_v2` instead — see README.md.'
    )
    import argparse

    log('Starting critical fixes...\n')

    # FIX 1: Full single-crane benchmark
    df_lee = run_full_single_crane_benchmark()

    # FIX 2: Multi-bay case study
    run_multi_bay_case_study()

    # FIX 3: Backward compatibility
    verify_backward_compatibility()

    log('\n' + '='*70)
    log('ALL CRITICAL FIXES COMPLETE')
    log(f'Results saved to {OUTPUT_DIR}/')
    log('='*70)
