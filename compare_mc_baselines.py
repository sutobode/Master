"""Compare zero-shot DRL strategies against naive baseline on M-CRP instances.

Naive baseline: LB_single / C (idealized linear speedup).
Zero-shot strategies: S1 RoundRobin, S2 ZoneSplit, S3 LoadBalance, S4 GreedyOptimal.
"""

import sys, os, glob, time, torch
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from policy.zero_shot import ZeroShotPolicy
from mcenv.mcenv import MCEnv
from engine.mcrp_inference import run_mcrp_episode
from bounds.lowerbound_mc import compute_lb_mc
from baselines.lowerbound import get_wt_lb
from strategies import RoundRobin, ZoneSplit, LoadBalance, GreedyOptimal

DRL_STRATEGIES = {
    'ZS+S1': RoundRobin,
    'ZS+S2': ZoneSplit,
    'ZS+S3': LoadBalance,
    'ZS+S4': GreedyOptimal,
}

LOG_FILE = 'results/mc_baseline_comparison.csv'


def log(msg):
    print(msg)
    with open('results/mc_baseline_log.txt', 'a', encoding='utf-8') as f:
        f.write(msg + '\n')


def main():
    torch.manual_seed(1234)
    os.makedirs('results', exist_ok=True)
    if os.path.exists('results/mc_baseline_log.txt'):
        os.remove('results/mc_baseline_log.txt')

    policy = ZeroShotPolicy()

    files = sorted(glob.glob('benchmarks/mc_instances/lee_mc/*_c2.txt'))[:10]
    log(f'Testing {len(files)} instances with 2 cranes')

    all_rows = []

    for fpath in files:
        from experiment import parse_instance_file, load_instance_tensor
        fname = os.path.basename(fpath)
        data_lines, file_cranes, crane_starts = parse_instance_file(fpath)
        parts = fname.replace('.txt', '').split('_')
        dims = parts[1][1:]
        n_bays = int(dims[0:2])
        n_rows = int(dims[2:4])
        n_tiers = int(dims[4:6])

        x = load_instance_tensor(data_lines, n_bays, n_rows, n_tiers)
        lb_mc = compute_lb_mc(x, n_bays, n_rows, n_tiers, 2).item()
        lb_single = get_wt_lb(x.reshape(1, -1, n_tiers))
        naive_baseline = lb_single / 2

        log(f'\n--- {fname} (B={n_bays}, LB_mc={lb_mc:.1f}, LB_single/2={naive_baseline:.1f}) ---')

        for sname, StratCls in DRL_STRATEGIES.items():
            try:
                strategy = StratCls(2, n_bays, n_rows)
                env = MCEnv('cpu', x, n_cranes=2, crane_start_bays=crane_starts)
                t0 = time.time()
                result = run_mcrp_episode(policy, env, strategy, n_bays, n_rows, n_tiers)
                elapsed = time.time() - t0
                gap_lb = 100 * (result['total_cost'] - lb_mc) / lb_mc if lb_mc > 0 else 0.0
                gap_naive = 100 * (result['total_cost'] - naive_baseline) / naive_baseline if naive_baseline > 0 else 0.0
                speedup = lb_single / result['total_cost'] if result['total_cost'] > 0 else 0.0

                row = {
                    'instance': fname, 'n_cranes': 2, 'method': sname,
                    'cost': round(result['total_cost'], 1),
                    'lb_mc': round(lb_mc, 1),
                    'lb_single': round(lb_single, 1),
                    'naive_C2': round(naive_baseline, 1),
                    'gap_vs_LB': round(gap_lb, 2),
                    'gap_vs_naive': round(gap_naive, 2),
                    'speedup_vs_C1': round(speedup, 3),
                    'time_s': round(elapsed, 3),
                    'interference': result['n_interference'],
                    'n_steps': result['n_steps'],
                }
                all_rows.append(row)
                log(f'  {sname:15s}: cost={result["total_cost"]:.1f} '
                    f'gap_LB={gap_lb:.2f}% gap_naive={gap_naive:.2f}% '
                    f'speedup={speedup:.2f}x intf={result["n_interference"]:.0f}')
            except Exception as e:
                log(f'  {sname:15s}: ERROR {e}')

    df = pd.DataFrame(all_rows)
    df.to_csv(LOG_FILE, index=False)

    log('\n\n=== SUMMARY ===')
    for metric in ['gap_vs_LB', 'gap_vs_naive', 'speedup_vs_C1', 'interference']:
        log(f'\n--- {metric} ---')
        summary = df.groupby('method')[metric].agg(['mean', 'std', 'min', 'max'])
        for method in summary.index:
            s = summary.loc[method]
            log(f'  {method:15s}: mean={s["mean"]:8.3f} std={s["std"]:6.3f} '
                f'min={s["min"]:8.3f} max={s["max"]:6.3f}')

    log(f'\nResults saved: {LOG_FILE}')
    best = df.loc[df.groupby('instance')['gap_vs_LB'].idxmin()]
    log('\n=== Best per instance (by gap_vs_LB) ===')
    for _, row in best.iterrows():
        log(f'  {row["instance"]:35s}: {row["method"]:15s} gap={row["gap_vs_LB"]:.2f}%')


if __name__ == '__main__':
    main()
