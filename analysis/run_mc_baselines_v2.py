"""Experiment 3 (v2): multi-crane heuristic baselines vs ZeroShot, full matrix.

Runs M-Lin2015 / M-Kim2016 / M-Leveling per-step destination rules through
the identical MCEnv + timing-model protocol used by ZeroShot, on all 70
unique M-CRP layouts at C in {2, 3} -- crossed with ALL FOUR crane-assignment
strategies (S1-S4), not just ZoneSplit. This answers two questions at once:
(1) does DRL destination-choice beat heuristic destination-choice under an
identical assignment strategy (the headline ZeroShot+S2 vs M-Lin2015+S2
comparison), and (2) does the S1-S4 ranking found in Experiment 2 (which
only used ZeroShot) hold for heuristic destination-choice too, or is it an
artifact of the DRL policy specifically. Join with
results/mcrp_experiment_v2_main.csv (method='ZeroShot' implicitly) on
(n_cranes, strategy) for the combined method x strategy table.

Usage: python -m analysis.run_mc_baselines_v2
Output: results/mc_baselines_v2.csv
"""

import sys, os, time, glob
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcenv.mcenv import MCEnv
from bounds.lowerbound_mc import compute_lb_mc
from strategies import RoundRobin, ZoneSplit, LoadBalance, GreedyOptimal
from baselines.multi_crane.multi_crane_baseline import (
    LinDest, KimDest, LevelingDest, run_mc_heuristic_episode
)
from experiment import parse_instance_file, load_instance_tensor

OUT = 'results/mc_baselines_v2.csv'
RULES = [LinDest(), KimDest(), LevelingDest()]
STRATEGY_MAP = {'S1': RoundRobin, 'S2': ZoneSplit, 'S3': LoadBalance, 'S4': GreedyOptimal}


def main():
    files = sorted(glob.glob('benchmarks/mc_instances/lee_mc/*.txt'))
    rows = []
    t0 = time.time()
    n_total = len(files) * 2 * len(RULES) * len(STRATEGY_MAP)

    for fpath in files:
        fname = os.path.basename(fpath)
        data_lines, crane_starts = parse_instance_file(fpath)
        dims = fname.split('_')[1][1:]
        n_bays, n_rows, n_tiers = int(dims[0:2]), int(dims[2:4]), int(dims[4:6])
        x = load_instance_tensor(data_lines, n_bays, n_rows, n_tiers)

        for n_cranes in (2, 3):
            lb = compute_lb_mc(x, n_bays, n_rows, n_tiers, n_cranes)
            lb_work = lb['work'][0].item()
            lb_makespan = lb['makespan'][0].item()

            for rule in RULES:
                for sname, StrategyCls in STRATEGY_MAP.items():
                    strategy = StrategyCls(n_cranes, n_bays, n_rows)
                    env = MCEnv('cpu', x, n_cranes, crane_start_bays=crane_starts.get(n_cranes))
                    bt0 = time.time()
                    res = run_mc_heuristic_episode(rule, env, strategy, n_bays, n_rows, n_tiers)
                    bt = time.time() - bt0

                    gap_work = 100 * (res['total_cost'] - lb_work) / lb_work
                    gap_makespan = 100 * (res['makespan'] - lb_makespan) / lb_makespan
                    eps = 1e-6
                    assert gap_work >= -eps * max(1.0, lb_work) and gap_makespan >= -eps * max(1.0, lb_makespan), (
                        f'negative gap: {fname} C={n_cranes} {rule.name} {sname} '
                        f'gap_work={gap_work:.4f} gap_makespan={gap_makespan:.4f}'
                    )

                    rows.append({
                        'instance': fname, 'n_cranes': n_cranes, 'method': rule.name, 'strategy': sname,
                        'total_cost': round(res['total_cost'], 1),
                        'makespan': round(res['makespan'], 1),
                        'lb_work': round(lb_work, 1), 'lb_makespan': round(lb_makespan, 1),
                        'gap_work': round(max(gap_work, 0.0), 2), 'gap_makespan': round(max(gap_makespan, 0.0), 2),
                        'interference': res['n_interference'],
                        'interference_wait': round(res['interference_wait'], 1),
                        'a7_reassignments': res['a7_reassignments'],
                        'a7_violations': res.get('a7_violations', 0),
                        'n_steps': res['n_steps'], 'time_s': round(bt, 3),
                    })
                    if len(rows) % 40 == 0:
                        print(f'  [{len(rows)}/{n_total}] {fname} C={n_cranes} {rule.name}+{sname} '
                              f'gap_w={gap_work:.1f}%')

    df = pd.DataFrame(rows)
    os.makedirs('results', exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f'\nTotal: {len(df)} runs in {time.time() - t0:.0f}s -> {OUT}')

    for gap_col in ('gap_work', 'gap_makespan'):
        print(f'\n=== {gap_col} (headline: method x S2 only) ===')
        s2 = df[df['strategy'] == 'S2']
        print(s2.groupby(['n_cranes', 'method'])[gap_col].agg(['mean', 'std']).round(2).to_string())
        print(f'\n=== {gap_col} (full method x strategy matrix) ===')
        print(df.groupby(['n_cranes', 'method', 'strategy'])[gap_col].mean().round(2).unstack().to_string())


if __name__ == '__main__':
    sys.exit(
        'DEPRECATED: writes a separate CSV from the ZeroShot rows (experiment.py), '
        'requiring a manual join. Use `python -m analysis.run_multi_crane_full '
        '--dataset small` instead (ZeroShot + all 3 heuristics x all 4 strategies '
        'in ONE csv) -- see README.md.'
    )
    main()
