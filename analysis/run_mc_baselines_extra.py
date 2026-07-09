"""Run M-Kim2016 and M-Leveling on all 140 M-CRP instances (~20 min).

Usage: python analysis/run_mc_baselines_extra.py
"""

import sys, os, time, torch, glob, json
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mcenv.mcenv import MCEnv
from bounds.lowerbound_mc import compute_lb_mc
from strategies import ZoneSplit
from baselines.kim2016 import Kim2016
from baselines.leveling import Leveling
from experiment import parse_instance_file, load_instance_tensor

OUT = 'results/mc_extra_baselines'
os.makedirs(OUT, exist_ok=True)

BASELINES = {'M-Kim2016': Kim2016, 'M-Leveling': Leveling}

def run_baseline_on_all():
    """Run each baseline on all 140 M-CRP instances via ZoneSplit wrapper."""
    files = sorted(glob.glob('benchmarks/mc_instances/lee_mc/*.txt'))
    results = []
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
        n_cranes_cfg = 2 if '_c2' in fname else 3

        x = load_instance_tensor(data_lines, n_bays, n_rows, n_tiers)
        lb = compute_lb_mc(x, n_bays, n_rows, n_tiers, n_cranes_cfg).item()

        for bname, BCls in BASELINES.items():
            try:
                bl = BCls()
                bt0 = time.time()
                result = bl.run(x)
                bt = time.time() - bt0
                cost = float(result[0])
                gap = 100 * (cost - lb) / lb if lb > 0 else 0
                results.append({'instance': fname, 'n_cranes': n_cranes_cfg, 'method': bname, 'cost': round(cost,1), 'lb_mc': round(lb,1), 'gap': round(gap,2), 'interference': 0, 'time_s': round(bt,3)})
            except Exception as e:
                print(f'  ERROR {bname} {fname}: {e}')

    df = pd.DataFrame(results)
    df.to_csv(f'{OUT}/results.csv', index=False)
    print(f'\nTotal: {len(df)} runs in {time.time()-t0:.0f}s')

    for nc in [2, 3]:
        print(f'\nC={nc}:')
        for method in sorted(df[df['n_cranes']==nc]['method'].unique()):
            m = df[(df['n_cranes']==nc)&(df['method']==method)]['gap']
            print(f'  {method}: mean={m.mean():.2f}% std={m.std():.2f}% (n={len(m)})')

if __name__ == '__main__':
    sys.exit(
        'DEPRECATED: uses the pre-revision parse_instance_file()/compute_lb_mc() '
        'API (silently swallowed by a broad try/except) and old _c2/_c3-suffixed '
        'instance filenames, neither of which exist anymore — this script would '
        'silently process zero instances. Use `analysis.run_mc_baselines_v2` '
        'instead — see README.md.'
    )
    run_baseline_on_all()
