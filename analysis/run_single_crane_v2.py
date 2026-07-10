"""Experiment 1 (v2): single-crane comparison on the 70 standard Lee instances.

Methods: Original pretrained model (greedy), the ZeroShotPolicy extraction
driven through MCEnv(n_cranes=1) (the proposed method's actual code path,
not just the original model), and 4 published heuristics. The two DRL rows
are expected to be numerically identical (the zero-shot claim's backward-
compatibility contract) -- this experiment verifies that empirically on all
70 instances rather than asserting it from a handful of unit-test instances.

Deterministic: a fixed seed is set before every heuristic run (Lin2015 uses
random tie-breaking). The earlier comprehensive run had unseeded duplicate
runs with inconsistent costs.

Usage: python -m analysis.run_single_crane_v2
Output: results/single_crane_v2.csv
"""

import sys, os, time, argparse
import pandas as pd
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.model import Model
from policy.zero_shot import ZeroShotPolicy
from mcenv.mcenv import MCEnv
from engine.mcrp_inference import run_mcrp_episode
from strategies import RoundRobin
from baselines.lin2015 import Lin2015
from baselines.kim2016 import Kim2016
from baselines.leveling import Leveling
from baselines.durasevic2025 import Durasevic2025
from baselines.lowerbound import get_wt_lb
from benchmarks.benchmarks import find_and_process_file

OUT = 'results/single_crane_v2.csv'

BAYS6 = [1, 2, 4, 6, 8, 10]
BAYS8 = [1, 2, 4, 6]


def standard_instances():
    for tier, bays in ((6, BAYS6), (8, BAYS8)):
        for bay in bays:
            for inst_type, idxs in (('random', range(1, 6)), ('upsidedown', range(1, 3))):
                for idx in idxs:
                    yield inst_type, bay, 16, tier, idx


def load_original_model():
    args = argparse.Namespace(
        device=torch.device('cpu'), embed_dim=128, n_encode_layers=3, n_heads=8,
        ff_hidden=512, tanh_c=10, lstm=True, bay_embedding=True,
        online=False, online_known_num=None
    )
    model = Model(args)
    model.load_state_dict(torch.load('baselines/models/proposed/epoch(100).pt', map_location='cpu'))
    model.eval()
    model.decoder.set_sampler('greedy')
    return model


def main():
    model = load_original_model()
    policy = ZeroShotPolicy(device=torch.device('cpu'))
    heuristics = {
        'Lin2015': Lin2015,
        'Kim2016': Kim2016,
        'Leveling': Leveling,
        'Durasevic2025': Durasevic2025,
    }

    rows = []
    max_zs_diff_pct = 0.0
    t0 = time.time()
    for inst_type, bay, row, tier, idx in standard_instances():
        x, name = find_and_process_file('benchmarks/Lee_instances', inst_type, bay, row, tier, idx, no_print=True)
        lb = float(get_wt_lb(x))

        with torch.no_grad():
            wt, _ = model(x, None)
        cost = float(wt[0].item())
        rows.append({'instance': name, 'type': 'R' if inst_type == 'random' else 'U',
                     'n_bays': bay, 'n_tiers': tier, 'method': 'OriginalModel',
                     'cost': round(cost, 1), 'lb': round(lb, 1),
                     'gap': round(100 * (cost - lb) / lb, 2)})

        # The proposed method's actual code path: ZeroShotPolicy driven
        # through MCEnv(n_cranes=1), not just an assumption that it matches
        # the original model. Empirically verifies backward-compatibility on
        # the full 70-instance benchmark instead of a handful of unit-test cases.
        env = MCEnv('cpu', x, n_cranes=1)
        zs_result = run_mcrp_episode(policy, env, RoundRobin(1, bay, row), bay, row, tier)
        cost_zs = zs_result['total_cost']
        diff_pct = 100 * abs(cost_zs - cost) / cost if cost > 0 else 0.0
        max_zs_diff_pct = max(max_zs_diff_pct, diff_pct)
        rows.append({'instance': name, 'type': 'R' if inst_type == 'random' else 'U',
                     'n_bays': bay, 'n_tiers': tier, 'method': 'ZeroShot',
                     'cost': round(cost_zs, 1), 'lb': round(lb, 1),
                     'gap': round(100 * (cost_zs - lb) / lb, 2)})

        for mname, MCls in heuristics.items():
            torch.manual_seed(1234)
            bt0 = time.time()
            try:
                cost_h, _ = MCls().run(x)
            except Exception as e:
                print(f'  ERROR {mname} {name}: {e}')
                continue
            rows.append({'instance': name, 'type': 'R' if inst_type == 'random' else 'U',
                         'n_bays': bay, 'n_tiers': tier, 'method': mname,
                         'cost': round(float(cost_h), 1), 'lb': round(lb, 1),
                         'gap': round(100 * (float(cost_h) - lb) / lb, 2)})
        print(f'  {name} done ({time.time() - t0:.0f}s elapsed)')

    assert max_zs_diff_pct < 0.01, (
        f'ZeroShot(C=1) diverged from OriginalModel by up to {max_zs_diff_pct:.4f}% '
        f'across the 70-instance benchmark — backward-compatibility contract broken'
    )
    print(f'\nBackward-compat check: max |ZeroShot - OriginalModel| = {max_zs_diff_pct:.4f}% across 70 instances')

    df = pd.DataFrame(rows)
    os.makedirs('results', exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f'\nTotal: {len(df)} runs in {time.time() - t0:.0f}s -> {OUT}')
    print('\n=== Gap by method (70 standard Lee instances) ===')
    print(df.groupby('method')['gap'].agg(['mean', 'std', 'min', 'max']).round(2).sort_values('mean').to_string())
    print('\n=== By type ===')
    print(df.groupby(['method', 'type'])['gap'].mean().round(2).unstack().to_string())


if __name__ == '__main__':
    main()
