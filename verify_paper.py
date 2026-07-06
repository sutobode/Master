"""Verify pretrained model reproduces original paper results (Tables 1, 2, A.7, A.8)."""

import sys, os, time, argparse, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
from baselines.lowerbound import get_wt_lb
from baselines.lin2015 import Lin2015
from baselines.kim2016 import Kim2016
from benchmarks.benchmarks import find_and_process_file
from model.model import Model

MODEL_PATH = 'baselines/models/proposed/epoch(100).pt'

PAPER_GAPS_R_TYPE = {
    'R011606_0070': 7.7, 'R021606_0140': 7.1, 'R041606_0280': 5.3,
    'R061606_0430': 5.9, 'R081606_0570': 5.8, 'R101606_0720': 5.9,
    'R011608_0090': 10.2, 'R021608_0190': 9.4, 'R041608_0380': 10.0,
    'R061608_0570': 10.2,
}

PAPER_LB_R_TYPE = {
    'R011606_0070': [4520.4, 4544.4, 4599.6, 4714.8, 4558.8],
    'R021606_0140': [12373.5, 11612.7, 12305.4, 12949.8, 12068.4],
    'R041606_0280': [29393.2, 28005.7, 28164.8, 28063.6, 28696.6],
    'R061606_0430': [46704.5, 45977.8, 45840.0, 45533.4, 46105.3],
    'R081606_0570': [63535.5, 63783.9, 64565.3, 63768.9, 64902.9],
    'R101606_0720': [83976.4, 82196.0, 82932.9, 83099.2, 82410.5],
    'R011608_0090': [6146.4, 6094.8, 6386.4, 6234.0, 6180.0],
    'R021608_0190': [17308.2, 17039.7, 17779.8, 17115.3, 17554.8],
    'R041608_0380': [39623.0, 39272.6, 38556.6, 39542.9, 39399.0],
    'R061608_0570': [62023.6, 62080.0, 62551.8, 62960.8, 63951.0],
}

PAPER_WT_R_TYPE = {
    'R011606_0070': [4807.2, 4909.2, 4872.0, 5184.0, 4940.4],
    'R021606_0140': [13067.1, 12594.3, 13151.4, 13939.8, 12913.2],
    'R041606_0280': [31252.0, 29469.1, 29502.8, 29638.0, 30020.2],
    'R061606_0430': [49427.3, 48741.4, 48545.2, 47965.0, 49053.1],
    'R081606_0570': [66863.9, 67141.5, 69125.9, 67910.7, 68192.9],
    'R101606_0720': [88874.2, 86938.6, 87841.3, 87990.8, 87415.7],
    'R011608_0090': [6835.2, 6796.8, 7060.8, 6709.2, 6814.8],
    'R021608_0190': [19479.0, 18581.7, 19549.8, 18411.3, 18944.4],
    'R041608_0380': [43565.0, 43713.8, 42304.2, 43138.7, 43336.2],
    'R061608_0570': [68784.2, 68978.6, 68339.8, 69155.2, 70186.4],
}


def load_model():
    args = argparse.Namespace(
        device=torch.device('cpu'), embed_dim=128, n_encode_layers=3, n_heads=8,
        ff_hidden=512, tanh_c=10, lstm=True, bay_embedding=True,
        online=False, online_known_num=None
    )
    model = Model(args)
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
    model.eval()
    model.decoder.set_sampler('greedy')
    return model


def run_baselines(x, bay, row, tier):
    """Run all baselines on single instance, return (cost, gap)."""
    results = {}
    for name, cls in [('Lin2015', Lin2015), ('Kim2016', Kim2016)]:
        try:
            bl = cls()
            res = bl.run(x)
            cost = res[0] if isinstance(res, tuple) else res
            bl_lb = get_wt_lb(x)
            gap = 100 * (float(cost) - float(bl_lb)) / float(bl_lb)
            results[name] = (float(cost), float(gap))
        except Exception as e:
            results[name] = (None, None)
    return results


def main():
    model = load_model()
    device = torch.device('cpu')

    rows = [16]
    tiers = [6, 8]
    bays = [1, 2, 4, 6, 8, 10]

    data_path = 'benchmarks/Lee_instances'
    all_rows = []
    total_gap_diff = 0
    total_per_instance_count = 0

    print('=' * 90)
    print(f'{"Config":>15s} | {"Paper LB":>8s} | {"Paper WT":>8s} | {"Our WT":>8s} | {"Our Gap":>7s} | {"Paper Gap":>8s} | Diff | LB match?')
    print('=' * 90)

    for tier in tiers:
        for row in rows:
            for bay in bays:
                if tier == 8 and bay in [8, 10]:
                    continue
                for idx in range(1, 6):
                    x, name = find_and_process_file(data_path, 'random', bay, row, tier, idx, no_print=True)
                    prefix = name.replace('.txt', '')

                    with torch.no_grad():
                        wt, _ = model(x.to(device), None)
                    our_cost = wt[0].item()
                    our_lb = float(get_wt_lb(x))

                    config_key = '_'.join(prefix.split('_')[:2])  # e.g., R011606_0070
                    paper_lb = PAPER_LB_R_TYPE.get(config_key, [None]*5)[idx-1]
                    paper_wt = PAPER_WT_R_TYPE.get(config_key, [None]*5)[idx-1]

                    our_gap = 100 * (our_cost - our_lb) / our_lb
                    paper_gap = 100 * (paper_wt - paper_lb) / paper_lb if paper_wt and paper_lb else None
                    diff_gap = our_gap - paper_gap if paper_gap else None
                    lb_ok = abs(our_lb - paper_lb) / paper_lb < 0.01 if paper_lb else None

                    status = 'OK' if lb_ok else 'MISMATCH'
                    diff_str = f'{diff_gap:+.2f}' if diff_gap else 'N/A'

                    paper_lb_str = f'{paper_lb:>8.1f}' if paper_lb is not None else '  N/A   '
                    paper_wt_str = f'{paper_wt:>8.1f}' if paper_wt is not None else '  N/A   '
                    paper_gap_str = f'{paper_gap:>7.1f}%' if paper_gap is not None else '  N/A   '
                    print(f'{prefix:>15s} | {paper_lb_str} | {paper_wt_str} | {our_cost:>8.1f} | {our_gap:>6.2f}% | {paper_gap_str} | {diff_str:>5s} | {status}')

                    all_rows.append({
                        'instance': prefix, 'config': config_key, 'idx': idx,
                        'paper_lb': paper_lb, 'paper_wt': paper_wt,
                        'our_lb': round(our_lb, 1), 'our_wt': round(our_cost, 1),
                        'our_gap': round(our_gap, 2), 'paper_gap': round(paper_gap, 2) if paper_gap else None,
                        'diff_gap': round(diff_gap, 2) if diff_gap else None,
                        'lb_match': lb_ok
                    })

                    if diff_gap is not None:
                        total_gap_diff += abs(diff_gap)
                        total_per_instance_count += 1

    # Summary
    df = pd.DataFrame(all_rows)
    avg_diff = total_gap_diff / total_per_instance_count if total_per_instance_count > 0 else 0
    paper_match_pct = 100 * df['our_wt'].notna().sum() / len(df)
    lb_match_pct = 100 * df['lb_match'].sum() / len(df) if 'lb_match' in df else 0

    print('\n' + '=' * 90)
    print(f'Average absolute gap diff vs paper: {avg_diff:.2f}%')

    lb_match_count = df['lb_match'].dropna().sum() if 'lb_match' in df else 0
    lb_match_total = df['lb_match'].dropna().shape[0]
    if lb_match_total > 0:
        print(f'LB match with paper: {100*lb_match_count/lb_match_total:.1f}% ({int(lb_match_count)}/{lb_match_total})')
    print()

    print('--- Average gap per config (matching paper Table 1) ---')
    print(f'{"Config":>12s} | {"Paper Gap":>9s} | {"Our Gap":>8s} | {"Diff":>6s}')
    print('-' * 42)
    for config_key in sorted(set(df['config'])):
        sub = df[df['config'] == config_key]
        our = sub['our_gap'].mean()
        paper = sub['paper_gap'].dropna().mean() if sub['paper_gap'].notna().any() else None
        if paper is not None:
            diff = our - paper
            print(f'{config_key:>12s} | {paper:>8.2f}% | {our:>7.2f}% | {diff:+5.2f}%')
        else:
            print(f'{config_key:>12s} | {"N/A":>8s} | {our:>7.2f}% | {"N/A"}')

    # Save
    path = 'results/paper_verification.csv'
    df.to_csv(path, index=False)
    print(f'\nSaved: {path}')


if __name__ == '__main__':
    main()
