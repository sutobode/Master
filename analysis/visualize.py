"""Generate paper-ready figures from experiment results.

Figures:
1. gap_by_strategy.png — Bar chart: gap per strategy × crane count
2. gap_by_scale.png — Bar chart: gap by yard scale
3. gap_by_bays.png — Line chart: gap as function of number of bays
4. interference_comparison.png — Bar chart: interference events per strategy
5. speedup_analysis.png — Bar chart: C=2 vs C=3 speedup
6. win_rate.png — Bar chart: how often each strategy wins
"""

import os, sys, glob, warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

plt.rcParams.update({
    'figure.figsize': (10, 6),
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'legend.fontsize': 10,
    'figure.dpi': 300,
})


def find_latest_csv():
    files = sorted(glob.glob('results/mcrp_experiment_*.csv'))
    return files[-1] if files else None


def add_scale_column(df):
    def get_scale(name):
        parts = str(name).replace('.txt', '').split('_')
        if len(parts) < 2:
            return 'unknown'
        dims = parts[1][1:]
        try:
            n_bays = int(dims[0:2])
        except ValueError:
            return 'unknown'
        if n_bays <= 4:
            return 'small'
        elif n_bays <= 10:
            return 'medium'
        else:
            return 'large'
    df['scale'] = df['instance'].apply(get_scale)
    return df


def add_bay_column(df):
    def get_bays(name):
        parts = str(name).replace('.txt', '').split('_')
        if len(parts) < 2:
            return -1
        dims = parts[1][1:]
        try:
            return int(dims[0:2])
        except ValueError:
            return -1
    df['n_bays'] = df['instance'].apply(get_bays)
    return df


STRATEGY_COLORS = {'S1': '#4C72B0', 'S2': '#DD8452', 'S3': '#55A868', 'S4': '#C44E52'}
STRATEGY_LABELS = {'S1': 'S1 RoundRobin', 'S2': 'S2 ZoneSplit', 'S3': 'S3 LoadBalance', 'S4': 'S4 GreedyOptimal'}


def fig1_gap_by_strategy(df, output_dir='results/figures'):
    """Grouped bar chart: gap per (strategy, crane_count)."""
    os.makedirs(output_dir, exist_ok=True)
    stats = df.groupby(['n_cranes', 'strategy'])['gap'].agg(['mean', 'std'])
    cranes = sorted(df['n_cranes'].unique())
    strategies = sorted(df['strategy'].unique())

    x = np.arange(len(strategies))
    width = 0.35

    fig, ax = plt.subplots()
    for i, nc in enumerate(cranes):
        offsets = x + (i - 0.5) * width
        means = [stats.loc[(nc, s), 'mean'] for s in strategies]
        stds = [stats.loc[(nc, s), 'std'] for s in strategies]
        bars = ax.bar(offsets, means, width, yerr=stds, capsize=3,
                      label=f'{int(nc)} cranes', color=[STRATEGY_COLORS[s] for s in strategies],
                      alpha=0.7 + 0.15 * i)
    ax.set_xlabel('Strategy')
    ax.set_ylabel('Gap(LB_MCRP) %')
    ax.set_title('Gap by Strategy and Crane Count')
    ax.set_xticks(x)
    ax.set_xticklabels([STRATEGY_LABELS[s] for s in strategies], rotation=15)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    path = os.path.join(output_dir, 'fig1_gap_by_strategy.png')
    plt.savefig(path)
    plt.close()
    print(f'Saved: {path}')


def fig2_gap_by_scale(df, output_dir='results/figures'):
    """Grouped bar chart: gap by yard scale."""
    os.makedirs(output_dir, exist_ok=True)
    df = add_scale_column(df)
    scale_order = ['small', 'medium', 'large']
    stats = df.groupby(['scale', 'n_cranes', 'strategy'])['gap'].agg(['mean', 'std'])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    strategies = sorted(df['strategy'].unique())

    for idx, nc in enumerate([2, 3]):
        ax = axes[idx]
        x = np.arange(len(scale_order))
        width = 0.2

        for si, s in enumerate(strategies):
            offsets = x + (si - 1.5) * width
            means = []
            for sc in scale_order:
                try:
                    means.append(stats.loc[(sc, nc, s), 'mean'])
                except KeyError:
                    means.append(0)
            ax.bar(offsets, means, width, label=STRATEGY_LABELS[s],
                   color=STRATEGY_COLORS[s], alpha=0.85)
        ax.set_title(f'{int(nc)} Cranes')
        ax.set_xlabel('Yard Scale')
        ax.set_xticks(x)
        ax.set_xticklabels([sc.capitalize() for sc in scale_order])
        ax.grid(axis='y', alpha=0.3)
        if idx == 0:
            ax.set_ylabel('Gap(LB_MCRP) %')
    axes[0].legend(loc='upper left')
    fig.suptitle('Gap by Yard Scale', fontsize=14)
    plt.tight_layout()
    path = os.path.join(output_dir, 'fig2_gap_by_scale.png')
    plt.savefig(path)
    plt.close()
    print(f'Saved: {path}')


def fig3_gap_by_bays(df, output_dir='results/figures'):
    """Line chart: gap as function of number of bays."""
    os.makedirs(output_dir, exist_ok=True)
    df = add_bay_column(df)
    df = df[df['n_bays'] > 0]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    strategies = sorted(df['strategy'].unique())

    for idx, nc in enumerate([2, 3]):
        ax = axes[idx]
        sub = df[df['n_cranes'] == nc]
        bay_stats = sub.groupby(['n_bays', 'strategy'])['gap'].agg(['mean', 'std']).reset_index()

        for s in strategies:
            s_data = bay_stats[bay_stats['strategy'] == s]
            if len(s_data) == 0:
                continue
            ax.errorbar(s_data['n_bays'], s_data['mean'], yerr=s_data['std'],
                       label=STRATEGY_LABELS[s], color=STRATEGY_COLORS[s],
                       marker='o', capsize=3, alpha=0.8)
        ax.set_xlabel('Number of Bays')
        ax.set_title(f'{int(nc)} Cranes')
        ax.grid(alpha=0.3)
        if idx == 0:
            ax.set_ylabel('Gap(LB_MCRP) %')
        ax.legend()
    fig.suptitle('Gap as Function of Number of Bays', fontsize=14)
    plt.tight_layout()
    path = os.path.join(output_dir, 'fig3_gap_by_bays.png')
    plt.savefig(path)
    plt.close()
    print(f'Saved: {path}')


def fig4_interference(df, output_dir='results/figures'):
    """Bar chart: interference events per (strategy, crane_count)."""
    os.makedirs(output_dir, exist_ok=True)
    stats = df.groupby(['n_cranes', 'strategy'])['interference'].agg(['mean', 'std'])
    cranes = sorted(df['n_cranes'].unique())
    strategies = sorted(df['strategy'].unique())

    x = np.arange(len(strategies))
    width = 0.35

    fig, ax = plt.subplots()
    for i, nc in enumerate(cranes):
        offsets = x + (i - 0.5) * width
        means = [stats.loc[(nc, s), 'mean'] for s in strategies]
        stds = [stats.loc[(nc, s), 'std'] for s in strategies]
        ax.bar(offsets, means, width, yerr=stds, capsize=3,
               label=f'{int(nc)} cranes', alpha=0.7 + 0.15 * i)
    ax.set_xlabel('Strategy')
    ax.set_ylabel('Interference Events')
    ax.set_title('Interference Events by Strategy')
    ax.set_xticks(x)
    ax.set_xticklabels([STRATEGY_LABELS[s] for s in strategies], rotation=15)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    path = os.path.join(output_dir, 'fig4_interference.png')
    plt.savefig(path)
    plt.close()
    print(f'Saved: {path}')


def fig5_speedup(df, output_dir='results/figures'):
    """Bar chart: speedup from C=2 to C=3."""
    os.makedirs(output_dir, exist_ok=True)
    merged = df[df['n_cranes'] == 2].merge(
        df[df['n_cranes'] == 3],
        on=['instance', 'strategy'], suffixes=('_2', '_3')
    )
    merged['speedup'] = merged['cost_2'] / merged['cost_3']
    stats = merged.groupby('strategy')['speedup'].agg(['mean', 'std'])

    fig, ax = plt.subplots()
    strategies = sorted(stats.index)
    means = [stats.loc[s, 'mean'] for s in strategies]
    stds = [stats.loc[s, 'std'] for s in strategies]
    ax.bar(range(len(strategies)), means, yerr=stds, capsize=5,
           color=[STRATEGY_COLORS[s] for s in strategies], alpha=0.85)
    ax.axhline(y=1.5, color='gray', linestyle='--', alpha=0.5, label='Ideal (1.5x)')
    ax.set_xlabel('Strategy')
    ax.set_xticks(range(len(strategies)))
    ax.set_xticklabels([STRATEGY_LABELS[s] for s in strategies], rotation=15)
    ax.set_ylabel('Speedup (C=2 cost / C=3 cost)')
    ax.set_title('Speedup from 2 to 3 Cranes')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    path = os.path.join(output_dir, 'fig5_speedup.png')
    plt.savefig(path)
    plt.close()
    print(f'Saved: {path}')


def fig6_win_rate(df, output_dir='results/figures'):
    """Bar chart: win rate per strategy."""
    os.makedirs(output_dir, exist_ok=True)

    results = []
    for inst in df['instance'].unique():
        for nc in df[df['instance'] == inst]['n_cranes'].unique():
            sub = df[(df['instance'] == inst) & (df['n_cranes'] == nc)]
            best_gap = sub['gap'].min()
            for _, row in sub.iterrows():
                results.append({
                    'instance': inst, 'n_cranes': nc,
                    'strategy': row['strategy'],
                    'is_best': abs(row['gap'] - best_gap) < 1e-10
                })
    win_df = pd.DataFrame(results)
    win_rate = win_df.groupby(['n_cranes', 'strategy'])['is_best'].mean() * 100
    win_rate = win_rate.reset_index()

    fig, ax = plt.subplots()
    cranes = sorted(win_rate['n_cranes'].unique())
    strategies = sorted(win_rate['strategy'].unique())
    x = np.arange(len(strategies))
    width = 0.35

    for i, nc in enumerate(cranes):
        offsets = x + (i - 0.5) * width
        rates = []
        for s in strategies:
            subset = win_rate[(win_rate['n_cranes'] == nc) & (win_rate['strategy'] == s)]
            rates.append(subset['is_best'].values[0] if len(subset) > 0 else 0)
        ax.bar(offsets, rates, width, label=f'{int(nc)} cranes',
               alpha=0.7 + 0.15 * i)
    ax.set_xlabel('Strategy')
    ax.set_ylabel('Win Rate (%)')
    ax.set_title('How Often Each Strategy Achieves Best Gap')
    ax.set_xticks(x)
    ax.set_xticklabels([STRATEGY_LABELS[s] for s in strategies], rotation=15)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    path = os.path.join(output_dir, 'fig6_win_rate.png')
    plt.savefig(path)
    plt.close()
    print(f'Saved: {path}')


def fig7_single_crane_comparison(output_dir='results/figures'):
    """Bar chart: all 7 single-crane baselines vs ZeroShot on 50 instances."""
    os.makedirs(output_dir, exist_ok=True)
    csv_path = 'results/final_comprehensive/single_crane_all_baselines.csv'
    if not os.path.exists(csv_path):
        print(f'SKIP fig7: {csv_path} not found (run run_comprehensive.py first)')
        return

    df = pd.read_csv(csv_path)
    methods = ['Random', 'NearestStack', 'LowestHeight', 'Durasevic2025', 'Kim2016', 'Leveling', 'Lin2015', 'ZeroShot']
    colors = ['#d62728', '#ff7f0e', '#ffbb78', '#bcbd22', '#2ca02c', '#98df8a', '#1f77b4', '#17becf']

    fig, ax = plt.subplots(figsize=(12, 6))
    x_pos = np.arange(len(methods))
    means, errs = [], []
    for m in methods:
        col = f'{m}_gap' if m != 'ZeroShot' else 'zero_shot_gap'
        if col in df.columns:
            vals = df[col].dropna()
            means.append(vals.mean())
            errs.append(vals.std())
        else:
            means.append(0)
            errs.append(0)

    bars = ax.bar(x_pos, means, yerr=errs, capsize=5, color=colors, alpha=0.85, edgecolor='gray', linewidth=0.5)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(methods, rotation=30, ha='right')
    ax.set_ylabel('Gap vs Lower Bound (%)')
    ax.set_title('Single-Crane Comparison (50 Lee Instances)')
    ax.grid(axis='y', alpha=0.3)

    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f'{val:.1f}%',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    path = os.path.join(output_dir, 'fig7_single_crane_comparison.png')
    plt.savefig(path)
    plt.close()
    print(f'Saved: {path}')


def fig8_multi_crane_baselines(output_dir='results/figures'):
    """Grouped bar chart: ZS+S2 vs M-Lin2015 on all 140 instances."""
    os.makedirs(output_dir, exist_ok=True)
    csv_path = 'results/final_comprehensive/multi_crane_all_baselines.csv'
    if not os.path.exists(csv_path):
        print(f'SKIP fig8: {csv_path} not found')
        return

    df = pd.read_csv(csv_path)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    for ax, nc in [(ax1, 2), (ax2, 3)]:
        sub = df[df['n_cranes'] == nc]
        methods = sorted(sub['method'].unique())
        x = np.arange(len(methods))
        means = [sub[sub['method'] == m]['gap'].mean() for m in methods]
        stds = [sub[sub['method'] == m]['gap'].std() for m in methods]
        bars = ax.bar(x, means, yerr=stds, capsize=5, color=['#d62728', '#17becf'], alpha=0.85, edgecolor='gray')
        ax.set_xticks(x)
        ax.set_xticklabels(methods)
        ax.set_ylabel('Gap vs LB_MCRP (%)')
        ax.set_title(f'{int(nc)} Cranes')
        ax.grid(axis='y', alpha=0.3)
        for bar, val in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, f'{val:.1f}%',
                    ha='center', va='bottom', fontsize=11, fontweight='bold')

    fig.suptitle('Multi-Crane Baseline Comparison (140 M-CRP Instances)', fontsize=14)
    plt.tight_layout()
    path = os.path.join(output_dir, 'fig8_multi_crane_baselines.png')
    plt.savefig(path)
    plt.close()
    print(f'Saved: {path}')


def generate_all(csv_path=None):
    if csv_path is None:
        csv_path = find_latest_csv()
    if csv_path is None:
        print('No experiment CSV found. Run experiment.py first.')
        return

    print(f'Loading: {csv_path}')
    df = pd.read_csv(csv_path)
    df['interference'] = pd.to_numeric(df['interference'], errors='coerce').fillna(0)

    report_lines = []
    report_lines.append(f'Generating figures from: {csv_path}')
    report_lines.append(f'Total runs: {len(df)}')
    report_lines.append('')

    fig1_gap_by_strategy(df)
    report_lines.append('[OK] fig1_gap_by_strategy.png')

    fig2_gap_by_scale(df)
    report_lines.append('[OK] fig2_gap_by_scale.png')

    fig3_gap_by_bays(df)
    report_lines.append('[OK] fig3_gap_by_bays.png')

    fig4_interference(df)
    report_lines.append('[OK] fig4_interference.png')

    fig5_speedup(df)
    report_lines.append('[OK] fig5_speedup.png')

    fig6_win_rate(df)
    report_lines.append('[OK] fig6_win_rate.png')

    fig7_single_crane_comparison()
    report_lines.append('[OK] fig7_single_crane_comparison.png')

    fig8_multi_crane_baselines()
    report_lines.append('[OK] fig8_multi_crane_baselines.png')

    report_path = 'results/figures/generation_report.txt'
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))
    print('\n'.join(report_lines))
    print(f'\nReport: {report_path}')
    print(f'All figures saved to results/figures/')


if __name__ == '__main__':
    generate_all()
