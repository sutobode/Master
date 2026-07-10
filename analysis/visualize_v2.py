"""Regenerate paper figures from the unified experiment CSVs (300 DPI).

Inputs (produced by analysis/run_single_crane_full.py and
analysis/run_multi_crane_full.py -- see README.md):
    results/multi_crane_small.csv   (Setting B, small scale: ZeroShot + 3 heuristics x 4 strategies)
    results/multi_crane_large.csv   (Setting B, large scale -- optional)
    results/single_crane_lee.csv    (Setting A, small scale: 6 methods)
    results/single_crane_shin.csv   (Setting A, large scale -- optional)
Outputs: results/figures_v2/*.png

Usage: python -m analysis.visualize_v2
"""

import os, sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OUT = 'results/figures_v2'
plt.rcParams.update({'figure.dpi': 300, 'savefig.dpi': 300, 'font.size': 9})

STRAT_ORDER = ['S1', 'S2', 'S3', 'S4']
STRAT_LABEL = {'S1': 'S1 RoundRobin', 'S2': 'S2 ZoneSplit',
               'S3': 'S3 LoadBalance', 'S4': 'S4 GreedyLookahead'}
MC_METHODS = ['ZeroShot', 'M-Lin2015', 'M-Kim2016', 'M-Leveling']


def _bays(name):
    return int(str(name).split('_')[1][1:3])


def _add_mc_type(df):
    df = df.copy()
    df['inst_type'] = df['instance'].apply(
        lambda n: 'Random' if str(n).startswith('mc_R') else ('Upside-down' if str(n).startswith('mc_U') else '?')
    )
    return df


def fig_gap_by_strategy(df):
    """ZeroShot only -- the proposed method's strategy comparison (Setting B)."""
    zs = df[df['method'] == 'ZeroShot']
    fig, axes = plt.subplots(1, 2, figsize=(8, 3), sharey=True)
    for ax, gap_col, title in zip(axes, ['gap_makespan', 'gap_work'],
                                  ['Makespan gap', 'Total-work gap']):
        stats = zs.groupby(['n_cranes', 'strategy'])[gap_col].agg(['mean', 'std'])
        width = 0.35
        xs = np.arange(len(STRAT_ORDER))
        for i, nc in enumerate(sorted(zs['n_cranes'].unique())):
            means = [stats.loc[(nc, s), 'mean'] for s in STRAT_ORDER]
            stds = [stats.loc[(nc, s), 'std'] for s in STRAT_ORDER]
            ax.bar(xs + (i - 0.5) * width, means, width, yerr=stds, capsize=2,
                   label=f'C={nc}')
        ax.set_xticks(xs)
        ax.set_xticklabels(STRAT_ORDER)
        ax.set_title(title)
        ax.set_ylabel('Gap vs LB (%)')
        ax.legend()
    fig.tight_layout()
    fig.savefig(f'{OUT}/fig1_gap_by_strategy.png', bbox_inches='tight')
    plt.close(fig)


def fig_interference_wait(df):
    zs = df[df['method'] == 'ZeroShot']
    fig, ax = plt.subplots(figsize=(5, 3))
    stats = zs.groupby(['n_cranes', 'strategy'])['interference_wait'].mean()
    width = 0.35
    xs = np.arange(len(STRAT_ORDER))
    for i, nc in enumerate(sorted(zs['n_cranes'].unique())):
        vals = [stats.loc[(nc, s)] for s in STRAT_ORDER]
        ax.bar(xs + (i - 0.5) * width, vals, width, label=f'C={nc}')
    ax.set_xticks(xs)
    ax.set_xticklabels([STRAT_LABEL[s] for s in STRAT_ORDER], rotation=15)
    ax.set_ylabel('Mean A6 waiting time (time units)')
    ax.legend()
    fig.tight_layout()
    fig.savefig(f'{OUT}/fig2_interference_wait.png', bbox_inches='tight')
    plt.close(fig)


def fig_gap_by_bays(df):
    zs = df[df['method'] == 'ZeroShot']
    fig, ax = plt.subplots(figsize=(5.5, 3))
    d = zs[zs['n_cranes'] == 2].copy()
    d['bays'] = d['instance'].map(_bays)
    for s in STRAT_ORDER:
        g = d[d['strategy'] == s].groupby('bays')['gap_makespan'].mean()
        ax.plot(g.index, g.values, marker='o', label=STRAT_LABEL[s])
    ax.set_xlabel('Number of bays')
    ax.set_ylabel('Makespan gap (%)')
    ax.legend()
    fig.tight_layout()
    fig.savefig(f'{OUT}/fig3_gap_by_bays.png', bbox_inches='tight')
    plt.close(fig)


def fig_speedup(df):
    zs = df[df['method'] == 'ZeroShot']
    fig, ax = plt.subplots(figsize=(5, 3))
    c2 = zs[zs['n_cranes'] == 2]
    c3 = zs[zs['n_cranes'] == 3]
    m = c2.merge(c3, on=['instance', 'strategy'], suffixes=('_c2', '_c3'))
    m['speedup'] = m['makespan_c2'] / m['makespan_c3']
    data = [m[m['strategy'] == s]['speedup'].values for s in STRAT_ORDER]
    ax.boxplot(data, showmeans=True)
    ax.set_xticklabels(STRAT_ORDER)
    ax.axhline(1.0, color='grey', lw=0.8, ls='--')
    ax.set_ylabel('Makespan speedup C=2 $\\rightarrow$ C=3')
    fig.tight_layout()
    fig.savefig(f'{OUT}/fig4_speedup.png', bbox_inches='tight')
    plt.close(fig)


def fig_single_crane(df1):
    fig, ax = plt.subplots(figsize=(5.5, 3))
    order = df1.groupby('method')['gap'].mean().sort_values().index
    d = df1.copy()
    d['bays'] = d['n_bays']
    for m in order:
        g = d[d['method'] == m].groupby('bays')['gap'].mean()
        ax.plot(g.index, g.values, marker='o', label=m)
    ax.set_xlabel('Number of bays')
    ax.set_ylabel('Gap (%)')
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(f'{OUT}/fig5_single_crane_by_bays.png', bbox_inches='tight')
    plt.close(fig)


def fig_mc_baselines(df):
    """ZeroShot+S2 vs M-heuristics+S2, makespan gap, grouped bars.

    All 4 methods now live in one CSV (results/multi_crane_small.csv), so this
    is a plain filter -- no cross-file merge needed anymore."""
    s2 = df[df['strategy'] == 'S2']
    fig, ax = plt.subplots(figsize=(5.5, 3))
    stats = s2.groupby(['n_cranes', 'method'])['gap_makespan'].mean()
    xs = np.arange(len(MC_METHODS))
    width = 0.35
    for i, nc in enumerate(sorted(s2['n_cranes'].unique())):
        vals = [stats.loc[(nc, m)] for m in MC_METHODS]
        ax.bar(xs + (i - 0.5) * width, vals, width, label=f'C={nc}')
    ax.set_xticks(xs)
    ax.set_xticklabels(MC_METHODS, rotation=10)
    ax.set_ylabel('Makespan gap (%), strategy=S2')
    ax.legend()
    fig.tight_layout()
    fig.savefig(f'{OUT}/fig6_mc_baselines.png', bbox_inches='tight')
    plt.close(fig)


def fig_mc_gap_by_type(df):
    """Random vs upside-down, ZeroShot+S2, makespan gap -- Setting B."""
    zs = _add_mc_type(df[(df['method'] == 'ZeroShot') & (df['strategy'] == 'S2')])
    fig, ax = plt.subplots(figsize=(5, 3))
    stats = zs.groupby(['inst_type', 'n_cranes'])['gap_makespan'].agg(['mean', 'std'])
    types = ['Random', 'Upside-down']
    xs = np.arange(len(types))
    width = 0.35
    for i, nc in enumerate(sorted(zs['n_cranes'].unique())):
        means = [stats.loc[(t, nc), 'mean'] for t in types]
        stds = [stats.loc[(t, nc), 'std'] for t in types]
        ax.bar(xs + (i - 0.5) * width, means, width, yerr=stds, capsize=3, label=f'C={nc}')
    ax.set_xticks(xs)
    ax.set_xticklabels(types)
    ax.set_ylabel('Makespan gap (%), ZeroShot+S2')
    ax.legend()
    fig.tight_layout()
    fig.savefig(f'{OUT}/fig7_mc_gap_by_type.png', bbox_inches='tight')
    plt.close(fig)


def fig_single_crane_gap_by_type(df1):
    """Random vs upside-down, per method, gap -- Setting A."""
    type_label = {'R': 'Random', 'U': 'Upside-down'}
    d = df1.copy()
    d['type_label'] = d['type'].map(type_label)
    methods = list(d.groupby('method')['gap'].mean().sort_values().index)
    fig, ax = plt.subplots(figsize=(6.5, 3))
    stats = d.groupby(['method', 'type_label'])['gap'].mean()
    types = ['Random', 'Upside-down']
    xs = np.arange(len(methods))
    width = 0.35
    for i, t in enumerate(types):
        vals = [stats.loc[(m, t)] if (m, t) in stats.index else np.nan for m in methods]
        ax.bar(xs + (i - 0.5) * width, vals, width, label=t)
    ax.set_xticks(xs)
    ax.set_xticklabels(methods, rotation=15, fontsize=7)
    ax.set_ylabel('Gap (%)')
    ax.legend()
    fig.tight_layout()
    fig.savefig(f'{OUT}/fig8_single_crane_gap_by_type.png', bbox_inches='tight')
    plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)

    if os.path.exists('results/multi_crane_small.csv'):
        df2 = pd.read_csv('results/multi_crane_small.csv')
        fig_gap_by_strategy(df2)
        fig_interference_wait(df2)
        fig_gap_by_bays(df2)
        fig_speedup(df2)
        fig_mc_baselines(df2)
        fig_mc_gap_by_type(df2)
        print('Setting B (multi-crane, small scale) figures done')
    else:
        print('SKIP: results/multi_crane_small.csv not found -- run analysis.run_multi_crane_full --dataset small first')

    if os.path.exists('results/single_crane_lee.csv'):
        df1 = pd.read_csv('results/single_crane_lee.csv')
        fig_single_crane(df1)
        fig_single_crane_gap_by_type(df1)
        print('Setting A (single-crane, small scale) figures done')
    else:
        print('SKIP: results/single_crane_lee.csv not found -- run analysis.run_single_crane_full --dataset lee first')

    print(f'Figures -> {OUT}/')


if __name__ == '__main__':
    main()
