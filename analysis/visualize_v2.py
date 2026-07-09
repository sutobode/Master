"""Regenerate paper figures from the v2 experiment CSVs (300 DPI).

Inputs:
    results/mcrp_experiment_v2_main.csv   (Experiment 2: strategies)
    results/mc_baselines_v2.csv           (Experiment 3: MC heuristics)
    results/single_crane_v2.csv           (Experiment 1: single-crane)
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


def _bays(name):
    return int(str(name).split('_')[1][1:3])


def fig_gap_by_strategy(df):
    fig, axes = plt.subplots(1, 2, figsize=(8, 3), sharey=True)
    for ax, gap_col, title in zip(axes, ['gap_makespan', 'gap_work'],
                                  ['Makespan gap', 'Total-work gap']):
        stats = df.groupby(['n_cranes', 'strategy'])[gap_col].agg(['mean', 'std'])
        width = 0.35
        xs = np.arange(len(STRAT_ORDER))
        for i, nc in enumerate(sorted(df['n_cranes'].unique())):
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
    fig, ax = plt.subplots(figsize=(5, 3))
    stats = df.groupby(['n_cranes', 'strategy'])['interference_wait'].mean()
    width = 0.35
    xs = np.arange(len(STRAT_ORDER))
    for i, nc in enumerate(sorted(df['n_cranes'].unique())):
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
    fig, ax = plt.subplots(figsize=(5.5, 3))
    d = df[df['n_cranes'] == 2].copy()
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
    fig, ax = plt.subplots(figsize=(5, 3))
    c2 = df[df['n_cranes'] == 2]
    c3 = df[df['n_cranes'] == 3]
    m = c2.merge(c3, on=['instance', 'strategy'], suffixes=('_c2', '_c3'))
    m['speedup'] = m['makespan_c2'] / m['makespan_c3']
    data = [m[m['strategy'] == s]['speedup'].values for s in STRAT_ORDER]
    ax.boxplot(data, labels=STRAT_ORDER, showmeans=True)
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


def fig_mc_baselines(df3, df2):
    """ZS+S2 vs M-heuristics, makespan gap, grouped bars."""
    fig, ax = plt.subplots(figsize=(5.5, 3))
    zs = df2[df2['strategy'] == 'S2'].copy()
    zs['method'] = 'ZeroShot+S2'
    cols = ['instance', 'n_cranes', 'method', 'gap_makespan', 'gap_work']
    allm = pd.concat([zs[cols], df3[cols]])
    stats = allm.groupby(['n_cranes', 'method'])['gap_makespan'].mean()
    methods = ['ZeroShot+S2', 'M-Lin2015', 'M-Leveling', 'M-Kim2016']
    xs = np.arange(len(methods))
    width = 0.35
    for i, nc in enumerate([2, 3]):
        vals = [stats.loc[(nc, m)] for m in methods]
        ax.bar(xs + (i - 0.5) * width, vals, width, label=f'C={nc}')
    ax.set_xticks(xs)
    ax.set_xticklabels(methods, rotation=10)
    ax.set_ylabel('Makespan gap (%)')
    ax.legend()
    fig.tight_layout()
    fig.savefig(f'{OUT}/fig6_mc_baselines.png', bbox_inches='tight')
    plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    df2 = pd.read_csv('results/mcrp_experiment_v2_main.csv')
    fig_gap_by_strategy(df2)
    fig_interference_wait(df2)
    fig_gap_by_bays(df2)
    fig_speedup(df2)
    print('Experiment-2 figures done')

    if os.path.exists('results/single_crane_v2.csv'):
        fig_single_crane(pd.read_csv('results/single_crane_v2.csv'))
        print('Experiment-1 figure done')
    if os.path.exists('results/mc_baselines_v2.csv'):
        fig_mc_baselines(pd.read_csv('results/mc_baselines_v2.csv'), df2)
        print('Experiment-3 figure done')
    print(f'Figures -> {OUT}/')


if __name__ == '__main__':
    main()
