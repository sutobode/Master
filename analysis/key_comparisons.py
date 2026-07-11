"""Statistical upgrade for key ZeroShot-vs-baseline comparisons (Phase 1.2 of
docs/superpowers/plans/2026-07-11-master-plan.md): Wilcoxon signed-rank with
Holm correction across the comparison family, Cohen's d (paired), and
sigma_d (paired-difference std, the quantity a real power analysis for a
paired test must be based on -- see the audit finding about the old
power-analysis paragraph in the .tex, which used the wrong/marginal std).

Usage: python -m analysis.key_comparisons
"""
import pandas as pd
import numpy as np
from scipy.stats import wilcoxon


def holm_correction(p_values):
    """Holm-Bonferroni step-down correction (no statsmodels dependency).
    Returns adjusted p-values in the original order, plus a reject-at-0.05 mask."""
    p = np.asarray(p_values)
    order = np.argsort(p)
    n = len(p)
    adj_sorted = np.empty(n)
    running_max = 0.0
    for i, idx in enumerate(order):
        val = min((n - i) * p[idx], 1.0)
        running_max = max(running_max, val)
        adj_sorted[i] = running_max
    p_adj = np.empty(n)
    p_adj[order] = adj_sorted
    return p_adj, p_adj < 0.05


def paired_stats(a, b, name):
    d = a - b
    d = d[~np.isnan(d)]
    n = len(d)
    mean_d = d.mean()
    sigma_d = d.std(ddof=1)
    cohens_d = mean_d / sigma_d if sigma_d > 0 else np.nan
    try:
        stat, p = wilcoxon(d)
    except ValueError:
        p = 1.0
    win_a = int((d < 0).sum())
    return {'comparison': name, 'n': n, 'mean_diff': mean_d, 'sigma_d': sigma_d,
            'cohens_d': cohens_d, 'p_raw': p, 'win_rate_pct': 100 * win_a / n}


def run_single_crane():
    df = pd.read_csv('results/single_crane_shin.csv')
    zs = df[df.method == 'ZeroShot'].set_index('instance')
    lv = df[df.method == 'Leveling'].set_index('instance')
    results = []
    for (b, t), grp in zs.groupby(['n_bays', 'n_tiers']):
        idx = grp.index.intersection(lv.index)
        a = zs.loc[idx, 'gap']
        bb = lv.loc[idx, 'gap']
        results.append(paired_stats(a.values, bb.values, f'ShinA: ZS vs Leveling @ bay{b}_tier{t}'))
    return results


def run_multi_crane():
    df = pd.read_csv('results/multi_crane_large.csv')
    results = []
    for c in [2, 3]:
        for metric, strat in [('gap_makespan', 'S4'), ('gap_work', 'S2')]:
            zs = df[(df.method == 'ZeroShot') & (df.strategy == strat) & (df.n_cranes == c)].set_index('instance')
            lv = df[(df.method == 'M-Leveling') & (df.strategy == strat) & (df.n_cranes == c)].set_index('instance')
            idx = zs.index.intersection(lv.index)
            results.append(paired_stats(zs.loc[idx, metric].values, lv.loc[idx, metric].values,
                                         f'MClarge: ZS vs M-Leveling {metric} C={c} {strat}'))
    return results


def main():
    all_results = run_single_crane() + run_multi_crane()
    dfres = pd.DataFrame(all_results)
    p_adj, reject = holm_correction(dfres['p_raw'].values)
    dfres['p_holm'] = p_adj
    dfres['significant_holm'] = reject
    dfres = dfres[['comparison', 'n', 'mean_diff', 'sigma_d', 'cohens_d',
                   'win_rate_pct', 'p_raw', 'p_holm', 'significant_holm']]
    pd.set_option('display.width', 160)
    pd.set_option('display.max_colwidth', 50)
    print(dfres.round(4).to_string(index=False))
    dfres.to_csv('results/key_comparisons.csv', index=False)
    print('\nSaved -> results/key_comparisons.csv')


if __name__ == '__main__':
    main()
