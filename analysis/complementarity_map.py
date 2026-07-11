"""Complementarity map (Phase 1.3 of docs/superpowers/plans/2026-07-11-master-plan.md):
which method wins where, across scale x type x n_cranes x strategy. This is
the evidence base for the "DRL and Leveling are complementary experts, not
uniformly ordered" framing, and the motivating table for any future
fusion/portfolio attempt (Gate G in the master plan).

Usage: python -m analysis.complementarity_map
"""
import pandas as pd


def single_crane_map():
    df = pd.read_csv('results/single_crane_lee.csv')
    dfs = pd.read_csv('results/single_crane_shin.csv')
    df = pd.concat([df, dfs], ignore_index=True)

    zs = df[df.method == 'ZeroShot'].set_index('instance')
    lv = df[df.method == 'Leveling'].set_index('instance')
    idx = zs.index.intersection(lv.index)
    d = zs.loc[idx, 'gap'] - lv.loc[idx, 'gap']
    meta = zs.loc[idx, ['n_bays', 'n_tiers', 'type']]
    meta['zs_wins'] = d < 0
    print('=== Setting A: ZS win-rate by (n_bays, n_tiers, type) ===')
    g = meta.groupby(['n_bays', 'n_tiers', 'type'])['zs_wins'].agg(['mean', 'count'])
    g.columns = ['zs_win_rate', 'n']
    print(g.to_string())
    return g


def multi_crane_map(csv_path, label):
    df = pd.read_csv(csv_path)
    df['type'] = df['instance'].str.extract(r'mc_([A-Za-z])')[0]
    rows = []
    for c in sorted(df.n_cranes.unique()):
        for strat in sorted(df.strategy.unique()):
            zs = df[(df.method == 'ZeroShot') & (df.strategy == strat) & (df.n_cranes == c)].set_index('instance')
            lv = df[(df.method == 'M-Leveling') & (df.strategy == strat) & (df.n_cranes == c)].set_index('instance')
            idx = zs.index.intersection(lv.index)
            d = zs.loc[idx, 'gap_makespan'] - lv.loc[idx, 'gap_makespan']
            typ = df.set_index('instance').loc[idx, 'type']
            typ = typ[~typ.index.duplicated()]
            for ty, grp_idx in typ.groupby(typ).groups.items():
                dd = d.loc[grp_idx]
                rows.append({'setting': label, 'n_cranes': c, 'strategy': strat, 'type': ty,
                             'n': len(dd), 'zs_win_rate': (dd < 0).mean(), 'mean_diff': dd.mean()})
    out = pd.DataFrame(rows)
    print(f'\n=== Setting B ({label}): ZS vs M-Leveling win-rate by (n_cranes, strategy, type) ===')
    print(out.to_string(index=False))
    return out


def main():
    m1 = single_crane_map()
    m2 = multi_crane_map('results/multi_crane_small.csv', 'small')
    m3 = multi_crane_map('results/multi_crane_large.csv', 'large')
    m1.to_csv('results/complementarity_single_crane.csv')
    pd.concat([m2, m3], ignore_index=True).to_csv('results/complementarity_multi_crane.csv', index=False)
    print('\nSaved -> results/complementarity_single_crane.csv, results/complementarity_multi_crane.csv')


if __name__ == '__main__':
    main()
