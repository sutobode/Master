"""Produces paper-ready tables and figures from experiment CSV."""

import pandas as pd
import numpy as np
from scipy import stats


class MCRPAnalyzer:
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)
        self.df['interference'] = pd.to_numeric(self.df['interference'], errors='coerce').fillna(0)

    def table1_gap_comparison(self):
        """Gap(LB_MCRP)% per strategy x crane count."""
        return self.df.groupby(['n_cranes', 'strategy'])['gap'].agg(['mean', 'std', 'min', 'max'])

    def table2_pairwise_wilcoxon(self, n_cranes=2):
        """Wilcoxon signed-rank between strategy pairs."""
        data = self.df[self.df['n_cranes'] == n_cranes]
        strategies = sorted(data['strategy'].unique())
        results = []
        for i, s1 in enumerate(strategies):
            for s2 in strategies[i+1:]:
                g1 = data[data['strategy'] == s1].sort_values('instance')['gap'].values
                g2 = data[data['strategy'] == s2].sort_values('instance')['gap'].values
                min_len = min(len(g1), len(g2))
                g1, g2 = g1[:min_len], g2[:min_len]
                stat, p = stats.wilcoxon(g1, g2)
                delta = g1.mean() - g2.mean()
                results.append({
                    's1': s1, 's2': s2, 'delta_gap': round(delta, 2),
                    'p_value': f'{p:.4f}', 'significant': p < 0.05
                })
        return pd.DataFrame(results)

    def table3_interference_summary(self):
        return self.df.groupby(['n_cranes', 'strategy'])['interference'].agg(['mean', 'std'])

    def table4_cost_by_scale(self):
        """Add scale column derived from instance name."""
        def get_scale(name):
            if 'mc_R01' in name or 'mc_U01' in name:
                return 'small'
            elif 'mc_R06' in name or 'mc_R08' in name or 'mc_R10' in name:
                return 'medium'
            else:
                return 'large'
        df = self.df.copy()
        df['scale'] = df['instance'].apply(get_scale)
        return df.groupby(['scale', 'n_cranes', 'strategy'])['gap'].agg(['mean', 'std'])

    def identify_failure_modes(self, threshold_gap=20.0):
        failures = self.df[self.df['gap'] > threshold_gap]
        return failures.groupby('instance').agg({
            'gap': 'max', 'interference': 'max', 'n_steps': 'max'
        }).sort_values('gap', ascending=False)

    def compute_effect_size(self, strategy_a, strategy_b, n_cranes=2):
        data = self.df[self.df['n_cranes'] == n_cranes]
        a = data[data['strategy'] == strategy_a]['gap'].values
        b = data[data['strategy'] == strategy_b]['gap'].values
        pooled = np.sqrt((np.var(a) + np.var(b)) / 2)
        return (a.mean() - b.mean()) / pooled if pooled > 0 else 0.0

    def summary_text(self):
        lines = []
        t1 = self.table1_gap_comparison()
        lines.append('=== Table 1: Gap(LB_MCRP)% per strategy ===')
        lines.append(t1.to_string())
        for nc in [2, 3]:
            t2 = self.table2_pairwise_wilcoxon(nc)
            lines.append(f'\n=== Table 2: Pairwise Wilcoxon ({nc} cranes) ===')
            lines.append(t2.to_string())
        t3 = self.table3_interference_summary()
        lines.append('\n=== Table 3: Interference summary ===')
        lines.append(t3.to_string())
        t4 = self.table4_cost_by_scale()
        lines.append('\n=== Table 4: Gap by scale ===')
        lines.append(t4.to_string())
        fail = self.identify_failure_modes(20.0)
        lines.append(f'\n=== Failure modes (gap > 20%) ===')
        lines.append(f'Count: {len(fail)}')
        if len(fail) > 0:
            lines.append(fail.head(10).to_string())
        return '\n'.join(lines)
