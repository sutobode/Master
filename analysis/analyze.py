"""Produces paper-ready tables, figures, and statistical analysis from experiment CSV.

Usage:
    from analysis.analyze import MCRPAnalyzer
    a = MCRPAnalyzer('results/mcrp_experiment_*.csv')
    a.run_all()
"""

import os, sys, glob, warnings
import pandas as pd
import numpy as np
from scipy import stats
import json

warnings.filterwarnings('ignore', category=RuntimeWarning)


class MCRPAnalyzer:
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)
        self.df['interference'] = pd.to_numeric(self.df['interference'], errors='coerce').fillna(0)
        self._add_scale_column()
        self._add_bay_column()

    def _add_scale_column(self):
        def get_scale(name):
            parts = str(name).replace('.txt', '').split('_')
            if len(parts) < 2:
                return 'unknown'
            dims = parts[1][1:]
            if len(dims) < 2:
                return 'unknown'
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
        self.df['scale'] = self.df['instance'].apply(get_scale)

    def _add_bay_column(self):
        def get_bays(name):
            parts = str(name).replace('.txt', '').split('_')
            if len(parts) < 2:
                return -1
            dims = parts[1][1:]
            if len(dims) < 2:
                return -1
            try:
                return int(dims[0:2])
            except ValueError:
                return -1
        self.df['n_bays'] = self.df['instance'].apply(get_bays)

    def table1_gap_per_strategy(self):
        return self.df.groupby(['n_cranes', 'strategy'])['gap'].agg(['mean', 'std', 'min', 'max'])

    def table2_pairwise_wilcoxon(self, n_cranes=2):
        data = self.df[self.df['n_cranes'] == n_cranes]
        strategies = sorted(data['strategy'].unique())
        results = []
        for i, s1 in enumerate(strategies):
            for s2 in strategies[i+1:]:
                g1 = data[data['strategy'] == s1].sort_values('instance')['gap'].values
                g2 = data[data['strategy'] == s2].sort_values('instance')['gap'].values
                min_len = min(len(g1), len(g2))
                g1, g2 = g1[:min_len], g2[:min_len]
                if np.all(g1 == g2):
                    continue
                try:
                    stat_val, p = stats.wilcoxon(g1, g2)
                except ValueError:
                    continue
                delta = g1.mean() - g2.mean()
                pooled = np.sqrt((np.var(g1) + np.var(g2)) / 2)
                cohens_d = delta / pooled if pooled > 0 else 0.0
                results.append({
                    's1': s1, 's2': s2, 'delta_gap': round(delta, 3),
                    'cohens_d': round(cohens_d, 3),
                    'p_value': f'{p:.4f}',
                    'significant': p < 0.05
                })
        return pd.DataFrame(results)

    def table3_interference_summary(self):
        return self.df.groupby(['n_cranes', 'strategy'])['interference'].agg(['mean', 'std', 'min', 'max'])

    def table4_gap_by_scale(self):
        return self.df.groupby(['scale', 'n_cranes', 'strategy'])['gap'].agg(['mean', 'std', 'count'])

    def table5_gap_by_bays(self):
        return self.df.groupby(['n_bays', 'n_cranes', 'strategy'])['gap'].agg(['mean', 'std', 'count'])

    def table6_win_count(self):
        results = []
        for inst in self.df['instance'].unique():
            for nc in self.df[self.df['instance'] == inst]['n_cranes'].unique():
                sub = self.df[(self.df['instance'] == inst) & (self.df['n_cranes'] == nc)]
                best_gap = sub['gap'].min()
                for _, row in sub.iterrows():
                    results.append({
                        'instance': inst, 'n_cranes': nc,
                        'strategy': row['strategy'],
                        'gap': row['gap'],
                        'is_best': row['gap'] == best_gap
                    })
        win_df = pd.DataFrame(results)
        return win_df.groupby(['n_cranes', 'strategy'])['is_best'].sum()

    def identify_failure_modes(self, threshold_gap=20.0):
        failures = self.df[self.df['gap'] > threshold_gap]
        return failures.groupby('instance').agg({
            'gap': 'max', 'interference': 'max', 'n_steps': 'max', 'n_bays': 'first'
        }).sort_values('gap', ascending=False)

    def compute_speedup(self):
        c1_data = self.df[self.df['n_cranes'] == 2].copy()
        c2_data = self.df[self.df['n_cranes'] == 3].copy()
        merged = c1_data.merge(
            c2_data, on=['instance', 'strategy'],
            suffixes=('_c2', '_c3')
        )
        merged['speedup_2v3'] = merged['cost_c2'] / merged['cost_c3']
        return merged.groupby('strategy')['speedup_2v3'].agg(['mean', 'std', 'min', 'max'])

    def run_all(self, output_dir='results'):
        os.makedirs(output_dir, exist_ok=True)

        lines = []
        lines.append('=' * 80)
        lines.append('FULL EXPERIMENT ANALYSIS REPORT')
        lines.append(f'Source: {os.path.basename(csv_path) if "csv_path" in dir() else "unknown"}')
        lines.append(f'Total runs: {len(self.df)}')
        lines.append(f'Instances: {self.df["instance"].nunique()}')
        lines.append(f'Configurations: {self.df["n_cranes"].nunique()} crane counts × {self.df["strategy"].nunique()} strategies')
        lines.append('=' * 80)

        lines.append('\n\n--- TABLE 1: Gap(LB_MCRP)% per Strategy ---')
        t1 = self.table1_gap_per_strategy()
        lines.append(t1.to_string())

        lines.append('\n\n--- TABLE 2: Pairwise Wilcoxon (2 cranes) ---')
        t2_2 = self.table2_pairwise_wilcoxon(2)
        lines.append(t2_2.to_string() if len(t2_2) > 0 else '  (No significant differences)')

        lines.append('\n\n--- TABLE 2: Pairwise Wilcoxon (3 cranes) ---')
        t2_3 = self.table2_pairwise_wilcoxon(3)
        lines.append(t2_3.to_string() if len(t2_3) > 0 else '  (No significant differences)')

        lines.append('\n\n--- TABLE 3: Interference Summary ---')
        t3 = self.table3_interference_summary()
        lines.append(t3.to_string())

        lines.append('\n\n--- TABLE 4: Gap by Scale ---')
        t4 = self.table4_gap_by_scale()
        lines.append(t4.to_string())

        lines.append('\n\n--- TABLE 5: Gap by Number of Bays ---')
        t5 = self.table5_gap_by_bays()
        lines.append(t5.to_string())

        lines.append('\n\n--- TABLE 6: Win Count ---')
        t6 = self.table6_win_count()
        lines.append(t6.to_string())

        lines.append('\n\n--- Speedup: C=2 vs C=3 ---')
        sp = self.compute_speedup()
        lines.append(sp.to_string())

        lines.append('\n\n--- Failure Modes (gap > 20%) ---')
        fail = self.identify_failure_modes(20.0)
        lines.append(f'Count: {len(fail)}')
        if len(fail) > 0:
            lines.append(fail.head(20).to_string())
        else:
            lines.append('  (None — all instances within acceptable range)')

        lines.append('\n\n--- Performance Summary ---')
        avg_time = self.df['time_s'].mean()
        total_time = self.df['time_s'].sum()
        lines.append(f'  Average time per run: {avg_time:.3f}s')
        lines.append(f'  Total experiment time: {total_time:.1f}s')

        report = '\n'.join(lines)
        report_path = os.path.join(output_dir, 'analysis_report.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        print(report)

        json_path = os.path.join(output_dir, 'analysis_summary.json')
        summary = {
            'total_runs': len(self.df),
            'n_instances': int(self.df['instance'].nunique()),
            'avg_gap_per_strategy': {
                f'C={nc}_{s}': round(g, 3)
                for (nc, s), g in self.df.groupby(['n_cranes', 'strategy'])['gap'].mean().items()
            },
            'avg_interference_per_strategy': {
                f'C={nc}_{s}': round(g, 1)
                for (nc, s), g in self.df.groupby(['n_cranes', 'strategy'])['interference'].mean().items()
            },
            'n_failures': len(fail),
            'total_experiment_time_s': round(total_time, 1),
        }
        with open(json_path, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f'\nReport saved: {report_path}')
        print(f'Summary saved: {json_path}')
        return report


def find_latest_csv():
    files = sorted(glob.glob('results/mcrp_experiment_*.csv'))
    if not files:
        return None
    return files[-1]


if __name__ == '__main__':
    csv_path = find_latest_csv()
    if csv_path is None:
        print('No experiment CSV found in results/')
        print('Run: python experiment.py --cranes 2 3 --strategies S1 S2 S3 S4')
        sys.exit(1)
    print(f'Analyzing: {csv_path}')
    a = MCRPAnalyzer(csv_path)
    a.run_all()
