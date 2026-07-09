"""Produces paper-ready tables, figures, and statistical analysis from experiment CSV.

Metric schema (v2, post-revision):
    gap_work      -- gap vs LB_work    (total working-time sum objective)
    gap_makespan  -- gap vs LB_makespan (makespan objective, primary)
    makespan, total_cost, interference, interference_wait, a7_reassignments

Usage:
    from analysis.analyze import MCRPAnalyzer
    a = MCRPAnalyzer('results/mcrp_experiment_v2_main.csv')
    a.run_all()
"""

import os, sys, glob, warnings
import pandas as pd
import numpy as np
from scipy import stats
import json

warnings.filterwarnings('ignore', category=RuntimeWarning)

PRIMARY_GAP = 'gap_makespan'
GAP_COLS = ['gap_makespan', 'gap_work']


class MCRPAnalyzer:
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)
        self.df['interference'] = pd.to_numeric(self.df['interference'], errors='coerce').fillna(0)
        self.gap_cols = [c for c in GAP_COLS if c in self.df.columns and self.df[c].notna().any()]
        self.primary = self.gap_cols[0] if self.gap_cols else 'gap_work'
        self._add_scale_column()
        self._add_bay_column()
        self._add_type_column()

    def _dims(self, name):
        parts = str(name).replace('.txt', '').split('_')
        if len(parts) < 2 or len(parts[1]) < 3:
            return None
        return parts[1]

    def _add_scale_column(self):
        def get_scale(name):
            d = self._dims(name)
            if d is None:
                return 'unknown'
            try:
                n_bays = int(d[1:3])
            except ValueError:
                return 'unknown'
            return 'small' if n_bays <= 4 else ('medium' if n_bays <= 10 else 'large')
        self.df['scale'] = self.df['instance'].apply(get_scale)

    def _add_bay_column(self):
        def get_bays(name):
            d = self._dims(name)
            try:
                return int(d[1:3]) if d else -1
            except ValueError:
                return -1
        self.df['n_bays'] = self.df['instance'].apply(get_bays)

    def _add_type_column(self):
        self.df['inst_type'] = self.df['instance'].apply(
            lambda n: 'R' if str(n).startswith('mc_R') else ('U' if str(n).startswith('mc_U') else '?')
        )

    # ------------------------------------------------------------------ #
    # Tables
    # ------------------------------------------------------------------ #

    def table1_gap_comparison(self, gap_col=None):
        gap_col = gap_col or self.primary
        return self.df.groupby(['n_cranes', 'strategy'])[gap_col].agg(['mean', 'std', 'min', 'max'])

    # Backward-compatible alias
    table1_gap_per_strategy = table1_gap_comparison

    def table2_pairwise_wilcoxon(self, n_cranes=2, gap_col=None):
        gap_col = gap_col or self.primary
        data = self.df[self.df['n_cranes'] == n_cranes]
        strategies = sorted(data['strategy'].unique())
        results = []
        for i, s1 in enumerate(strategies):
            for s2 in strategies[i + 1:]:
                g1 = data[data['strategy'] == s1].sort_values('instance')[gap_col].values
                g2 = data[data['strategy'] == s2].sort_values('instance')[gap_col].values
                min_len = min(len(g1), len(g2))
                g1, g2 = g1[:min_len], g2[:min_len]
                diffs = g1 - g2
                if np.all(diffs == 0):
                    results.append({
                        's1': s1, 's2': s2, 'delta_gap': 0.0, 'sigma_d': 0.0,
                        'cohens_d': 0.0, 'p_value': 'identical', 'significant': False
                    })
                    continue
                try:
                    stat_val, p = stats.wilcoxon(g1, g2)
                except ValueError:
                    continue
                delta = diffs.mean()
                sigma_d = diffs.std(ddof=1)
                cohens_d = delta / sigma_d if sigma_d > 0 else 0.0  # paired effect size
                results.append({
                    's1': s1, 's2': s2, 'delta_gap': round(delta, 3),
                    'sigma_d': round(sigma_d, 3),
                    'cohens_d': round(cohens_d, 3),
                    'p_value': f'{p:.4f}',
                    'significant': p < 0.05
                })
        return pd.DataFrame(results)

    def table3_interference_summary(self):
        cols = ['interference']
        for extra in ('interference_wait', 'a7_reassignments', 'a7_violations'):
            if extra in self.df.columns:
                cols.append(extra)
        return self.df.groupby(['n_cranes', 'strategy'])[cols].agg(['mean', 'std', 'max'])

    def table4_gap_by_scale(self, gap_col=None):
        gap_col = gap_col or self.primary
        return self.df.groupby(['scale', 'n_cranes', 'strategy'])[gap_col].agg(['mean', 'std', 'count'])

    def table5_gap_by_bays(self, gap_col=None):
        gap_col = gap_col or self.primary
        return self.df.groupby(['n_bays', 'n_cranes', 'strategy'])[gap_col].agg(['mean', 'std', 'count'])

    def table6_win_count(self, gap_col=None):
        gap_col = gap_col or self.primary
        results = []
        for (inst, nc), sub in self.df.groupby(['instance', 'n_cranes']):
            best_gap = sub[gap_col].min()
            for _, row in sub.iterrows():
                results.append({
                    'n_cranes': nc, 'strategy': row['strategy'],
                    'is_best': row[gap_col] == best_gap
                })
        win_df = pd.DataFrame(results)
        return win_df.groupby(['n_cranes', 'strategy'])['is_best'].sum()

    def table7_gap_by_type(self, gap_col=None):
        gap_col = gap_col or self.primary
        return self.df.groupby(['inst_type', 'n_cranes', 'strategy'])[gap_col].agg(['mean', 'std', 'count'])

    def identify_failure_modes(self, threshold_gap=20.0, gap_col=None):
        gap_col = gap_col or self.primary
        failures = self.df[self.df[gap_col] > threshold_gap]
        if len(failures) == 0:
            return failures
        return failures.groupby('instance').agg({
            gap_col: 'max', 'interference': 'max', 'n_steps': 'max', 'n_bays': 'first'
        }).sort_values(gap_col, ascending=False)

    def compute_speedup(self):
        """Makespan ratio C=2 / C=3 per strategy (>1 means C=3 finishes sooner).

        Only meaningful for the makespan metric; the total-work sum is
        (near-)invariant in C by construction."""
        if 'makespan' not in self.df.columns or self.df['makespan'].isna().all():
            return pd.DataFrame()
        c2 = self.df[self.df['n_cranes'] == 2]
        c3 = self.df[self.df['n_cranes'] == 3]
        merged = c2.merge(c3, on=['instance', 'strategy'], suffixes=('_c2', '_c3'))
        merged['speedup_makespan'] = merged['makespan_c2'] / merged['makespan_c3']
        merged['work_ratio'] = merged['total_cost_c2'] / merged['total_cost_c3']
        return merged.groupby('strategy')[['speedup_makespan', 'work_ratio']].agg(['mean', 'std'])

    # ------------------------------------------------------------------ #
    # Reports
    # ------------------------------------------------------------------ #

    def summary_text(self):
        lines = []
        lines.append('=' * 80)
        lines.append('FULL EXPERIMENT ANALYSIS REPORT (v2 metrics)')
        lines.append(f'Total runs: {len(self.df)}')
        lines.append(f'Unique layouts: {self.df["instance"].nunique()}')
        lines.append(f'Configurations: {self.df["n_cranes"].nunique()} crane counts x {self.df["strategy"].nunique()} strategies')
        lines.append(f'Primary gap metric: {self.primary}')
        lines.append('=' * 80)

        for gap_col in self.gap_cols or [self.primary]:
            lines.append(f'\n\n--- Table 1: {gap_col} per Strategy ---')
            lines.append(self.table1_gap_comparison(gap_col).to_string())

        for nc in sorted(self.df['n_cranes'].unique()):
            lines.append(f'\n\n--- Table 2: Pairwise Wilcoxon ({self.primary}, C={nc}) ---')
            t2 = self.table2_pairwise_wilcoxon(nc)
            lines.append(t2.to_string() if len(t2) > 0 else '  (no comparable pairs)')

        lines.append('\n\n--- Table 3: Interference Summary ---')
        lines.append(self.table3_interference_summary().to_string())

        lines.append('\n\n--- Table 4: Gap by Scale ---')
        lines.append(self.table4_gap_by_scale().to_string())

        lines.append('\n\n--- Table 5: Gap by Number of Bays ---')
        lines.append(self.table5_gap_by_bays().to_string())

        lines.append('\n\n--- Table 6: Win Count ---')
        lines.append(self.table6_win_count().to_string())

        lines.append('\n\n--- Table 7: Gap by Instance Type (R/U) ---')
        lines.append(self.table7_gap_by_type().to_string())

        sp = self.compute_speedup()
        lines.append('\n\n--- Makespan Speedup: C=2 -> C=3 ---')
        lines.append(sp.to_string() if len(sp) > 0 else '  (makespan not available)')

        fail = self.identify_failure_modes(20.0)
        lines.append(f'\n\n--- Failure Modes ({self.primary} > 20%) ---')
        lines.append(f'Count (instances): {len(fail)} of {self.df["instance"].nunique()}')
        n_runs_fail = int((self.df[self.primary] > 20.0).sum()) if self.primary in self.df else 0
        lines.append(f'Count (runs):      {n_runs_fail} of {len(self.df)}')
        if len(fail) > 0:
            lines.append(fail.head(20).to_string())

        lines.append('\n\n--- Performance Summary ---')
        avg_time = self.df['time_s'].mean()
        total_time = self.df['time_s'].sum()
        steps_total = self.df['n_steps'].sum()
        lines.append(f'  Average time per run: {avg_time:.3f}s')
        lines.append(f'  Total experiment time: {total_time:.1f}s')
        lines.append(f'  Total steps across all runs: {int(steps_total)}')
        lines.append(f'  Average inference time per step: {total_time / steps_total * 1000:.2f}ms')

        return '\n'.join(lines)

    def run_all(self, output_dir='results'):
        os.makedirs(output_dir, exist_ok=True)
        report = self.summary_text()
        report_path = os.path.join(output_dir, 'analysis_report.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        print(report)

        fail = self.identify_failure_modes(20.0)
        summary = {
            'total_runs': len(self.df),
            'n_instances': int(self.df['instance'].nunique()),
            'primary_gap': self.primary,
        }
        for gap_col in self.gap_cols or [self.primary]:
            summary[f'avg_{gap_col}_per_strategy'] = {
                f'C={nc}_{s}': round(g, 3)
                for (nc, s), g in self.df.groupby(['n_cranes', 'strategy'])[gap_col].mean().items()
            }
        summary['avg_interference_per_strategy'] = {
            f'C={nc}_{s}': round(g, 1)
            for (nc, s), g in self.df.groupby(['n_cranes', 'strategy'])['interference'].mean().items()
        }
        summary['n_failure_instances'] = len(fail)

        json_path = os.path.join(output_dir, 'analysis_summary.json')
        with open(json_path, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f'\nReport saved: {report_path}')
        print(f'Summary saved: {json_path}')
        return report


def find_latest_csv():
    files = sorted(glob.glob('results/mcrp_experiment_*.csv'), key=os.path.getmtime)
    return files[-1] if files else None


if __name__ == '__main__':
    csv_path = sys.argv[1] if len(sys.argv) > 1 else find_latest_csv()
    if csv_path is None:
        print('No experiment CSV found in results/')
        print('Run: python experiment.py --cranes 2 3 --strategies S1 S2 S3 S4')
        sys.exit(1)
    print(f'Analyzing: {csv_path}')
    a = MCRPAnalyzer(csv_path)
    a.run_all()
