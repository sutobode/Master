"""Entrypoint: run analysis on latest experiment CSV."""

import os, sys, glob
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.analyze import MCRPAnalyzer

csv_files = sorted(glob.glob('results/mcrp_experiment_*.csv'))
if not csv_files:
    print('No experiment CSV found in results/.')
    print('Run: python experiment.py --cranes 2 3 --strategies S1 S2 S3 S4')
    sys.exit(1)

latest = csv_files[-1]
print(f'Analyzing: {latest}')
analyzer = MCRPAnalyzer(latest)
print(analyzer.summary_text())
