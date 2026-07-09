# HƯỚNG DẪN CHẠY THỰC NGHIỆM

## Chạy từng bước, mỗi bước ~30-60 phút

---

## Bước 1: Chạy Shin instances (cái còn thiếu duy nhất)

```bash
cd CRP_RL
PYTHONUNBUFFERED=1 python analysis/run_missing.py
```

Kết quả: `results/missing/`

---

## Bước 2: Tổng hợp kết quả

```bash
PYTHONUNBUFFERED=1 python -c "
import pandas as pd, glob, os

# Gộp tất cả results
files = glob.glob('results/comprehensive/*.csv') + glob.glob('results/missing/*.csv')
files = [f for f in files if 'all_' not in f]
dfs = []
for f in files:
    try:
        dfs.append(pd.read_csv(f))
        print(f'  {f}: {len(pd.read_csv(f))} runs')
    except: pass

combined = pd.concat(dfs, ignore_index=True)
combined.to_csv('results/all_results_combined.csv', index=False)
print(f'\nTotal: {len(combined)} runs')

# Summary
print('\n=== GAP BY DATASET AND METHOD ===')
pivot = combined.pivot_table(index='dataset', columns='method', values='gap', aggfunc='mean')
print(pivot.to_string())

print('\n=== OVERALL RANKING ===')
ranking = combined.groupby('method')['gap'].agg(['mean','std','min','max']).sort_values('mean')
print(ranking.to_string())
"
```

---

## Các results đã có sẵn:

| Dataset | Methods | File |
|---------|---------|------|
| Lee_random (71) | OriginalModel, ZeroShot, Lin2015, Kim2016, Leveling, Durasevic2025, NearestStack, LowestHeight | `results/comprehensive/Lee_random.csv` |
| Lee_upsidedown (40) | Tương tự | `results/comprehensive/Lee_upsidedown.csv` |
| M-CRP (140×2 cranes) | S1,S2,S3,S4 | `results/mcrp_experiment_*.csv` |
| M-CRP baselines (140×2) | M-Lin2015, M-Kim2016, M-Leveling | `results/mc_extra_baselines/results.csv` |

## Cần chạy thêm (Bước 1):

| Dataset | Methods | File output |
|---------|---------|-------------|
| Shin_random (80) | Tất cả fast + Durasevic subset + BeamSearch 1-bay | `results/missing/shin_random.csv` |
| Shin_upsidedown (80) | Tương tự | `results/missing/shin_ud.csv` |
| Lee_random (3) | BeamSearchCRP | `results/missing/beamsearch_lee.csv` |
