# M-CRP: Multi-Crane Container Retrieval Problem

**Zero-shot Transfer of Single-Crane Deep Reinforcement Learning Policies for Multi-Crane Container Retrieval**

---

## 1. Tổng quan

Repository này thực hiện nghiên cứu về **Zero-shot Transfer** của DRL policy từ single-crane sang multi-crane CRP. Code bao gồm:

- **M-CRP Environment** (`mcenv/`): Multi-crane environment với interference constraints (A6, A7)
- **ZeroShotPolicy** (`policy/`): Trích xuất encoder + scorer từ pretrained model
- **4 Crane Assignment Strategies** (`strategies/`): S1 RoundRobin, S2 ZoneSplit, S3 LoadBalance, S4 GreedyOptimal
- **M-CRP Lower Bound** (`bounds/`): Theorem 3 (LB_MCRP)
- **Full Experiment Pipeline** (`experiment.py`, `run_full_experiment.py`)
- **Analysis + Visualization** (`analysis/`)
- **Baselines** (`baselines/`): 7 single-crane heuristics + multi-crane wrapper

---

## 2. Cài đặt

### Yêu cầu

- Python 3.10+
- PyTorch 2.x
- pandas, numpy, scipy, matplotlib

### Cài dependencies

```bash
pip install -r requirements.txt
```

### Tải pretrained model

Model của Shin et al. (2026) được lưu tại:
```
baselines/models/proposed/epoch(100).pt
baselines/models/online/epoch(100).pt
```

---

## 3. Hướng dẫn chạy

### 3.1. Sinh M-CRP dataset

```bash
python benchmarks/generate_mc_instances.py
```

Tạo 140 instance files trong `benchmarks/mc_instances/lee_mc/`.

### 3.2. Chạy unit tests

```bash
python -m pytest tests/ -v
```

Kỳ vọng: **36/36 passed**

### 3.3. So sánh single-crane baselines

```bash
python compare_all.py
```

Kỳ vọng: ZeroShot thắng tất cả baselines (gap 5.63% vs Lin2015 10.28%)

### 3.4. Quick test M-CRP

```bash
python experiment.py --quick
```

Chạy 3 instances × 2 cranes × 2 strategies = 12 runs (~5-10 giây)

### 3.5. Full experiment (quan trọng nhất)

```bash
python experiment.py --cranes 2 3 --strategies S1 S2 S3 S4
```

Hoặc (khuyến nghị — lưu intermediate results):

```bash
python run_full_experiment.py --batch_size 15
```

- 140 instances × 2 cranes × 4 strategies = 1,120 runs
- Thời gian: ~60 phút (CPU laptop)
- Output: `results/mcrp_experiment_{timestamp}.csv`

### 3.6. Analysis + Visualization

```bash
python -m analysis.analyze       # Statistical analysis report
python -m analysis.visualize     # Generate 6 figures
```

Output:
- `results/analysis_report.txt` — Full analysis (Table 1-6)
- `results/figures/fig1-6_*.png` — Biểu đồ

---

## 4. File Structure

```
CRP_RL/
├── main.py                    # Training entrypoint (original model)
├── trainer.py                 # Training loop
├── model/                     # DRL model (encoder + decoder)
├── env/                       # Single-crane environment
│
├── mcenv/mcenv.py             # Multi-crane environment (M-CRP)
├── policy/zero_shot.py        # Zero-shot policy extraction
├── strategies/                # 4 crane assignment strategies
│   ├── base.py                # Abstract base class
│   ├── round_robin.py         # S1
│   ├── zone_split.py          # S2
│   ├── load_balance.py        # S3
│   └── greedy_optimal.py      # S4
├── bounds/lowerbound_mc.py    # Theorem 3: LB_MCRP
├── engine/mcrp_inference.py   # M-CRP episode runner
│
├── experiment.py              # Full experiment pipeline
├── run_full_experiment.py     # Batch experiment (recovery-safe)
├── compare_all.py             # Single-crane baseline comparison
│
├── analysis/
│   ├── analyze.py             # Statistical analysis (6 tables)
│   └── visualize.py           # Figure generation (6 figures)
│
├── benchmarks/
│   ├── generate_mc_instances.py  # M-CRP dataset generator
│   └── mc_instances/lee_mc/      # 140 generated instances
│
├── baselines/
│   ├── lin2015.py, kim2016.py, ...    # Single-crane heuristics
│   └── multi_crane/                   # Multi-crane baseline wrapper
│
├── tests/                     # 36 unit tests
├── results/                   # Experiment outputs (gitignored)
│   ├── mcrp_experiment_*.csv  # Raw results
│   ├── analysis_report.txt    # Statistical report
│   └── figures/               # 6 figures
│
└── docs/
    └── latex/
        ├── crp_rl_paper_Q1.pdf     # Paper (12 trang, Q1 format)
        └── crp_rl_paper_Q1.tex     # LaTeX source
```

---

## 5. Kết quả chính

| Strategy | C=2 gap(%) | C=3 gap(%) | Interference | Complexity |
|----------|-----------|-----------|-------------|-----------|
| S1 RoundRobin | 10.99 ± 5.46 | 11.19 ± 6.59 | 105.9 ± 62.7 | O(1) |
| **S2 ZoneSplit** 🏆 | **10.46 ± 5.28** | **10.71 ± 6.44** | **0.5 ± 1.5** | **O(B)** |
| S3 LoadBalance | 10.99 ± 5.46 | 11.19 ± 6.59 | 105.9 ± 62.7 | O(C) |
| S4 GreedyOptimal | 10.48 ± 5.28 | 10.75 ± 6.48 | 0.0 ± 0.0 | O(C²) |

### Key findings

1. **Zero-shot transfer HOẠT ĐỘNG** — gap 10-11% không cần retrain, 0 GPU
2. **S2 ZoneSplit TỐT NHẤT** — gap thấp nhất, gần như 0 interference, O(B)
3. **S2 vs S4 không khác biệt** (p > 0.1) — ZoneSplit đơn giản là đủ
4. **Spatial awareness > task balancing** — S3 (LoadBalance) = S1 (RoundRobin)

---

## 6. Claim → Code → Result Traceability

| Claim | Code | Test | Result |
|-------|------|------|--------|
| C1: M-CRP definition | `mcenv/mcenv.py` | `tests/test_mcenv.py` | A6+A7 constraints verified |
| C2: Theorem 3 LB_MCRP | `bounds/lowerbound_mc.py` | `tests/test_lowerbound_mc.py` | Backward compatible C=1 (diff < 2%) |
| C3: 4 strategies | `strategies/*.py` | `tests/test_strategies.py` | All pass, correct interface |
| C4: Zero-shot evaluation | `experiment.py`, `run_full_experiment.py` | `tests/test_policy.py` | 1120 runs, Wilcoxon p < 0.001 |
| C5: Public benchmark | `benchmarks/generate_mc_instances.py` | `tests/test_mc_instances.py` | 140 instances generated |

---

## 7. References

- Shin et al. (2026). "Learning to Retrieve Containers." *Transportation Research Part C*.
- Kwon et al. (2020). "POMO." *NeurIPS*.
- Lin et al. (2015). "A heuristic algorithm for CRP." *Transportation Research Part E*.
