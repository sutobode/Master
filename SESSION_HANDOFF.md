# SESSION HANDOFF: M-CRP Zero-shot Transfer

> File này ghi lại toàn bộ context của project để session tiếp theo có thể nắm bắt ngay.

---

## 1. Project Overview

**Repository:** `CRP_RL` — Multi-Crane Container Retrieval Problem  
**Research:** Zero-shot transfer of single-crane DRL policy → multi-crane environments  
**Target:** Transportation Research Part C (TR-C, Q1, IF ~8.9)  
**Backup:** Computers & Operations Research (COR, Q1, IF ~4.5)  

**Key paper:** Shin et al. (2026) — "Learning to Retrieve Containers" (TR-C)  
**Our contribution:** First formal M-CRP definition + zero-shot transfer analysis

---

## 2. Claims (C1-C5)

| Claim | Description | Code Location | Status |
|-------|-------------|---------------|--------|
| C1 | M-CRP definition + NP-hardness | `mcenv/mcenv.py` | ✅ |
| C2 | Theorem 3: LB_MCRP | `bounds/lowerbound_mc.py` | ✅ |
| C3 | 4 Crane Assignment Strategies | `strategies/*.py` | ✅ |
| C4 | Zero-shot empirical evaluation | `experiment.py`, `results/` | ✅ |
| C5 | Public benchmark dataset | `benchmarks/mc_instances/` | ✅ |

---

## 3. Key Results

| Strategy | C=2 gap(%) | C=3 gap(%) | Interference | Complexity |
|----------|-----------|-----------|-------------|-----------|
| S1 RoundRobin | 10.99 ± 5.46 | 11.19 ± 6.59 | 105.9 | O(1) |
| **S2 ZoneSplit** 🏆 | **10.46 ± 5.28** | **10.71 ± 6.44** | **0.5** | **O(B)** |
| S3 LoadBalance | 10.99 ± 5.46 | 11.19 ± 6.59 | 105.9 | O(C) |
| S4 GreedyOptimal | 10.48 ± 5.28 | 10.75 ± 6.48 | 0.0 | O(C²) |

**Single-crane:** ZeroShot 5.63% vs Lin2015 10.28% (beat SOTA by 4.65pp)

---

## 4. Experimental Setup

| Config | Value |
|--------|-------|
| Hardware | CPU laptop (0 GPU) |
| Total runs | 1120 (140 instances × 2 cranes × 4 strategies) |
| Runtime | ~60 phút |
| Dataset | 140 M-CRP instances từ Lee benchmark |
| Model | Shin et al. epoch(100) pretrained |

---

## 5. Key Files

### Documentation
| File | Description |
|------|-------------|
| `docs/latex/crp_rl_paper_Q1.pdf` | Paper (11 trang elsarticle ≈ 18 trang) |
| `docs/latex/crp_rl_paper_Q1.tex` | LaTeX source |
| `docs/latex/presentation.pdf` | 16 slides Beamer |
| `docs/latex/presentation.tex` | Beamer source |
| `README.md` | Full hướng dẫn vận hành |
| `FIX_PLAN.md` | Bug fixes history |
| `docs/superpowers/plans/RESEARCH_PLAN_Q1.md` | Research plan |

### Core Code
| File | Description |
|------|-------------|
| `mcenv/mcenv.py` | Multi-crane environment (A6, A7 constraints) |
| `policy/zero_shot.py` | Zero-shot policy extraction |
| `strategies/` | S1-S4: RoundRobin, ZoneSplit, LoadBalance, GreedyOptimal |
| `bounds/lowerbound_mc.py` | Theorem 3: LB_MCRP |
| `engine/mcrp_inference.py` | M-CRP episode runner |
| `experiment.py` | Experiment pipeline |
| `run_full_experiment.py` | Batch experiment (recovery-safe) |
| `compare_all.py` | Single-crane baseline comparison |
| `analysis/analyze.py` | Statistical analysis (6 tables) |
| `analysis/visualize.py` | Figure generation (6 figures) |

### Results
| File | Description |
|------|-------------|
| `results/mcrp_experiment_20260708_013214.csv` | Raw data (1120 runs) |
| `results/analysis_report.txt` | Full analysis report |
| `results/baseline_comparison.csv` | Single-crane comparison |
| `results/figures/fig1-6_*.png` | 6 figures |

---

## 6. Code Architecture

```
User input (experiment.py)
    ↓
ZeroShotPolicy.get_scores()  ← Frozen pretrained encoder + scorer
    ↓
Strategy.assign()            ← S1 RoundRobin / S2 ZoneSplit / S3 LoadBalance / S4 GreedyOptimal
    ↓
MCEnv.step(dest, crane)     ← Multi-crane environment with A6/A7 constraints
    ↓
run_mcrp_episode()           ← Episode loop (Algorithm 1)
    ↓
Results CSV                  ← Cost, gap, interference, steps
    ↓
MCRPAnalyzer                 ← Statistical analysis + visualization
```

---

## 7. How to Run

```bash
# 1. Setup
pip install -r requirements.txt

# 2. Tests
python -m pytest tests/ -v           # 36/36 passed

# 3. Single-crane comparison
python compare_all.py

# 4. Quick M-CRP test
python experiment.py --quick

# 5. Full experiment (60 min)
python run_full_experiment.py --batch_size 15

# 6. Analysis
python -m analysis.analyze
python -m analysis.visualize
```

---

## 8. Installation Note (LaTeX)

MiKTeX được cài tại: `C:\Users\X1\AppData\Local\Programs\MiKTeX\miktex\bin\x64\`  
PATH: đã thêm vào `.bashrc` nhưng cần add vào Windows PATH để VS Code detect.  
VS Code extension: LaTeX Workshop (James-Yu.latex-workshop) đã cài.

---

## 9. Pending Items

- [ ] Full single-crane benchmark (36 instances) — hiện tại mới 4 instances
- [ ] Expand dataset to 20-30 bay instances (hiện tại ≤10 bays)
- [ ] Author info → thay "Anonymous" bằng tên thật trước khi submit
- [ ] Plagiarism check (iThenticate/Turnitin)
- [ ] Figures DPI check (cần ≥300 for TR-C)

---

## 10. Research Plan Timeline

```
Week 1: Full benchmark + paper submission prep
Week 2: Submit TR-C + prepare slide defense
Week 3: Defense presentation
```

---

## 11. Key Insights for Next Session

- Paper dùng elsarticle format (Elsevier TR-C)
- S2 ZoneSplit là strategy tốt nhất: gap thấp nhất, 0 interference, O(B)
- Zero-shot transfer hoạt động: gap 10.5% không cần retrain
- Sequential simulation là limitation chính (speedup C2→C3 chỉ 1.01x)
- Paper hiện tại: 11 trang (≈18 trang article standard)
