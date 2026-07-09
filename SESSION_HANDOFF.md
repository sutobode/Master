# SESSION HANDOFF: M-CRP Zero-shot Transfer

> ⚠️ **10/07/2026 — Đang trong quá trình sửa lỗi lớn phát hiện qua ultra-review (Theorem 3 sai, cost-model bug, dataset double-count...).**
> **Mọi số liệu/claims trong phần dưới đây (8.x.2026) hiện KHÔNG còn đúng** — paper và code đã thay đổi đáng kể, thí nghiệm cần chạy lại.
> Đọc `docs/superpowers/plans/2026-07-10-revision-handoff.md` TRƯỚC — đó là bàn giao mới nhất, có đầy đủ danh sách thay đổi + lệnh chạy lại.
> Nội dung gốc bên dưới giữ nguyên làm tham chiếu lịch sử (kết quả cũ đã submit-ready trước khi phát hiện lỗi).

> **Cập nhật lần cuối:** 08/07/2026  
> **Git commit gần nhất:** `e7683e1`  
> **Mục đích:** File này ghi lại toàn bộ context để session tiếp theo có thể nắm bắt ngay lập tức.

---

## 1. PROJECT OVERVIEW

**Repository:** `CRP_RL` — Multi-Crane Container Retrieval Problem  
**Research:** Zero-shot transfer of single-crane DRL policy → multi-crane environments  
**Paper target:** Transportation Research Part C (TR-C, Q1, IF ~8.9)  
**Backup venue:** Computers & Operations Research (COR, Q1, IF ~4.5)  

**Key paper:** Shin et al. (2026) — "Learning to Retrieve Containers" (TR-C)  
**Our contribution:** First formal M-CRP definition + zero-shot transfer analysis  

**Trạng thái hiện tại:** Paper hoàn chỉnh, sẵn sàng submit sau khi review nội bộ.

---

## 2. CLAIMS (C1-C5) & TRẠNG THÁI

| Claim | Mô tả | Code | Paper | Data |
|-------|-------|------|-------|------|
| **C1** | M-CRP definition + NP-hardness | `mcenv/mcenv.py` ✅ | Section 3 ✅ | — |
| **C2** | Theorem 3: LB_MCRP | `bounds/lowerbound_mc.py` ✅ | Section 3.3 ✅ | Verified ✅ |
| **C3** | 4 Crane Assignment Strategies | `strategies/*.py` ✅ | Section 4.4 ✅ | All tested ✅ |
| **C4** | Zero-shot empirical evaluation | `experiment.py` ✅ | Section 6 ✅ | 1120 runs ✅ |
| **C5** | Public benchmark dataset | `benchmarks/mc_instances/` ✅ | Data Availability ✅ | 140 instances ✅ |

---

## 3. KEY RESULTS (FINAL)

### Single-crane: 111 Lee instances (comprehensive, 888 runs)

| Method | Mean Gap(%) | So với ZeroShot |
|--------|-------------|-----------------|
| **ZeroShot (ours)** 🏆 | **6.03%** | — |
| Original Model (Shin et al.) | 7.06% | +1.03pp |
| Lin2015 (SOTA heuristic) | 22.42% | **+16.39pp** (3.7× worse) |
| Leveling | 22.78% | +16.75pp |
| Kim2016 | 44.32% | +38.29pp |
| Durasevic2025 (GP) | 51.33% | +45.30pp |
| NearestStack | 78.60% | +72.57pp |
| LowestHeight | 74.74% | +68.71pp |

### Multi-crane: 140 M-CRP instances

| Strategy | C=2 gap(%) | C=3 gap(%) | Interference(C=2) | Interference(C=3) | Cost |
|----------|-----------|-----------|-------------------|-------------------|------|
| S1 RoundRobin | 10.99 | 11.19 | 105.9 | 147.2 | O(1) |
| **S2 ZoneSplit** 🏆 | **10.46** | **10.71** | **0.5** | **0.7** | **O(B)** |
| S3 LoadBalance | 10.99 | 11.19 | 105.9 | 147.2 | O(C) |
| S4 GreedyOptimal | 10.48 | 10.75 | 0.0 | 0.0 | O(C²) |

### Multi-crane: ZeroShot+S2 vs Heuristic Baseline

| Method | C=2 gap(%) | C=3 gap(%) | Improvement |
|--------|-----------|-----------|-------------|
| M-Lin2015 (heuristic) | 31.73 | 33.93 | — |
| **ZS+S2 (ours)** 🏆 | **10.46** | **10.74** | **~3× lower gap** |

---

## 4. EXPERIMENTAL SETUP

| Config | Value |
|--------|-------|
| Hardware | CPU laptop (0 GPU) |
| Total single-crane runs | 50 instances × 8 methods = 400 runs |
| Total multi-crane runs | 140 instances × 2 cranes × 4 strategies = 1,120 runs |
| Total multi-crane baselines | 140 instances × 2 cranes × 2 methods = 560 runs |
| **Grand total** | **~2,080 runs** |
| Single-crane runtime | ~20 min (all baselines on 50 instances) |
| Multi-crane runtime | ~60 min (4 strategies) |
| Multi-crane baselines | ~28 min (ZS+S2 + M-Lin2015) |
| Dataset | 140 M-CRP instances từ Lee benchmark |
| Model | Shin et al. epoch(100) pretrained (public) |

### Kết quả backward compatibility:
- **5/5 instances passed** (threshold 2%)
- Average diff: **1.32%**, Max diff: **1.95%**

---

## 5. KEY FILES

### Documentation
| File | Description |
|------|-------------|
| `docs/latex/crp_rl_paper_Q1.pdf` | **Paper English** (23 trang, 1.5MB, 300 DPI figures) |
| `docs/latex/crp_rl_paper_Q1.tex` | LaTeX source (790 lines) |
| `docs/latex/crp_rl_paper_VI.pdf` | **Paper Vietnamese** (5 trang) |
| `docs/latex/presentation.pdf` | Slides (16 slides) |
| `docs/latex/ULTRA_REVIEW.md` | Professor review report |
| `docs/latex/PROFESSOR_REVIEW.md` | Previous review |
| `docs/superpowers/plans/RESEARCH_PLAN_Q1.md` | Research plan |
| `SESSION_HANDOFF.md` | **THIS FILE** |

### Core Code
| File | Lines | Description |
|------|-------|-------------|
| `mcenv/mcenv.py` | 95 | Multi-crane environment (A6, A7 constraints) |
| `policy/zero_shot.py` | 72 | Zero-shot policy extraction |
| `strategies/*.py` | 110 | S1-S4: RoundRobin, ZoneSplit, LoadBalance, GreedyOptimal |
| `bounds/lowerbound_mc.py` | 90 | Theorem 3: LB_MCRP |
| `engine/mcrp_inference.py` | 50 | M-CRP episode runner |
| `experiment.py` | 160 | Experiment pipeline |
| `run_full_experiment.py` | 155 | Batch experiment (recovery-safe) |
| `compare_all.py` | 142 | Single-crane baseline comparison |
| `analysis/analyze.py` | 235 | Statistical analysis (6 tables) |
| `analysis/visualize.py` | 365 | Figure generation (8 figures) |
| `analysis/supplementary_analysis.py` | 210 | Cost decomposition + case study |
| `analysis/run_comprehensive.py` | 200 | All baselines on all instances |
| `analysis/fix_critical_issues.py` | 220 | Full benchmark + backward compat |

### Results
| File | Description |
|------|-------------|
| `results/mcrp_experiment_20260708_013214.csv` | **Main** — 1120 runs (4 strategies × 2 cranes × 140 instances) |
| `results/final_comprehensive/single_crane_all_baselines.csv` | **Full** — 7 baselines × 50 instances |
| `results/final_comprehensive/multi_crane_all_baselines.csv` | **Full** — ZS+S2 + M-Lin2015 × 140 instances |
| `results/supplementary/multi_crane_baselines.csv` | Supplementary — 9 instances |
| `results/critical_fixes/backward_compat.csv` | Backward compat — 5 instances |
| `results/critical_fixes/full_lee_benchmark.csv` | ZeroShot on 50 instances |
| `results/analysis_report.txt` | Full analysis (6 tables) |
| `results/figures/fig1-8_*.png` | **8 figures** (300 DPI) |

---

## 6. CODE ARCHITECTURE

```
experiment.py / run_full_experiment.py
    ↓
ZeroShotPolicy.get_scores()  ← Frozen pretrained encoder + scorer (policy/zero_shot.py)
    ↓
Strategy.assign()            ← strategies/*.py (S1 RoundRobin / S2 ZoneSplit / S3 LoadBalance / S4 GreedyOptimal)
    ↓
MCEnv.step(dest, crane)     ← Multi-crane environment with A6/A7 constraints (mcenv/mcenv.py)
    ↓
run_mcrp_episode()           ← Episode loop - Algorithm 1 (engine/mcrp_inference.py)
    ↓
Results CSV                  ← Cost, gap, interference, steps
    ↓
MCRPAnalyzer                 ← Statistical analysis (analysis/analyze.py)
    ↓
Visualization                ← 8 figures (analysis/visualize.py)
    ↓
Supplementary Analysis       ← Cost decomposition + Case study + Full benchmark (analysis/supplementary_analysis.py, analysis/run_comprehensive.py)
```

---

## 7. HOW TO RUN (QUICK REFERENCE)

```bash
# 1. Setup
pip install -r requirements.txt

# 2. Tests (36 tests)
python -m pytest tests/ -v

# 3. Generate M-CRP dataset
python benchmarks/generate_mc_instances.py

# 4. Single-crane comparison (4 instances, fast)
python compare_all.py

# 5. Quick M-CRP test (12 runs)
python experiment.py --quick

# 6. Full experiment (1120 runs, ~60 min)
python run_full_experiment.py --batch_size 15

# 7. ALL baselines on ALL instances (~48 min)
python analysis/run_comprehensive.py

# 8. Analysis
python -m analysis.analyze
python -m analysis.visualize
```

---

## 8. PAPER STRUCTURE (23 trang)

| Section | Pages | Content |
|---------|-------|---------|
| Abstract | 1 | Summary with all key numbers |
| 1. Introduction | 1-4 | Motivation, SotA, Gap, Contributions |
| 2. Related Work | 4-6 | CRP methods, Multi-crane ops, Zero-shot transfer |
| 3. Problem Formulation | 6-8 | CRP, M-CRP definition, Theorem 3 LB |
| 4. Methodology | 8-12 | Base DRL, Decoupled architecture, Extraction, 4 strategies, MCEnv |
| 5. Experimental Design | 12-13 | Dataset, Baselines, Metrics |
| 6. Results | 13-17 | Single-crane (Fig5+Table1), M-CRP (Fig1+Table3), Scale (Fig2), Stats (Table4), Baselines (Fig6+Table5), Cost decomposition, Case study, Failure |
| 7. Discussion | 17-18 | Why transfer works, Recommendations, Limitations, Future |
| 8. Conclusion | 18 | Key findings |
| Data Availability | 18 | GitHub link |
| Appendix | 18-19 | Gap by bays (Fig4), Speedup, Computation time |
| References | 19-23 | 17 references |

---

## 9. IMPORTANT NOTES FOR NEXT SESSION

### LaTeX setup
- MiKTeX: `C:\Users\X1\AppData\Local\Programs\MiKTeX\miktex\bin\x64\`
- PATH đã thêm vào `.bashrc`
- VS Code extension: LaTeX Workshop (James-Yu.latex-workshop)
- Compile: `pdflatex -interaction=nonstopmode file.tex && pdflatex -interaction=nonstopmode file.tex`

### Paper writing tips
- Dùng elsarticle class (Elsevier TR-C format)
- Figures cần ≥300 DPI
- Abstract ≤300 words

### Key insights to remember
- ZoneSplit (S2) là chiến lược tốt nhất: gap thấp nhất, ~0 interference, O(B)
- Zero-shot transfer hoạt động tốt: gap 10.5% không cần retrain
- Heuristic baselines đánh giá thấp hơn nhiều trên full benchmark (Lin2015: 10.28% trên 4 instances → 22.42% trên 50 instances)
- Sequential simulation là limitation chính (speedup C2→C3 chỉ 1.01×)

---

## 10. PENDING ITEMS

### Trước khi submit TR-C
- [ ] Thay "Anonymous Author(s)" bằng tên thật + email + affiliation
- [ ] Kiểm tra plagiarism (iThenticate/Turnitin)
- [ ] Kiểm tra format TR-C guidelines
- [ ] Generate supplementary material (appendix with all raw data tables)
- [ ] Xem xét reviewer comments từ internal review

### Có thể bổ sung (optional)
- [ ] Chạy thêm sensitivity analysis (pOMO size, tanh clipping ảnh hưởng?)
- [ ] So sánh với multi-agent PPO (cần GPU)

---

## 11. ALL GIT COMMITS

```
e7683e1 (HEAD) Ultra review fixes: shortened SotA, fixed wording, higher DPI
3107976 Add 2 new figures, 8 figures total
63616ef Complete: all baselines, all instances, full benchmarks
bc2e101 Fix critical issues: 50-instance benchmark, case study, backward compat
66fc0db Supplementary analyses: cost decomposition, multi-crane baselines
...
```
