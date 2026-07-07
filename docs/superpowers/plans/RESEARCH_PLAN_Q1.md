# Research Plan: Zero-shot Transfer for M-CRP
## Target: Transportation Research Part C (TR-C, IF ~8.9)
## Backup: Computers & Operations Research (COR, IF ~4.5)

---

## 1. Paper Map (Story)

### Title (draft)
> *"Zero-shot Transfer of Single-Crane Deep Reinforcement Learning Policies for Multi-Crane Container Retrieval"*

### Claims → Evidence → Paper Section

| Claim | Evidence | Paper Section | Code Location |
|-------|----------|---------------|---------------|
| C1: M-CRP định nghĩa + NP-hard | Formal definition, Theorem 1 proof | Section 2 | `mcenv/mcenv.py` (implementation), paper Section 2 |
| C2: LB_MCRP Theorem 3 | Formula + backward compatible C=1 | Section 3 | `bounds/lowerbound_mc.py` |
| C3: 4 Strategies | S1 RoundRobin, S2 ZoneSplit, S3 LoadBalance, S4 GreedyOptimal | Section 4 | `strategies/*.py` |
| C4: Zero-shot transfer | Single-crane: gap 5.63% (beats Lin2015 10.28%), Multi-crane: gap 10.8-12.8% | Section 5 | `experiment.py`, `compare_all.py` |
| C4a: Multi-crane baselines | Extended heuristics comparison | Section 5.3 | `baselines/multi_crane/` |
| C5: Public benchmark | 160 instances, CC BY 4.0 | Section 6 | `benchmarks/generate_mc_instances.py` |

### Target contribution statement for TR-C review
> *"This paper presents the first formal definition and analysis of the Multi-Crane Container Retrieval Problem (M-CRP). We demonstrate that a single-crane DRL policy can be transferred zero-shot to multi-crane environments through heuristic crane assignment strategies, achieving 5.63% average gap on single-crane benchmarks and 10.8-12.8% on multi-crane variants."*

---

## 2. Implementation Tasks

### Phase 1: Multi-crane Baselines (Priority: HIGH)
**Goal:** Extend single-crane heuristics to multi-crane for fair comparison.

| Baseline | Single-crane | Multi-crane extension | Effort | File |
|----------|-------------|----------------------|--------|------|
| Lin2015 | ✅ có sẵn | Wrap với ZoneSplit strategy | 1 day | `baselines/multi_crane/lin2015_mc.py` |
| Kim2016 | ✅ có sẵn | Wrap với ZoneSplit strategy | 1 day | `baselines/multi_crane/kim2016_mc.py` |
| Leveling | ✅ có sẵn | Wrap với LoadBalance strategy | 0.5 day | `baselines/multi_crane/leveling_mc.py` |
| Durasevic2025 | ✅ có sẵn | Wrap với greedy assignment | 0.5 day | `baselines/multi_crane/durasevic2025_mc.py` |

**Interface mới:** Multi-crane baseline wrapper:
```python
class MultiCraneBaseline:
    def __init__(self, single_crane_baseline, strategy):
        self.bl = single_crane_baseline
        self.strategy = strategy
    
    def run(self, x_4d, n_cranes):
        # 1. Gọi single-crane baseline → quyết định dest_stack
        # 2. Strategy gán crane_id
        # 3. MCEnv execute
        pass
```

### Phase 2: Full Experiment (Priority: HIGH)
**Config:** 120 instances × 2 crane counts × 4 strategies = 960 runs

```bash
python experiment.py --cranes 2 3 --strategies S1 S2 S3 S4
```

Expected runtime: ~30-60 phút (CPU laptop).

**Output:** `results/mcrp_experiment_{timestamp}.csv`

### Phase 3: Statistical Analysis (Priority: HIGH)

| Test | Purpose | Code | 
|------|---------|------|
| Wilcoxon signed-rank | So sánh paired strategies | `analysis/analyze.py:table2_pairwise_wilcoxon()` |
| Cohen's d | Effect size top-2 strategies | `analysis/analyze.py:compute_effect_size()` |
| Failure mode | Instances gap >20% | `analysis/analyze.py:identify_failure_modes()` |

### Phase 4: Multi-crane Baseline Comparison (Priority: MEDIUM)
Run multi-crane baselines vs zero-shot on same 120 instances:
```bash
python compare_mc_baselines.py
```

### Phase 5: Speedup Analysis (Priority: LOW)
Compute speedup factor: `cost(C=1) / cost(C=n)` cho mỗi instance.

---

## 3. Paper Traceability Matrix

Mỗi số liệu trong paper phải trace được đến code và data.

### Table 1: Single-crane Baseline Comparison
| Cell | Source | Command |
|------|--------|---------|
| ZeroShot gap 5.63% | `compare_all.py` output | `python compare_all.py` |
| Lin2015 gap 10.28% | `compare_all.py` output | `python compare_all.py` |
| Kim2016 gap 25.02% | `compare_all.py` output | `python compare_all.py` |

### Table 2: M-CRP Strategy Comparison
| Cell | Source | Command |
|------|--------|---------|
| Gap per (scale, C, strategy) | `experiment.py` output CSV | `python experiment.py --cranes 2 3 --strategies S1 S2 S3 S4` |
| Mean, std, min, max | `analysis/analyze.py:table1_gap_comparison()` | `python analysis/run_analysis.py` |

### Table 3: Pairwise Wilcoxon
| Cell | Source | Command |
|------|--------|---------|
| p-value per (C, strategy pair) | `analysis/analyze.py:table2_pairwise_wilcoxon()` | `python analysis/run_analysis.py` |

### Table 4: Multi-crane Baselines
| Cell | Source | Command |
|------|--------|---------|
| M-Lin2015, M-Kim2016 gaps | `compare_mc_baselines.py` output | `python compare_mc_baselines.py` |

### Figure 1: Gap by Scale
| Data | Source | Command |
|------|--------|---------|
| Gap bars per (scale, C, strategy) | `analysis/analyze.py:table4_cost_by_scale()` | `python analysis/run_analysis.py` |

---

## 4. Reviewer Response Preparation

### Predicted Reviewer Questions + Pre-prepared Answers

| Câu hỏi dự kiến | Câu trả lời dự trù | Evidence |
|-----------------|-------------------|----------|
| "Why only 4 strategies?" | "4 strategies span design space from simple (RoundRobin) to complex (GreedyOptimal). This is systematic." | Paper Section 4 |
| "Why no multi-crane training?" | "Zero-shot transfer is the research question. Fine-tuning is future work." | Paper Section 7 |
| "Why not train from scratch?" | "Our goal is to test transferability of single-crane knowledge." | Paper Section 1 |
| "Results on 1 bay are meaningless" | "Agreed. Analysis focuses on multi-bay (B ≥ C) configurations." | Paper Section 5.3 |
| "Gap 10-13% is high" | "Zero-shot without retraining. The question is whether transfer works at all. It does." | Paper Section 5.4 |
| "Why no real-world validation?" | "Lee & Lee benchmark is standard in CRP literature. Public code for reproducibility." | Paper Section 6 |
| "Statistical significance?" | "Wilcoxon signed-rank test p < 0.05 on all strategy comparisons." | Table 3 |

### Risk Matrix

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| "Incremental contribution" (reject) | Medium | High | Cấu trúc paper tập trung vào C1 (formulation) + C4 (first study). Nhấn mạnh novelty. |
| Zero-shot gap ≥ 15% | Low | Medium | Phân tích failure mode, chỉ ra khi nào transfer thất bại. |
| TR-C desk reject | Medium | High | Chuẩn bị COR submission song song. |
| Reviewer yêu cầu baselines mạnh hơn | Medium | Medium | Phase 1: multi-crane baselines đã sẵn sàng. |

---

## 5. Timeline

| Tuần | Task | Deliverable |
|------|------|-------------|
| **1** | Phase 1: Multi-crane baselines code | `baselines/multi_crane/*.py` |
| **1** | Phase 2: Full experiment | `results/mcrp_experiment_*.csv` |
| **1** | Phase 3: Statistical analysis | `results/analysis_*.csv`, figures |
| **2** | Phase 4: Baseline comparison | `results/mc_baseline_comparison.csv` |
| **2** | Viết paper draft | `docs/latex/crp_rl_paper.tex` → PDF |
| **2** | Internal review + sửa | Paper draft v2 |
| **3** | Submit TR-C | Paper + supplementary material |

---

## 6. Documentation Structure

```
docs/
  latex/
    crp_rl_paper.tex                # Paper draft (TR-C format)
    REVIEW_REPORT.md                 # Code review report
  superpowers/
    proposal/
      mcrp_de_xuat_Q1.md             # Proposal (Vietnamese)
      mcrp_zero_shot_proposal.md      # Proposal (English)
      phuong_phap_chi_tiet.md         # Detailed method
    plans/
      RESEARCH_PLAN_Q1.md             # THIS FILE
    specs/                           # Design specs (future)
results/
  mcrp_experiment_*.csv              # Raw experiment results
  baseline_comparison.csv             # Single-crane baseline results
  mc_baseline_comparison.csv          # Multi-crane baseline results
  paper_verification.csv             # Paper result verification
analysis/                            # Analysis code
tests/                               # 36 tests covering all claims
```

---

## 7. Critical Path

```
Tuần 1                    Tuần 2                    Tuần 3
│                         │                         │
├─ Multi-crane baselines  │                         │
├─ Full experiment ───────┤                         │
│                         ├─ Paper draft ───────────┤
│                         ├─ Internal review         │
│                         │                         ├─ Submit
│                         │                         │
└─ Statistical analysis ──┘                         │
                          └─ Rebuttal prep ──────────┘
```
