# ULTRA REVIEW — Senior Researcher Assessment

## Paper: Zero-shot Transfer of Single-Crane DRL Policies for M-CRP

---

## 1. CONSISTENCY CHECK: Claims vs Evidence

| Claim in Abstract/Intro | Evidence in Paper | Match? |
|------------------------|-------------------|--------|
| ZeroShot 6.56% on 50 instances | Table 1 (50-instance baseline) | ✅ |
| Lin2015 22.42% | Table 1 | ✅ |
| ZS+S2 C=2 gap 10.46% | Table 3 | ✅ |
| ZS+S2 C=3 gap 10.71% | Table 3 | ✅ |
| Interference reduction >99.5% | 105.9→0.5 = 99.53% | ✅ |
| 19-113pp improvement vs baselines | Table 1: 119.71-6.56=113.15pp | ✅ |
| "3× better" than M-Lin2015 | 31.73/10.46=3.03× | ✅ * |

*Wording should be "approximately one-third the gap" not "3× better"

---

## 2. ISSUES FOUND (Critical first)

### 🔴 Issue 1: Redundant SotA sections
**Location:** Sec 1.2 "State of the Art" + Sec 2 "Related Work"
**Problem:** Both cover the same literature. Creates redundancy.
**Fix:** Shorten 1.2 to 1 paragraph, keep full Related Work in Section 2.

### 🔴 Issue 2: Naive baseline (LB/C) misleading  
**Location:** Table 3 (main results)
**Problem:** Gap of 90-122% for "Naive baseline" is not a fair comparison. 
**Fix:** Remove or add explicit caveat that LB/C is not a valid lower bound.

### 🟡 Issue 3: "3× better" imprecise wording
**Location:** Section 6.3
**Problem:** "3× better" could be misinterpreted. 
**Fix:** "achieves approximately one-third the gap of M-Lin2015 (10.46% vs 31.73%)"

### 🟡 Issue 4: Figure DPI too low
**Location:** All figures
**Problem:** 150 DPI (TR-C requires ≥300 DPI)
**Fix:** Change in visualize.py: `figure.dpi: 300`

### 🟡 Issue 5: ZeroShot 6.56% vs Shin et al. 7.8%
**Location:** Section 6.1
**Problem:** ZeroShot actually outperforms the reported Shin et al. baseline but this isn't highlighted.
**Fix:** Add note: "Our 6.56% gap on the full 50-instance benchmark compares favorably with the reported 7.8% in Shin et al. [2026]"

### 🟢 Issue 6: Missing computational complexity analysis
**Location:** Section 4
**Problem:** Paper mentions O(B), O(C²) for strategies but doesn't analyze total system cost including DRL inference.
**Fix:** Add brief note: "Total per-step inference: DRL encoder+scorer ~5ms + strategy O(B) ~0.01ms = ~5ms"

---

## 3. CORRECTIONS NEEDED

### Fix priorities:

| # | Priority | Fix | Effort |
|---|----------|-----|--------|
| 1 | 🔴 | Shorten Sec 1.2, merge into Sec 2 | 10 min |
| 2 | 🔴 | Remove naive baseline or add caveat | 5 min |
| 3 | 🟡 | Fix "3×" wording | 2 min |
| 4 | 🟡 | Increase figure DPI to 300 | 2 min |
| 5 | 🟡 | Add ZeroShot vs Shin et al. note | 2 min |
| 6 | 🟢 | Add complexity analysis note | 5 min |

---

## 4. VERDICT

**Overall readiness: 85%** — Paper is solid. 6 minor-to-moderate fixes needed before submission.

**Strongest aspects:**
- First M-CRP formalization (clear C1 contribution)
- Comprehensive experiments (50 instances single-crane + 140 M-CRP + 1120 runs)
- Strong empirical evidence (3× better than heuristic baselines)
- Clear practical recommendation (ZoneSplit)

**Weakest aspects:**
- Redundant literature review (SotA in Intro + Related Work)
- Figure quality (needs higher DPI)
- Some imprecise wording ("3× better")

**Recommendation:** Fix 6 issues above (~25 min), then submit to TR-C.
