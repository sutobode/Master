# 📋 ULTRA REVIEW REPORT: M-CRP Zero-shot Transfer

## Tổng quan: 5 Claims

| Claim | Code | Paper | Tests | Severity |
|-------|------|-------|-------|----------|
| C1: M-CRP Definition + NP-hardness | ⚠️ | ✅ | ⚠️ | **HIGH** |
| C2: Theorem 3 Lower Bound | ✅ | ⚠️ | ⚠️ | **MEDIUM** |
| C3: 4 Strategies | ⚠️ | ❌ | ✅ | **HIGH** |
| C4: Zero-shot Pipeline | ✅ | ❌ | ✅ | **CRITICAL** |
| C5: Benchmark Dataset | ⚠️ | ✅ | ✅ | **MEDIUM** |

---

## 1️⃣ C1: M-CRP Definition + NP-hardness — HIGH SEVERITY

### ✅ Code: MCEnv (`mcenv/mcenv.py`)
- State space: ✅ yard config + crane positions
- Action space: ✅ (dest_stack, crane_id)
- Interference constraints A6: ✅ `_validate_interference()`
- Objective: ✅ cost tracking per crane

### ❌ ISSUE 1: Interference Resolution Bug
**File:** `mcenv/mcenv.py:50-54`
```python
def _resolve_interference(self, crane_id, dest_bay):
    for c in range(self.n_cranes):
        if c != crane_id and self._validate_interference(c, dest_bay):
            return c  # <-- Returns first free crane, not necessarily optimal
    return crane_id  # <-- Falls back to original (still interfering!)
```
**Fix needed:** When interference detected, strategy should re-assign. Current fallback to `crane_id` still violates A6. Đây là bug logic.

### ❌ ISSUE 2: NP-hardness proof not in paper
**File:** Paper Theorem 1 proof
**Problem:** Proof is only 2 sentences. For Q1 review, need formal reduction from BRP.
**Fix:** Mở rộng proof: (1) BRP → CRP reduction, (2) CRP → M-CRP reduction, (3) preserve solution cost.

### ❌ ISSUE 3: No non-crossing constraint (A7) enforcement
**Code:** `mcenv.py` không có check A7 nào.
**Fix:** Thêm `_validate_non_crossing()` nếu A7 được claim.

---

## 2️⃣ C2: Theorem 3 Lower Bound — MEDIUM SEVERITY

### ✅ Code: `bounds/lowerbound_mc.py`
- LB_retrieval: ✅ extracted from single-crane LB
- LB_relocation/C: ✅ divided by cranes
- LB_interference: ✅ max(0, max_per_bay - ideal_per_crane) penalty

### ⚠️ ISSUE 1: Backward compatibility test tolerance too loose
**File:** `tests/test_lowerbound_mc.py:13`
```python
assert abs(lb_mc.item() - lb_t2) / lb_t2 < 0.1  # 10% tolerance!
```
**Fix:** Tolerance phải ≤ 2% (same as backward compatibility criteria)

### ⚠️ ISSUE 2: `_count_mandatory_relocations` may disagree with `get_wt_lb`'s internal count
**File:** `bounds/lowerbound_mc.py:64-81`
**Problem:** `get_wt_lb` computes mandatory relocations internally. Code recomputes them separately. If the two implementations differ, `lb_retrieval = lb_single - lb_reloc_total` could be wrong.
**Fix:** Refactor `get_wt_lb` to expose `n_mandatory_relocs`.

### ❌ ISSUE 3: Paper formula missing in paper
**File:** Paper shows `LB_MCRP = LB_retrieval + LB_relocation/C + LB_interference` but code also needs to compute `n_mandatory_relocs × (2×t_row + t_pd)`.
**Fix:** Add explicit formula for `LB_interference` with formal definition.

---

## 3️⃣ C3: 4 Strategies — HIGH SEVERITY

### ❌ ISSUE 1 (CRITICAL): GreedyOptimal ≠ 1-step lookahead
**File:** `strategies/greedy_optimal.py`
**Paper claims:** "1-step lookahead cost minimization O(C·B·R)"
**Code reality:** Heuristic cost with zone bonus — NOT 1-step lookahead
```python
cost = travel_cost + interference_penalty - zone_bonus  # heuristic function
```
Không có simulation nào được thực hiện. Complexity thực tế: O(C²) chứ không phải O(C·B·R).
**Fix 1:** Sửa paper description thành "Cost-based heuristic with zone affinity" và complexity thành O(C²)
**Fix 2 (recommended):** Implement true 1-step lookahead:
```python
for c in range(n_cranes):
    simulated_cost = simulate_step(env, dest_stack, c)
    if simulated_cost < best_cost: ...
```

### ⚠️ ISSUE 2: ZoneSplit + GreedyOptimal code duplication
**File:** `zone_split.py` và `greedy_optimal.py` có cùng zone initialization code.
**Fix:** Factor shared logic vào base class.

### ⚠️ ISSUE 3: LoadBalance không tính đến interference
**File:** `strategies/load_balance.py`
**Problem:** Chỉ đếm số tasks, không xét vị trí crane. Crane ở bay 1 có thể được gán task ở bay 10.
**Fix:** Thêm travel cost weight vào balancing decision.

---

## 4️⃣ C4: Zero-shot Pipeline — CRITICAL SEVERITY

### ❌ ISSUE 1 (CRITICAL): Paper chứa kết quả chưa có thực nghiệm
**File:** `docs/latex/crp_rl_paper.tex`
**Problem:** Paper có sẵn bảng kết quả với số liệu cụ thể (gap 5.70%, S4 gap 11.5%, etc.)
**Reality:** Bạn chưa chạy thực nghiệm nào. Reviewer sẽ reject ngay nếu phát hiện số liệu không có thật.
**Fix:** Thay tất cả số liệu bằng **"TBD"** (To Be Determined). Chỉ chèn số thật sau khi chạy.

### ❌ ISSUE 2: `max_steps=2000` không đủ cho large instances
**File:** `engine/mcrp_inference.py:4`
```python
def run_mcrp_episode(..., max_steps=2000):
```
Với 2880 containers, cần ít nhất 4000+ steps (relocations + retrievals).
**Fix:** Tính max_steps dynamic: `max_steps = n_stacks * n_tiers * 2`

### ❌ ISSUE 3: MCEnv batch dim hardcoded = 1
**File:** `mcenv/mcenv.py:46,59,72`
**Problem:** Assumes `batch=1` throughout.
```python
self.crane_bays[0, c]  # hardcoded batch=1
```
**Fix:** Generalize to arbitrary batch size (or document as limitation).

### ❌ ISSUE 4: `compare_all.py` test set quá nhỏ
**File:** `compare_all.py:75`
```python
for idx in range(1, 3):  # Only 2 instances!
```
So sánh 7 baselines trên **4 instances** là không đủ cho Q1.
**Fix:** Chạy ít nhất 10-20 instances, tất cả configs.

### ❌ ISSUE 5: No single-crane backward compatibility test in experiment.py
**File:** `experiment.py`
Không có verification step trước khi chạy multi-crane.
**Fix:** Add C=1 sanity check đầu tiên.

---

## 5️⃣ C5: Benchmark Dataset — MEDIUM SEVERITY

### ⚠️ ISSUE 1: Instance generation depends on `find_and_process_file`
**File:** `benchmarks/generate_mc_instances.py:32`
**Problem:** Nếu `find_and_process_file` fail hoặc thay đổi format, generation sẽ sai.
**Fix:** Add isolated parser không phụ thuộc vào original benchmark code.

### ⚠️ ISSUE 2: Not all scale combinations generated
**File:** `benchmarks/generate_mc_instances.py:26-27`
```python
if tier == 8 and bay in [8, 10]:
    continue
```
Nhiều tổ hợp bị skip. Tổng instances có thể < 160.
**Fix:** Document exact count và justification.

---

## 📊 Tổng hợp các Fix cần làm

### TRƯỚC KHI CHẠY EXPERIMENT (PRIORITY CRITICAL)

| # | Fix | File | Impact |
|---|-----|------|--------|
| 1 | **Xóa kết quả giả khỏi paper** | `crp_rl_paper.tex` | **Blocking** |
| 2 | GreedyOptimal rename/sửa description | Paper + code | **Blocking** |
| 3 | Max_steps cho large instances | `engine/mcrp_inference.py` | Bug |
| 4 | Interference resolution bug | `mcenv/mcenv.py:50` | Bug |
| 5 | Tolerance LB backward compat | `tests/test_lowerbound_mc.py:13` | Test |

### SAU KHI CHẠY EXPERIMENT

| # | Fix | File |
|---|-----|------|
| 6 | Chèn kết quả thật vào paper | `crp_rl_paper.tex` |
| 7 | Chạy compare_all.py đầy đủ | `compare_all.py` |
| 8 | Chạy full experiment 960 runs | `experiment.py` |
| 9 | Verify backward compatibility | C=1 check |

### PAPER FORMAT ISSUES

| # | Fix | Section |
|---|-----|---------|
| 10 | Mở rộng NP-hardness proof | Theorem 1 |
| 11 | Thêm formulas cho LB_interference | Theorem 3 |
| 12 | Sửa complexity GreedyOptimal | Table Strategies |

---

## ✅ ĐÃ ĐÚNG: Những phần không cần sửa

- **ZeroShotPolicy**: Encoder/scorer extraction ✅ đúng kiến trúc
- **Encoder forward**: Tham số khớp original model ✅
- **4 strategies interface**: Polymorphic base class ✅
- **Test coverage**: 9 test files, 36 tests ✅
- **MCEnv single-crane match**: Backward compatible ✅
- **LaTeX compilation**: PDF tạo thành công ✅
