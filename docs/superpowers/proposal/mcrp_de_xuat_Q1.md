# M-CRP: Multi-Crane Container Retrieval Problem — Phân tích Zero-shot Transfer của Single-Crane DRL Policy

## 1. Vấn đề và động lực

**Container Retrieval Problem (CRP):** Tối ưu hóa thời gian làm việc của yard crane khi lấy container theo thứ tự từ bãi chứa (B bays × R rows × T tiers), NP-hard.

**Khoảng trống:** Toàn bộ nghiên cứu CRP hiện tại — gồm cả Shin et al. (2026, *Transportation Research Part C*, Q1) — chỉ giải bài toán **single-crane**. Trong thực tế, terminal tự động vận hành 2-3 cần cẩu trên cùng một yard block. Chưa có công trình nào định nghĩa hoặc phân tích multi-crane CRP (M-CRP).

**Câu hỏi nghiên cứu:** Một DRL policy được huấn luyện chỉ trên single-crane instances (Shin et al. 2026) có thể zero-shot transfer sang môi trường multi-crane thông qua heuristic crane assignment strategies hay không?

---

## 2. Đóng góp khoa học (Q1-grounded)

### C1: Định nghĩa M-CRP + NP-hardness proof (Lý thuyết mới — novelty cao)

**Nội dung:** Định nghĩa hình thức đầu tiên của M-CRP:
- State space: yard config (B×R×T) + crane positions (C cranes)
- Action space: (dest_stack, crane_id) — không chỉ stack như CRP gốc
- Constraints: (A6) spatial non-overlap: |bay_c − bay_c'| ≥ 1; (A7) non-crossing: thứ tự cần cẩu trên trục bay không đổi
- Objective: minimize Σ_c (travel_c + handling_c) + interference penalty
- NP-hardness: CRP với C=1 là NP-hard (Shin et al. 2026 Theorem 1), do đó M-CRP với C≥1 là NP-hard

> **Tại sao đủ Q1:** Định nghĩa bài toán mới có ý nghĩa thực tiễn cao. Thiết lập nền tảng lý thuyết cho toàn bộ nghiên cứu M-CRP sau này.

### C2: Lower bound M-CRP — Theorem 3 (Lý thuyết mới — bound chặt hơn)

**Nội dung:** Mở rộng Theorem 2 (Shin et al.) lên multi-crane:

**LB_MCRP = LB_retrieval + LB_relocation/C + LB_interference**

Trong đó LB_interference capture penalty khi relocation demand tập trung không đều vào một bay, vượt quá năng lực per-crane. Khi C=1, LB_MCRP = LB từ Theorem 2 (backward compatible).

> **Tại sao đủ Q1:** Provable lower bound cho phép đánh giá solution quality một cách khách quan. Cộng đồng CRP đang thiếu bounds cho multi-crane variant.

### C3: Bốn Crane Assignment Strategies (Phương pháp mới — systematic design)

| Strategy | Cơ chế | Độ phức tạp |
|----------|--------|-------------|
| S1 RoundRobin | Gán vòng tròn | O(1) |
| S2 ZoneSplit | Chia bay tĩnh theo vùng | O(B) |
| S3 LoadBalance | Cân bằng tải theo hàng đợi | O(C·log C) |
| S4 GreedyOptimal | 1-step lookahead cost min | O(C·B·R) |

> **Tại sao đủ Q1:** Thiết kế có hệ thống 4 strategies từ đơn giản đến phức tạp, bao phủ toàn bộ không gian thiết kế. So sánh có paired statistical tests.

### C4: Zero-shot Transfer Evaluation (Hiểu biết-phân tích mới — first empirical study)

**Nội dung:** Đánh giá pretrained DRL model của Shin et al. (public, verified gap ~7.8%) trên **120 instances M-CRP** (2-3 cranes, 70-2880 containers). So sánh 4 strategies. Phân tích failure modes, interference patterns, bottleneck bays.

> **Tại sao đủ Q1:** Zero-shot transfer của DRL policy sang multi-crane environment chưa ai nghiên cứu. Lượng hóa khi nào transfer thành công, khi nào thất bại, và tại sao.

### C5: M-CRP Public Benchmark Dataset (Benchmark mới — practical contribution)

**Nội dung:** 120 instances mở rộng từ Lee & Lee (2010) với 2-3 crane configs, 3 scale groups (small/medium/large), cả R-type và U-type. Public release CC BY 4.0.

> **Tại sao đủ Q1:** Thiết lập evaluation standard cho M-CRP research sau này. Không có benchmark = không có reproducible progress.

---

## 3. Phương pháp

### Kiến trúc

```
┌─────────────┐    state (B×R×T)    ┌──────────────────┐
│  MCEnv       │ ◄───────────────── │  Pretrained       │
│  (C cranes)  │                    │  Encoder (frozen) │
│  interference│                    │  + Scorer weights │
│  tracking    │                    │  (ZeroShotPolicy) │
└──────┬──────┘                    └────────┬─────────┘
       │                                    │ dest_stack
       │ crane_id                           ▼
       │                           ┌──────────────────┐
       └──────────────────────────►│ Crane Assignment │
                                   │ Strategy (S1-S4) │
                                   └────────┬─────────┘
                                            │ (dest, crane)
                                            ▼
                                   ┌──────────────────┐
                                   │  MCEnv.step()    │
                                   │  → cost + state  │
                                   └──────────────────┘
```

### Zero-shot inference per step
1. Encode yard state qua frozen pretrained encoder (LSTM + attention)
2. Score stacks qua original decoder's scorer (W_Q, W_K, W_V, MHA weights)
3. Chọn `dest_stack` = greedy argmax over feasible stacks
4. Strategy chọn `crane_id` ∈ {1..C}
5. MCEnv validate interference, update state, return cost

### Key decision: Decoupled inference
- Original `Model.forward()` tightly coupled với single-crane `Env`. Không thể gọi model() cho multi-crane.
- `ZeroShotPolicy` class extract encoder + scorer weights → expose `get_action()` cho external environment (MCEnv).

---

## 4. Thiết kế thực nghiệm

### Datasets
| Group | Bays | Rows | Tiers | Containers | Instances (×2 crane counts) |
|-------|------|------|-------|------------|-----------------------------|
| Small | 1-4 | 16 | 6-8 | 70-380 | 40 |
| Medium | 6-10 | 16 | 6-8 | 430-720 | 40 |
| Large | 20-30 | 16 | 6-8 | 1440-2880 | 40 |

### Baseline
- Model: Shin et al. (2026) epoch(100) — verified gap ≈ 7.8% trên R-type Lee benchmark
- Single-crane (C=1) sanity check: zero-shot pipeline cost phải match original model cost trong ±2%

### Metrics
- **Primary:** `Gap(LB_MCRP)% = 100 × (CTWT − LB_MCRP) / LB_MCRP`
- **Secondary:** Interference count, per-crane utilization, solution time, failure mode frequency

### Ablation
1. By strategy: S1 vs S2 vs S3 vs S4
2. By crane count: C=2 vs C=3
3. By yard scale: small vs medium vs large
4. By instance type: random (R) vs upside-down (U)
5. Backward compatibility: C=1 must match original model cost

### Statistical testing
- Pairwise Wilcoxon signed-rank test (p < 0.05)
- Cohen's d effect size between top-2 strategies
- Failure mode analysis: identify instances with gap > 20%, characterize by bottleneck bay

---

## 5. Tài nguyên

| Thành phần | Compute | Thời gian (laptop) |
|-----------|---------|-------------------|
| Code development | CPU | 3-5 ngày |
| Sinh instances | CPU | <1 phút |
| Verify baseline (36 tests) | CPU | ~2.5 phút |
| Full experiment (120 × 2 × 4 = 960 runs) | CPU | ~30-60 phút |
| Analysis + statistics | CPU | <30 giây |
| **Tổng** | **0 GPU hours** | **4-7 ngày** |

Toàn bộ chạy CPU laptop. Zero-shot inference nhanh vì pretrained model dùng greedy decoding (1 trajectory, không sampling).

---

## 6. Cách chạy

```bash
# 1. Sinh dataset
python benchmarks/generate_mc_instances.py

# 2. Verify code (36 tests)
python -m pytest tests/ -v

# 3. Quick test: 3 instances x 2 cranes x 2 strategies (12 runs)
python experiment.py --quick

# 4. Full experiment: 120 instances x 2-3 cranes x 4 strategies (960-1440 runs)
python experiment.py --cranes 2 3 --strategies S1 S2 S3 S4

# 5. Analyze results
python analysis/run_analysis.py
```

---

## 7. Q1 Venue & Tính khả thi

### Target chính: Transportation Research Part C (TR-C, IF ~8-9)
- **Scope match:** "Emerging technologies" — zero-shot DRL transfer cho multi-crane terminal
- **Cùng venue với baseline:** Shin et al. 2026 — editorial familiarity, continuity
- **Practical relevance:** Real terminal vận hành multi-crane, method CPU-inference deployable

### Secondary: Computers & Operations Research (COR, Q1, IF ~4-5)
- Chấp nhận benchmark + empirical comparison papers
- Kể cả null results (negative findings về zero-shot transfer) vẫn publish được

### Rủi ro reject và cách phòng tránh

| Rủi ro | Cách phòng tránh |
|--------|-----------------|
| "Incremental contribution" | C1+C2 là lý thuyết mới (M-CRP formulation + bound). C4 là phân tích failure modes không quan sát được trong single-crane. |
| "Không có method innovation" | Decoupled zero-shot inference architecture + transferability analysis là mới. Backup: C5 benchmark standalone. |
| "Baselines yếu" | 4 strategies bao phủ toàn bộ không gian thiết kế. Paired statistical tests. |
| "Không train trên real data" | Lee & Lee benchmark là standard trong CRP literature. Model verified reproduce paper results. |

### Tóm tắt novelty Q1

| Claim | Loại | Novelty (1-5) | Q1-ready? |
|-------|------|--------------|-----------|
| C1: M-CRP formulation | Lý thuyết | 5/5 | ✅ Chưa ai làm |
| C2: LB_MCRP Theorem 3 | Lý thuyết | 4/5 | ✅ Extension có proof |
| C3: 4 strategies | Phương pháp | 3/5 | ✅ Systematic comparison |
| C4: Zero-shot empirical | Phân tích | 4/5 | ✅ First study |
| C5: Benchmark | Dataset | 4/5 | ✅ Public release |

**Kết luận:** 4/5 claims ở mức ≥ 4/5 novelty. Đủ cho TR-C hoặc COR.
