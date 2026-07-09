# KẾ HOẠCH THỰC NGHIỆM TOÀN DIỆN
## Zero-shot Transfer of Single-Crane DRL for M-CRP

---

## 1. DANH SÁCH DATASETS (371 files tổng cộng)

### Single-crane Datasets (231 files)

| Dataset | Random | Upside-down | Tổng | Scale | Ghi chú |
|---------|--------|-------------|------|-------|---------|
| **Lee_instances** | 51 files | 20 files | **71** | 1-10 bays, 6-8 tiers | Standard benchmark |
| **Shin_instances** | 80 files | 80 files | **160** | 20-30 bays, 6-8 tiers | Large-scale |
| **Tổng single** | **131** | **100** | **231** | | |

### Multi-crane Dataset (140 files)

| Dataset | R-type | U-type | Tổng | Cranes |
|---------|--------|--------|------|--------|
| **mc_instances/lee_mc** | 70 | 70 | **140** | 2, 3 |

---

## 2. DANH SÁCH BASELINES + SOTA (10 methods)

| ID | Method | File | Loại | Tốc độ | Notes |
|----|--------|------|------|--------|-------|
| **M0** | **Original Model** (Shin et al. 2026) | `model/model.py` + `epoch(100).pt` | **SOTA DRL** | ~0.5s/inst | Paper gốc, pretrained |
| **M1** | **ZeroShot** (ours) | `policy/zero_shot.py` | **Proposed** | ~0.5s/inst | Phương pháp đề xuất |
| M2 | Lin2015 | `baselines/lin2015.py` | Heuristic SSI | ~0.3s/inst | SOTA heuristic |
| M3 | Kim2016 | `baselines/kim2016.py` | Heuristic class | ~0.3s/inst | |
| M4 | Leveling | `baselines/leveling.py` | Heuristic | ~0.3s/inst | |
| M5 | Durasevic2025 | `baselines/durasevic2025.py` | GP-evolved | ~0.5-140s/inst | Chậm trên instance lớn |
| M6 | NearestStack | `baselines/simple_baselines.py` | Simple | ~0.2s/inst | |
| M7 | LowestHeight | `baselines/simple_baselines.py` | Simple | ~0.2s/inst | |
| M8 | BeamSearchCRP | `baselines/advanced_baselines.py` | Search | ~35s-30ph/inst | Rất chậm |
| M9 | RolloutGP | `baselines/advanced_baselines.py` | Search+GP | ~140s-2h/inst | ❌ Skip (quá chậm) |

---

## 3. TRẠNG THÁI HIỆN TẠI (Đã chạy 2,415 runs)

| Dataset | Methods | Runs | File |
|---------|---------|------|------|
| Lee_random (71) | M0, M1, M2, M3, M4, M5, M6, M7 | 568 | `results/comprehensive/Lee_random.csv` |
| Lee_upsidedown (40) | M0, M1, M2, M3, M4, M5, M6, M7 | 320 | `results/comprehensive/Lee_upsidedown.csv` |
| M-CRP (140 × 2 cranes) | S1, S2, S3, S4 | 1,120 | `results/mcrp_experiment_*.csv` |
| M-CRP baselines (140 × 2 cranes) | M2+M3+M4 via ZoneSplit | 280 | `results/mc_extra_baselines/results.csv` |
| Backward compat | M0 vs M1 | 5 | `results/critical_fixes/backward_compat.csv` |
| Cost decomposition | M1-S1→S4 | 1,120 | (from experiment CSV) |
| Case study | S1 vs S2 (6-bay) | 2 | `results/critical_fixes/case_study.json` |

### Còn thiếu (cần chạy thêm):

| Dataset | Thiếu methods | Lý do |
|---------|---------------|-------|
| Shin_random (80) | **Tất cả M0-M8** | Chưa chạy bao giờ |
| Shin_upsidedown (80) | **Tất cả M0-M8** | Chưa chạy bao giờ |
| Lee_random (71) | M8 BeamSearchCRP | Chưa chạy |
| Lee_upsidedown (40) | M8 BeamSearchCRP | Chưa chạy |

---

## 4. KẾ HOẠCH CHẠY THEO 5 PHASES

### 🟢 PHASE 1: Shin instances — ZeroShot + OriginalModel (quan trọng nhất)
**Mục tiêu:** So sánh trực tiếp với phương pháp gốc của paper (Shin et al. 2026) trên toàn bộ Shin dataset.
**Thời gian:** ~40 phút

| Batch | Dataset | Files | Methods | Runs | Output |
|-------|---------|-------|---------|------|--------|
| 1a | Shin_random | 80 | M0 + M1 | 160 | `results/phase1/shin_random_zo.csv` |
| 1b | Shin_upsidedown | 80 | M0 + M1 | 160 | `results/phase1/shin_ud_zo.csv` |

```bash
# Chạy: PYTHONUNBUFFERED=1 python analysis/phase1_shin_zo.py
# Output: 320 runs, ~40 phút
```

---

### 🟢 PHASE 2: Shin instances — 5 Heuristics nhanh
**Mục tiêu:** So sánh ZeroShot với tất cả heuristics trên Shin dataset.
**Chỉ chạy 5 methods nhanh nhất** (M2, M3, M4, M6, M7). **KHÔNG** chạy M5 Durasevic2025 (quá chậm trên instance lớn).
**Thời gian:** ~30 phút

| Batch | Dataset | Files | Methods | Runs | Output |
|-------|---------|-------|---------|------|--------|
| 2a | Shin_random | 80 | M2+M3+M4+M6+M7 | 400 | `results/phase2/shin_random_fast.csv` |
| 2b | Shin_upsidedown | 80 | M2+M3+M4+M6+M7 | 400 | `results/phase2/shin_ud_fast.csv` |

```bash
# Chạy: PYTHONUNBUFFERED=1 python analysis/phase2_shin_fast.py
# Output: 800 runs, ~30 phút
```

---

### 🟡 PHASE 3: Shin instances — Durasevic2025 + BeamSearch subset
**Mục tiêu:** So sánh với advanced baselines nhưng chỉ trên subset nhỏ (20-30 bay nhỏ nhất).
**Thời gian:** ~30 phút

| Batch | Dataset | Files | Methods | Runs | Output |
|-------|---------|-------|---------|------|--------|
| 3a | Shin_random (10 nhỏ nhất) | 10 | M5 Durasevic2025 | 10 | `results/phase3/shin_durasevic.csv` |
| 3b | Shin_upsidedown (10 nhỏ nhất) | 10 | M5 Durasevic2025 | 10 | |
| 3c | Shin_random (3 nhỏ nhất) | 3 | M8 BeamSearchCRP | 3 | `results/phase3/shin_beamsearch.csv` |

```bash
# Chạy: PYTHONUNBUFFERED=1 python analysis/phase3_slow.py
# Output: 23 runs, ~30 phút
```

---

### 🟡 PHASE 4: BeamSearchCRP trên Lee subset
**Mục tiêu:** So sánh BeamSearch trên Lee instances (chỉ subset nhỏ vì quá chậm).
**Thời gian:** ~30 phút

| Batch | Dataset | Files | Methods | Runs | Output |
|-------|---------|-------|---------|------|--------|
| 4a | Lee_random (5 nhỏ nhất) | 5 | M8 BeamSearchCRP | 5 | `results/phase4/beamsearch_lee.csv` |
| 4b | Lee_upsidedown (3 nhỏ nhất) | 3 | M8 BeamSearchCRP | 3 | |

```bash
# Chạy: PYTHONUNBUFFERED=1 python analysis/phase4_beamsearch.py
# Output: 8 runs, ~30 phút
```

---

### 🔴 PHASE 5: Tổng hợp + Update Paper
**Mục tiêu:** Gộp tất cả kết quả, phân tích, cập nhật paper.
**Thời gian:** ~60 phút

| Batch | Nội dung | Thời gian | Output |
|-------|----------|-----------|--------|
| 5a | Merge tất cả CSV vào 1 file | 5 phút | `results/comprehensive_all.csv` |
| 5b | Chạy full analysis (tables + figures) | 10 phút | `results/comprehensive_analysis.txt` |
| 5c | Update paper: | | |
| | - Thêm Shin results vào Table I, III | 20 phút | `crp_rl_paper_Q1.tex` |
| | - Thêm BeamSearch results | | |
| | - Cập nhật Abstract với Shin numbers | | |
| 5d | Compile + verify | 10 phút | `crp_rl_paper_Q1.pdf` |

---

## 5. MA TRẬN KẾT QUẢ ĐẦY ĐỦ (MONG ĐỢI SAU 5 PHASES)

| Dataset | M0 Orig | M1 ZeroShot | M2 Lin2015 | M3 Kim2016 | M4 Level | M5 Durasevic | M6 Nearest | M7 Lowest | M8 BeamSearch |
|---------|---------|-------------|------------|------------|----------|-------------|------------|-----------|---------------|
| **Lee_random** (71) | 7.87% | **6.77%** | 22.30% | 41.12% | 25.31% | 47.52% | 79.56% | 77.22% | ⏳ P4 |
| **Lee_UD** (40) | 5.61% | **4.71%** | 22.62% | 50.00% | 18.30% | 58.09% | 76.90% | 70.36% | ⏳ P4 |
| **Shin_random** (80) | ⏳ P1 | ⏳ P1 | ⏳ P2 | ⏳ P2 | ⏳ P2 | ⏳ P3 | ⏳ P2 | ⏳ P2 | ⏳ P3 |
| **Shin_UD** (80) | ⏳ P1 | ⏳ P1 | ⏳ P2 | ⏳ P2 | ⏳ P2 | ⏳ P3 | ⏳ P2 | ⏳ P2 | ⏳ P3 |

---

## 6. CHI TIẾT LỆNH CHẠY

### Phase 1:
```python
# analysis/phase1_shin_zo.py
# Chạy M0 (OriginalModel) + M1 (ZeroShot) trên 160 Shin instances
# Output: results/phase1/
```

### Phase 2:
```python
# analysis/phase2_shin_fast.py
# Chạy M2, M3, M4, M6, M7 trên 160 Shin instances
# Output: results/phase2/
```

### Phase 3:
```python
# analysis/phase3_slow.py
# Chạy M5 (Durasevic2025) + M8 (BeamSearchCRP) trên subset
# Output: results/phase3/
```

### Phase 4:
```python
# analysis/phase4_beamsearch.py
# Chạy M8 (BeamSearchCRP) trên Lee subset
# Output: results/phase4/
```

### Phase 5:
```python
# analysis/phase5_merge.py
# Gộp tất cả results + update paper
```

---

## 7. SKIP

| Phương pháp | Dataset | Lý do |
|-------------|---------|-------|
| RolloutGP (M9) | Tất cả | ~140s-2h/instance, không khả thi |
| Durasevic2025 (M5) | Shin full | ~140s/instance × 160 = ~6 giờ, chỉ chạy subset P3 |
| BeamSearchCRP (M8) | Shin full | ~35s-30ph/instance × 160 = không khả thi |

---

## 8. TỔNG THỜI GIAN

| Phase | Nội dung | Thời gian | Có thể dừng? |
|-------|----------|-----------|-------------|
| **1** 🔴 | Shin: ZeroShot + OriginalModel | **40 phút** | ✅ Sau batch 1a |
| **2** 🟡 | Shin: 5 heuristics nhanh | **30 phút** | ✅ |
| **3** 🟡 | Shin: Durasevic + BeamSearch subset | **30 phút** | ✅ |
| **4** 🟡 | BeamSearch Lee subset | **30 phút** | ✅ |
| **5** 🔴 | Tổng hợp + Paper | **60 phút** | ✅ |
| **Tổng** | | **~3 giờ** (chia 5 lần) | |
