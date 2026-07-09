# 📋 KẾ HOẠCH THỰC NGHIỆM TOÀN DIỆN
## Zero-shot Transfer of Single-Crane DRL Policies for M-CRP

---

## 1. TỔNG QUAN DATASETS

| Dataset | Path | Random | Upside-down | Total | Ghi chú |
|---------|------|--------|-------------|-------|---------|
| **Lee benchmark** | `benchmarks/Lee_instances/` | 51 files | 20 files | **71** | Standard CRP benchmark |
| **Shin et al. instances** | `benchmarks/Shin_instances/` | ~80 files | ~80 files | **160** | Paper's own test set |
| **M-CRP instances** | `benchmarks/mc_instances/` | 70 files | 70 files | **140** | Our extension |
| **Tổng single-crane** | | ~131 | ~100 | **231** | |
| **Tổng multi-crane** | | 70 | 70 | **140** | |

---

## 2. METHODS (BASELINES + PROPOSED)

### 2.1 Single-crane Methods (8 methods)

| ID | Method | Code | Loại | Tốc độ | Đã chạy? |
|----|--------|------|------|--------|---------|
| M0 | **Original Model (Shin et al.)** | `Model` class, `epoch(100).pt` | SOTA DRL | ~0.5s/inst | ❌ Chưa trên Shin + LeeUD |
| M1 | **ZeroShot (ours)**  | `ZeroShotPolicy` | Proposed | ~0.5s/inst | ❌ Chưa trên Shin + LeeUD |
| M2 | Lin2015 | `baselines/lin2015.py` | Heuristic | ~0.3s/inst | ❌ Chưa trên Shin + LeeUD |
| M3 | Kim2016 | `baselines/kim2016.py` | Heuristic | ~0.3s/inst | ❌ Chưa trên Shin + LeeUD |
| M4 | Leveling | `baselines/leveling.py` | Heuristic | ~0.3s/inst | ❌ Chưa trên Shin + LeeUD |
| M5 | Durasevic2025 | `baselines/durasevic2025.py` | GP-evolved | ~0.5s/inst | ❌ Chưa trên Shin + LeeUD |
| M6 | NearestStack | `baselines/simple_baselines.py` | Simple | ~0.2s/inst | ❌ Chưa trên Shin + LeeUD |
| M7 | LowestHeight | `baselines/simple_baselines.py` | Simple | ~0.2s/inst | ❌ Chưa trên Shin + LeeUD |

### 2.2 Multi-crane Methods (7 methods)

| ID | Method | Code | Loại | Đã chạy? |
|----|--------|------|------|---------|
| MC1-S1 | ZeroShot + RoundRobin | `strategies/round_robin.py` | Proposed | ✅ 140 inst |
| MC1-S2 | ZeroShot + ZoneSplit | `strategies/zone_split.py` | Proposed | ✅ 140 inst |
| MC1-S3 | ZeroShot + LoadBalance | `strategies/load_balance.py` | Proposed | ✅ 140 inst |
| MC1-S4 | ZeroShot + GreedyOptimal | `strategies/greedy_optimal.py` | Proposed | ✅ 140 inst |
| MC2 | M-Lin2015 | Lin2015 + ZoneSplit | Heuristic | ✅ 140 inst |
| MC3 | M-Kim2016 | Kim2016 + ZoneSplit | Heuristic | ✅ 140 inst |
| MC4 | M-Leveling | Leveling + ZoneSplit | Heuristic | ✅ 140 inst |

---

## 3. KẾ HOẠCH CHẠY THEO BATCH

### Phase A: Chạy single-crane trên datasets còn thiếu

**Mục tiêu:** Chạy 8 methods trên 180 instances chưa test (Shin_random + Shin_upsidedown + Lee_upsidedown)

| Batch | Dataset | Methods | Runs | Thời gian | Log file |
|-------|---------|---------|------|-----------|----------|
| **A1** | Shin_random (~80) | M0 + M1 (Original + ZeroShot) | ~160 | ~2 phút | `logs/batch_A1_shin_random_original_zs.txt` |
| **A2** | Shin_upsidedown (~80) | M0 + M1 (Original + ZeroShot) | ~160 | ~2 phút | `logs/batch_A2_shin_ud_original_zs.txt` |
| **A3** | Lee_upsidedown (20) | M0 + M1 (Original + ZeroShot) | ~40 | ~1 phút | `logs/batch_A3_lee_ud_original_zs.txt` |
| **A4** | Shin_random (~80) | M2-M7 (6 heuristics) | ~480 | ~3 phút | `logs/batch_A4_shin_random_heuristics.txt` |
| **A5** | Shin_upsidedown (~80) | M2-M7 (6 heuristics) | ~480 | ~3 phút | `logs/batch_A5_shin_ud_heuristics.txt` |
| **A6** | Lee_upsidedown (20) | M2-M7 (6 heuristics) | ~120 | ~1 phút | `logs/batch_A6_lee_ud_heuristics.txt` |

**Tổng Phase A:** ~1,440 runs, ~12 phút

### Phase B: So sánh Original Model vs ZeroShot

| Batch | Nội dung | Runs | Thời gian | Log file |
|-------|----------|------|-----------|----------|
| **B1** | Paired comparison trên ALL 231 instances | 231 | ~5 phút | `logs/batch_B1_original_vs_zs_analysis.txt` |
| B1 includes: | Gap difference, correlation, % diff | -- | -- | -- |
| | Instances where diff > 2% | -- | -- | -- |

### Phase C: Cập nhật Paper

| Step | Nội dung | Thời gian |
|------|----------|-----------|
| C1 | Update Table 1: thêm Shin + LeeUD results | ~15 phút |
| C2 | Update Table 2: Multi-crane results (đã có) | ~5 phút |
| C3 | Update Table 3: Multi-crane baselines (đã có) | ~5 phút |
| C4 | Thêm "Original Model vs ZeroShot" comparison section | ~15 phút |
| C5 | Update figures với data mới | ~10 phút |
| C6 | Compile + review | ~10 phút |

---

## 4. CHI TIẾT LOGGING

Mỗi batch sẽ ghi log ra file riêng với format:

```
=== BATCH A1: Shin_random - Original Model + ZeroShot ===
Started: 2026-07-09 22:00:00
Instance: Shin_R011606_001 -> Original cost=4807.2 ZeroShot cost=4713.6 diff=1.95%
Instance: Shin_R011606_002 -> Original cost=4872.0 ZeroShot cost=4793.0 diff=1.63%
...
[50/160] 25% complete | 90s remaining
...
Completed: 160 runs in 125s
Results saved: results/comprehensive/phase_A1.csv
```

### Metrics ghi lại cho mỗi run:
- `instance`: tên file
- `dataset`: Lee_random / Lee_UD / Shin_random / Shin_UD
- `method`: OriginalModel / ZeroShot / Lin2015 / ...
- `cost`: total working time
- `lb`: lower bound
- `gap`: gap(%)
- `time_s`: thời gian chạy
- `n_steps`: số steps

---

## 5. CODE THỰC HIỆN

```bash
# Gộp tất cả vào 1 script duy nhất:
python analysis/run_comprehensive_experiments.py
# Script này sẽ:
#   1. Chạy Phase A1-A6 tuần tự
#   2. Ghi log từng batch
#   3. Lưu kết quả vào results/comprehensive/
#   4. Chạy Phase B1 (so sánh Original vs ZeroShot)
#   5. Tạo summary tables cho paper
```

---

## 6. TIMELINE DỰ KIẾN

| Phase | Thời gian | Có thể theo dõi? |
|-------|-----------|-----------------|
| A1 + A2 (Shin: Original + ZS) | ~4 phút | ✅ progress bar + log |
| A3 (LeeUD: Original + ZS) | ~1 phút | ✅ |
| A4 + A5 (Shin: heuristics) | ~6 phút | ✅ |
| A6 (LeeUD: heuristics) | ~1 phút | ✅ |
| B1 (Analysis) | ~5 phút | ✅ |
| C (Paper update) | ~60 phút | Thủ công |
| **Tổng** | **~1.5 giờ** | |

---

Bạn duyệt plan này để tôi bắt đầu chạy nhé?
