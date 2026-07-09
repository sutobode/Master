# 📋 KẾ HOẠCH THỰC NGHIỆM ĐẦY ĐỦ

## Mục tiêu: Chạy TẤT CẢ methods trên TẤT CẢ datasets

---

## 1. TRẠNG THÁI HIỆN TẠI

### ✅ Đã hoàn thành: 2,368 runs

| Dataset | Methods | Runs | Status |
|---------|---------|------|--------|
| **Lee_random** (71 instances) | ZeroShot, OriginalModel, Lin2015, Kim2016, Leveling, Durasevic2025, NearestStack, LowestHeight | 568 | ✅ |
| **Lee_upsidedown** (40 instances) | ZeroShot, OriginalModel, Lin2015, Kim2016, Leveling, Durasevic2025, NearestStack, LowestHeight | 320 | ✅ |
| **M-CRP (140 instances × 2 cranes)** | S1 RoundRobin, S2 ZoneSplit, S3 LoadBalance, S4 GreedyOptimal | 1,120 | ✅ |
| **M-CRP baselines (140 instances × 2 cranes)** | M-Lin2015, M-Kim2016, M-Leveling | 280 | ✅ |
| **Backward compatibility** | ZeroShot(C=1) vs OriginalModel | 5 | ✅ |
| **Cost decomposition** | All strategies | 1,120 | ✅ |
| **Case study (6-bay)** | S1 vs S2 | 2 | ✅ |
| **Tổng** | | **~2,415** | ✅ |

### ❌ Chưa chạy:

| Dataset | Lý do | Thời gian dự kiến |
|---------|-------|-------------------|
| **Shin_random** (80 inst, 20-30 bays, 1,440-2,880 containers) | Instances quá lớn → heuristics rất chậm | **~30-45 phút** (chỉ ZeroShot + OriginalModel) |
| **Shin_upsidedown** (80 inst) | Tương tự | **~30-45 phút** |
| **BeamSearchCRP** trên subset | Baseline advanced chưa chạy | **~15 phút** (3 instances) |
| **Chi tiết per-bay breakdown** (Table cho từng B=1,2,4,6,8,10) | Đã có data, cần tổng hợp | ~5 phút |

---

## 2. KẾ HOẠCH CHẠY BỔ SUNG

### Phase 1: Shin instances (quan trọng nhất)

Chạy ZeroShot + OriginalModel trên 160 Shin instances để so sánh với paper gốc.

| Batch | Dataset | Methods | Runs | Thời gian | Lệnh |
|-------|---------|---------|------|-----------|-------|
| **S1** | Shin_random (80) | ZeroShot + OriginalModel | 160 | ~20 phút | `python analysis/run_shin_zs_original.py --type random` |
| **S2** | Shin_upsidedown (80) | ZeroShot + OriginalModel | 160 | ~20 phút | `python analysis/run_shin_zs_original.py --type upsidedown` |

```bash
# Script chạy Shin instances:
# analysis/run_shin_zs_original.py
# Output: results/shin_zeroshot_original.csv (320 runs)
# Log: logs/shin_experiment_YYYYMMDD_HHMMSS.txt
```

### Phase 2: Heuristics trên Shin subset (optional)

Chạy heuristics trên 20 Shin instances nhỏ nhất (20 bays, 6 tiers).

| Batch | Dataset | Methods | Runs | Thời gian |
|-------|---------|---------|------|-----------|
| **S3** | Shin 20-bay × 6-tier (10 inst) | Lin2015, Kim2016, Leveling | 30 | ~5 phút |
| **S4** | Shin 20-bay × 6-tier (10 inst) | Durasevic2025 | 10 | ~20 phút |

### Phase 3: BeamSearchCRP (optional)

| Batch | Dataset | Runs | Thời gian |
|-------|---------|------|-----------|
| **S5** | 3 Lee instances nhỏ (1-bay) | 3 | ~15 phút |

---

## 3. TỔNG HỢP KẾT QUẢ MONG ĐỢI

Sau khi chạy xong Phase 1, bảng so sánh sẽ đầy đủ:

| Dataset | ZeroShot | OriginalModel | Lin2015 | Leveling | Kim2016 | Durasevic2025 |
|---------|----------|---------------|---------|----------|---------|---------------|
| **Lee_random** (71) | 6.77% | 7.87% | 22.30% | 25.31% | 41.12% | 47.52% |
| **Lee_UD** (40) | 4.71% | 5.61% | 22.62% | 18.30% | 50.00% | 58.09% |
| **Shin_random** (80) | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ |
| **Shin_UD** (80) | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ |

---

## 4. FILE LOGGING

Mỗi batch ghi log riêng, progress bar rõ ràng:

```
logs/
├── shin_experiment_20260710_120000.txt    # Phase 1
├── shin_heuristics_20260710_130000.txt    # Phase 2
└── beamsearch_20260710_140000.txt         # Phase 3

results/
├── shin_zeroshot_original.csv             # Phase 1 output
├── shin_heuristics_subset.csv             # Phase 2 output
├── beamsearch_subset.csv                  # Phase 3 output
└── comprehensive_summary.csv              # Tổng hợp tất cả
```

---

## 5. TÁC ĐỘNG ĐẾN PAPER

Sau khi chạy xong, paper cần cập nhật:

| Phần | Cập nhật |
|------|----------|
| Abstract | Thêm Shin instances results |
| Table I (SOTA) | Thêm dòng Shin instances |
| Table III (Single-crane) | Mở rộng với Shin data |
| Section 6.1 | Phân tích thêm Shin results |
| Section 6.5 (Sensitivity) | Thêm discussion về dataset sensitivity |
| Appendix | Thêm Shin result tables |

---

## 6. TỔNG THỜI GIAN DỰ KIẾN

| Phase | Thời gian | Ưu tiên | 
|-------|-----------|---------|
| **S1+S2: Shin ZeroShot + Original** | **~40 phút** | 🔴 **P0 - Phải chạy** |
| S3+S4: Heuristics Shin subset | ~25 phút | 🟡 P1 - Nên chạy |
| S5: BeamSearch subset | ~15 phút | 🟢 P2 - Optional |
| **Tổng** | **~80 phút** | |

---

Bạn muốn tôi bắt đầu chạy Phase 1 (Shin instances, ~40 phút) ngay bây giờ không?
