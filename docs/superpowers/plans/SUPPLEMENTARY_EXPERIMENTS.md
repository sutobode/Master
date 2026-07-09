# Kế hoạch thực nghiệm bổ sung

## Vấn đề: Shin instances (20-30 bays, 1440-2880 containers)

160 instances, mỗi instance cực lớn. Heuristics (Durasevic2025) mất ~140s/instance → 160 × 140s = ~6 giờ.

## Giải pháp: Chạy có chọn lọc

### Phase 1: Chỉ chạy ZeroShot + Original Model (nhanh, ~30 phút)

| Batch | Dataset | Methods | Runs | Thời gian |
|-------|---------|---------|------|-----------|
| S1 | Shin_random (80) | ZeroShot + OriginalModel | 160 | ~15 phút |
| S2 | Shin_upsidedown (80) | ZeroShot + OriginalModel | 160 | ~15 phút |

**Kết quả:** So sánh ZeroShot vs Original Model trên Shin instances (cùng scale với paper gốc).

### Phase 2: Heuristics trên subset (optional, ~30 phút)

| Batch | Dataset | Methods | Runs | Thời gian |
|-------|---------|---------|------|-----------|
| S3 | Shin_random + UD (mỗi loại 10 inst nhỏ nhất) | Lin2015, Kim2016, Leveling | 60 | ~5 phút |
| S4 | Shin_random + UD (mỗi loại 10 inst) | Durasevic2025 | 20 | ~25 phút |

### Phase 3: BeamSearchCRP trên subset nhỏ (~15 phút)

| Batch | Dataset | Runs | Thời gian |
|-------|---------|------|-----------|
| S5 | Lee_random (3 inst nhỏ nhất) | 3 | ~15 phút |

---

## Code chạy

```bash
# Phase 1: ZeroShot + OriginalModel trên Shin (30 phút)
python -u -c "
from policy.zero_shot import ZeroShotPolicy
from model.model import Model
# ... (script sẽ được viết sẵn)
" 2>&1 | tee logs/shin_original_zs.log

# Phase 2: Heuristics
python analysis/run_shin_heuristics.py

# Phase 3: BeamSearch
python analysis/run_beamsearch_subset.py
```

## Output

Mỗi batch output ra file CSV riêng + log riêng:
```
results/shin_original_zs.csv
results/shin_heuristics.csv
logs/shin_phase_1.log
...
```

---

Bạn muốn tôi chạy Phase 1 trước (30 phút) rồi quyết định Phase 2-3 sau?
