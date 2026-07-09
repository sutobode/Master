# M-CRP: Multi-Crane Container Retrieval Problem

**Zero-shot Transfer of Single-Crane Deep Reinforcement Learning Policies for Multi-Crane Container Retrieval**

> ⚠️ **10/07/2026 — README này đã viết lại sau ultra-review + sửa lỗi lớn.** Số liệu trong README cũ (gap 6.56%, 140 instances, LB_MCRP = LB_retrieval + LB_relocation/C...) đến từ một lower bound KHÔNG hợp lệ (có thể cho gap âm) và một dataset bị đếm đôi (140 = 70 layout × 2). Đã sửa cả hai. Chi tiết đầy đủ: `docs/superpowers/plans/2026-07-10-revision-handoff.md`. README này mô tả pipeline **sau khi sửa** — chưa có số liệu cuối cùng (bạn tự chạy theo hướng dẫn dưới để lấy số thật).

---

## 1. Tổng quan

Repository nghiên cứu **Zero-shot Transfer** của DRL policy từ single-crane sang multi-crane CRP (M-CRP). Câu hỏi nghiên cứu: một policy DRL chỉ huấn luyện trên bài toán 1 crane có transfer được sang môi trường nhiều crane (2-3 crane) thông qua các chiến lược phân công crane, mà không cần huấn luyện lại?

- **M-CRP Environment** (`mcenv/`): môi trường nhiều crane với mô hình thời gian per-crane (A6 interference = chờ, A7 non-crossing = enforce thật), đo cả `total_cost` (tổng công) và `makespan` (thời gian hoàn thành — objective chính)
- **ZeroShotPolicy** (`policy/`): trích xuất encoder + scorer từ model gốc của Shin et al. (2026), tại C=1 khớp model gốc **chính xác 0.00%**
- **4 Crane Assignment Strategies** (`strategies/`): S1 RoundRobin, S2 ZoneSplit, S3 LoadBalance, S4 GreedyOptimal — đây là **phương pháp đề xuất chính** của repo
- **M-CRP Lower Bound hợp lệ** (`bounds/lowerbound_mc.py`): `LB_work` + `LB_makespan`, chứng minh không bao giờ vượt quá chi phí một lịch trình khả thi
- **Baselines** (`baselines/`): heuristic gốc từ literature (Lin2015, Kim2016, Leveling, Durasevic2025) + phiên bản multi-crane của chúng (`baselines/multi_crane/`)
- **SOTA gốc để so sánh**: model pretrained của Shin et al. (2026) (`baselines/models/proposed/epoch(100).pt`) — đây chính là "Original Model" trong Experiment 1

---

## 2. Cài đặt

```bash
pip install -r requirements.txt   # file có version trùng lặp; cài bản khớp CPU/CUDA torch của bạn
```

Model pretrained của Shin et al. (2026) đã có sẵn tại `baselines/models/proposed/epoch(100).pt`.

```bash
python -m pytest tests/ -v        # xác nhận môi trường chạy đúng trước khi thực nghiệm
```

---

## 3. ⚠️ Đọc trước khi chạy

- **`python -m benchmarks.generate_mc_instances` XOÁ toàn bộ file cũ trong `benchmarks/mc_instances/lee_mc/` trước khi ghi lại.** Nếu bạn muốn giữ dataset hiện tại (140 file `_c2`/`_c3` cũ), hãy backup trước: `cp -r benchmarks/mc_instances/lee_mc benchmarks/mc_instances/lee_mc.bak`.
- Dataset multi-crane **đúng chuẩn mới** là **1 file / layout** (không có hậu tố `_c2`/`_c3`) — số crane là tham số thực nghiệm (`--cranes`), không phải một chiều của dataset. Nếu thư mục `lee_mc/` hiện đang là 140 file định dạng cũ, bạn **phải chạy lại `generate_mc_instances`** trước khi dùng `experiment.py`/`run_full_experiment.py`/`analysis.run_mc_baselines_v2` — các script này đọc header `# crane_start_bays_c2 = ...` mà định dạng cũ không có.
- Không có script nào ở đây tự động chạy khi bạn mở repo — mọi lệnh dưới đây bạn tự gõ và tự quan sát log.

---

## 4. Ma trận thực nghiệm: Dataset × Baseline × SOTA × Phương pháp đề xuất

| # | Dataset | Baseline (heuristic literature) | SOTA gốc (Shin et al. 2026) | Phương pháp đề xuất (repo này) | Script | Output |
|---|---------|----------------------------------|------------------------------|----------------------------------|--------|--------|
| **E1** | `benchmarks/Lee_instances/` — 70 instance chuẩn Lee & Lee (2010): 50 random + 20 upside-down, 1-10 bays | Lin2015, Kim2016, Leveling, Durasevic2025 | **Original Model** (pretrained, greedy) | *(ZeroShot(C=1) ≡ Original Model chính xác 0.00% — không cần chạy riêng)* | `python -m analysis.run_single_crane_v2` | `results/single_crane_v2.csv` |
| **E2** | `benchmarks/mc_instances/lee_mc/` — 70 layout M-CRP (50R+20U) × C∈{2,3} | — (không có baseline multi-crane ở bước này) | — (Shin et al. không có multi-crane) | **S1/S2/S3/S4** (4 crane-assignment strategies) | `python experiment.py --cranes 2 3 --strategies S1 S2 S3 S4 --output results/mcrp_experiment_v2_main.csv` | `results/mcrp_experiment_v2_main.csv` |
| **E3** | cùng dataset E2 | **M-Lin2015, M-Kim2016, M-Leveling** (baseline đơn-crane mở rộng, chạy qua cùng driver/protocol) | — | **ZeroShot + S2 (ZoneSplit)** — cấu hình tốt nhất từ E2 | `python -m analysis.run_mc_baselines_v2` | `results/mc_baselines_v2.csv` |

Cả 3 experiment dùng chung `analysis/analyze.py` (bảng thống kê) và `analysis/visualize_v2.py` (hình, 300 DPI).

---

## 5. Hướng dẫn chạy theo thứ tự

### Bước 0 — Unit test

```bash
python -m pytest tests/ -v
```

### Bước 1 — (chỉ nếu cần) sinh lại dataset multi-crane đúng format mới

```bash
# backup nếu muốn giữ dataset cũ, xem mục 3
python -m benchmarks.generate_mc_instances
python -m pytest tests/test_mc_instances.py -v   # phải pass 4/4 sau bước này
```

### Bước 2 — Experiment 1: Single-crane (SOTA gốc + baselines)

```bash
python -m analysis.run_single_crane_v2
```

- 70 instances × 5 methods (Original Model, Lin2015, Kim2016, Leveling, Durasevic2025) = 350 runs.
- Thời gian ước tính: vài phút trên CPU (Original Model ~1s/instance; heuristics nhanh hơn).
- Seed cố định cho các heuristic có tie-break ngẫu nhiên (Lin2015) → kết quả reproducible.

### Bước 3 — Experiment 2: So sánh 4 chiến lược phân công crane (phương pháp đề xuất)

```bash
# smoke test trước (5-10s):
python experiment.py --quick

# full run:
python experiment.py --cranes 2 3 --strategies S1 S2 S3 S4 --output results/mcrp_experiment_v2_main.csv
```

- 70 layouts × 2 crane-counts × 4 strategies = 560 runs.
- Mỗi run được assert `gap_work >= 0` và `gap_makespan >= 0` — nếu thấy `AssertionError: ... bound invalid`, đó là dấu hiệu lower bound sai ở edge case, **đừng bỏ qua**, báo lại.
- Muốn chạy theo batch có thể resume (hữu ích cho scale lớn `--cranes` nhiều hoặc dataset lớn hơn): `python run_full_experiment.py --batch_size 15`.

### Bước 4 — Experiment 3: So sánh với multi-crane baselines

```bash
python -m analysis.run_mc_baselines_v2
```

- 70 layouts × 2 crane-counts × 3 heuristics (M-Lin2015, M-Kim2016, M-Leveling) = 420 runs, chạy qua **cùng** `MCEnv` + `ZoneSplit` + timing model như ZeroShot (Experiment 2's S2) — so sánh move-for-move công bằng.

### Bước 5 — Phân tích + hình vẽ

```bash
python -m analysis.analyze results/mcrp_experiment_v2_main.csv
python -m analysis.visualize_v2
```

Output:
- `results/analysis_report.txt` — Table 1 (gap theo strategy), Table 2 (Wilcoxon signed-rank có `sigma_d`), Table 3 (interference + A6 wait + A7 reassignments/violations), Table 4-5 (theo scale/số bay), Table 6 (win count), Table 7 (theo R/U type), speedup makespan C=2→C=3, failure modes.
- `results/figures_v2/*.png` — 6 hình (gap theo strategy, interference wait, gap theo số bay, speedup boxplot, single-crane theo bay, so sánh multi-crane baselines).

### Bước 6 — Điền số vào paper

Mở `docs/latex/crp_rl_paper_Q1.tex`, thay hết các đoạn `%%FILL_...` bằng số liệu từ 3 file CSV ở trên (qua `analysis/analyze.py`, không gõ tay từ trí nhớ), rồi biên dịch:

```bash
cd docs/latex && pdflatex -interaction=nonstopmode crp_rl_paper_Q1.tex && pdflatex -interaction=nonstopmode crp_rl_paper_Q1.tex
```

---

## 6. Metric chính

| Metric | Ý nghĩa | Công thức |
|--------|---------|-----------|
| `gap_makespan` | **Metric chính.** % chênh lệch makespan (thời gian hoàn thành muộn nhất trong các crane) so với `LB_makespan` | `100*(makespan - LB_makespan)/LB_makespan` |
| `gap_work` | Metric phụ. % chênh lệch tổng công việc so với `LB_work` | `100*(total_cost - LB_work)/LB_work` |
| `interference_wait` | Tổng thời gian một crane phải CHỜ vì bay đang bận (A6) | tích luỹ trong `MCEnv.step()` |
| `a7_reassignments` | Số lần A7 (non-crossing) buộc phải đổi crane thực thi so với crane strategy chọn | đếm trong `_enforce_a7` |
| `a7_violations` | Số lần A7 **không thể** enforce được (không crane nào compatible, ví dụ C > số bay) — cần xem là dấu hiệu cấu hình instance không phù hợp với số crane | đếm trong `_enforce_a7` nhánh degenerate |

Tại sao `makespan` là metric chính chứ không phải `total_cost`: thêm crane không giảm được tổng khối lượng công việc (mọi retrieval/relocation vẫn phải làm), chỉ giảm được thời gian hoàn thành (makespan) nhờ chạy song song — đây là lý do multi-crane operation có giá trị.

---

## 7. Cấu trúc file (sau revision)

```
CRP_RL/
├── model/, env/, trainer.py, main.py     # DRL model gốc (frozen, không train lại)
│
├── mcenv/mcenv.py                        # Multi-crane env: timing model per-crane, A6 wait, A7 enforce
├── policy/zero_shot.py                   # Trích xuất policy (fidelity 0.00% tại C=1)
├── strategies/                           # S1-S4 (phương pháp đề xuất)
├── bounds/lowerbound_mc.py                # LB_work + LB_makespan hợp lệ (dict trả về, không phải tensor)
├── engine/mcrp_inference.py              # Driver loop episode
│
├── experiment.py                         # Experiment 2 (in-memory)
├── run_full_experiment.py                # Experiment 2 (batched/resumable)
├── analysis/run_single_crane_v2.py       # Experiment 1
├── analysis/run_mc_baselines_v2.py       # Experiment 3
├── analysis/analyze.py                   # Bảng thống kê (metric mới)
├── analysis/visualize_v2.py              # Hình (300 DPI)
│
├── baselines/*.py                        # Heuristic single-crane (Lin2015, Kim2016, Leveling, Durasevic2025)
├── baselines/multi_crane/                # LinDest/KimDest/LevelingDest (destination rule qua cùng driver)
│
├── benchmarks/
│   ├── Lee_instances/                    # 70 instance chuẩn (single-crane)
│   ├── Shin_instances/                   # benchmark scale cực lớn (không dùng cho M-CRP, xem mục 9)
│   ├── generate_mc_instances.py          # Sinh dataset multi-crane (1 file/layout)
│   └── mc_instances/lee_mc/              # 70 layout M-CRP
│
├── tests/                                # Unit tests
├── results/                               # Output thực nghiệm (gitignored)
│   ├── single_crane_v2.csv               # Experiment 1
│   ├── mcrp_experiment_v2_main.csv       # Experiment 2
│   ├── mc_baselines_v2.csv               # Experiment 3
│   ├── analysis_report.txt
│   └── figures_v2/
│
├── compare_mc_baselines.py, analysis/run_comprehensive.py,   # DEPRECATED — xem CLAUDE.md,
│ analysis/supplementary_analysis.py, fix_critical_issues.py, # exit ngay khi chạy với thông báo
│ run_mc_baselines_extra.py, analysis/visualize.py            # trỏ sang script _v2 tương ứng
│
└── docs/
    ├── latex/crp_rl_paper_Q1.tex          # Paper (còn %%FILL_ placeholder chờ số liệu)
    └── superpowers/plans/2026-07-10-revision-handoff.md   # Chi tiết mọi thay đổi + lý do
```

---

## 8. Claim → Code → Test traceability

| Claim | Code | Test |
|-------|------|------|
| C1: M-CRP definition (A6 interference, A7 non-crossing) | `mcenv/mcenv.py` | `tests/test_mcenv.py` (A6 = wait test, A7 = compatible/enforce test) |
| C2: Lower bound hợp lệ (không có gap âm) | `bounds/lowerbound_mc.py` | `tests/test_lowerbound_mc.py::test_lb_mc_valid_no_negative_gap` |
| C3: 4 crane-assignment strategies | `strategies/*.py` | `tests/test_strategies.py` |
| C4: Zero-shot fidelity tại C=1 (0.00%) | `mcenv/mcenv.py`, `policy/zero_shot.py` | `tests/test_mcenv.py::test_mcenv_c1_episode_cost_identical_to_env` |
| C5: Multi-crane baselines công bằng (cùng driver) | `baselines/multi_crane/multi_crane_baseline.py` | chạy `analysis/run_mc_baselines_v2` và so `gap_makespan` cột `method` |
| C6: Public benchmark dataset | `benchmarks/generate_mc_instances.py` | `tests/test_mc_instances.py` |

---

## 9. Giới hạn đã biết (chưa sửa, cần cân nhắc khi diễn giải kết quả)

- **A6 timing model không khoá hết các bay trung gian khi một chuỗi auto-retrieval đi qua nhiều bay liên tiếp** (`clear()` cascade) — chỉ khoá bay bắt đầu và bay kết thúc của chuỗi. Có thể làm `interference_wait`/`a7_reassignments` bị đánh giá thấp hơn thực tế trong các trường hợp retrieval cascade dài qua nhiều bay (thường gặp ở instance upside-down). Không sửa được nhanh mà không động vào `Env.clear()` (rủi ro phá vỡ invariant C=1=0.00% đã verify).
- `benchmarks/Shin_instances/` (20/30 bay, scale cực lớn) hiện **chưa** có script `_v2` benchmark tương ứng — Experiment 1 chỉ chạy trên `Lee_instances`. Nếu muốn kiểm tra khả năng scale của ZeroShot/Original Model trên Shin_instances, cần viết thêm script tương tự `analysis/run_single_crane_v2.py` trỏ sang thư mục đó (không nằm trong phạm vi M-CRP contribution chính).
- `benchmarks/generate_mc_instances.py`'s `crane_start_bays()` có thể tạo cấu hình degenerate khi `n_cranes > n_bays` (ví dụ instance 1-bay chạy C=3) — A7 khi đó không thể enforce (`a7_violations` sẽ > 0), đã được log rõ ràng thay vì âm thầm bỏ qua, nhưng bản thân cấu hình này về mặt vật lý không thực tế (3 crane trên 1 bay).

---

## 10. References

- Shin, W.-J., Choi, I., Cho, S.-H., Kim, H.-J. (2026). "Learning to retrieve containers: A scale-diverse deep reinforcement learning approach for the container retrieval problem." *Transportation Research Part C*, 183, 105496.
- Lee, Y., Lee, Y.-J. (2010). "A heuristic for retrieving containers from a yard." *Computers & Operations Research*, 37(6), 1139-1147.
- Lin, D.-Y., Lee, Y.-J., Lee, Y. (2015). "The container retrieval problem with respect to relocation." *Transportation Research Part C*, 52, 132-143.
- Kim, Y., Kim, T., Lee, H. (2016). "Heuristic algorithm for retrieving containers." *Computers & Industrial Engineering*, 101, 352-360.
- Kwon, Y.-D. et al. (2020). "POMO: Policy Optimization with Multiple Optima." *NeurIPS*.
