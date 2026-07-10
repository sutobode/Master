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
| **E1** | `benchmarks/Lee_instances/` — 70 instance chuẩn Lee & Lee (2010): 50 random + 20 upside-down, 1-10 bays | Lin2015, Kim2016, Leveling, Durasevic2025 | **Original Model** (pretrained, greedy) | **ZeroShotPolicy chạy qua `MCEnv(n_cranes=1)`** — đo thật trên cả 70 instance (không chỉ assert bằng unit test), kèm assert `diff < 0.01%` so với Original Model | `python -m analysis.run_single_crane_v2` | `results/single_crane_v2.csv` |
| **E1b** *(mới)* | `benchmarks/Shin_instances/` — scale cực lớn: 20/30 bay, 1440-2880 container | Lin2015 (baseline mạnh nhất từ E1) | **Original Model** | **ZeroShotPolicy** | `python -m analysis.run_single_crane_large` | `results/single_crane_large.csv` |
| **E2** | `benchmarks/mc_instances/lee_mc/` — 70 layout M-CRP (50R+20U) × C∈{2,3} | — (không có baseline multi-crane ở bước này) | — (Shin et al. không có multi-crane) | **S1/S2/S3/S4** (4 crane-assignment strategies) | `python experiment.py --cranes 2 3 --strategies S1 S2 S3 S4 --output results/mcrp_experiment_v2_main.csv` | `results/mcrp_experiment_v2_main.csv` |
| **E3** | cùng dataset E2 | **M-Lin2015, M-Kim2016, M-Leveling** × **cả 4 strategy S1-S4** (ma trận đầy đủ 3×4, không chỉ S2) | — | **ZeroShot** × cả 4 strategy (đã có từ E2, join theo `n_cranes`+`strategy`) | `python -m analysis.run_mc_baselines_v2` | `results/mc_baselines_v2.csv` |
| **E4** *(mới)* | `benchmarks/mc_instances/lee_mc_large/` — subset layout scale lớn (20/30 bay) | — | — | **S1 vs S2**, C=2 (thu gọn phạm vi vì mỗi instance rất chậm) | `python -m analysis.run_mc_large --n_instances 3` | `results/mc_large_v2.csv` |

Cả các experiment dùng chung `analysis/analyze.py` (bảng thống kê) và `analysis/visualize_v2.py` (hình, 300 DPI).

**Vì sao thêm E1b/E4:** dataset `Shin_instances/` (20/30 bay, benchmark scale cực lớn của chính Shin et al.) trước đây **chưa từng được đưa vào bất kỳ thực nghiệm nào** — không phải do cố ý bỏ qua mà do một bug thật: `benchmarks/generate_mc_instances.py`'s scale `'large'` (bays 20/30) trỏ nhầm vào `Lee_instances` (chỉ có tới bay=10) nên luôn sinh ra 0 file, bị `except: continue` nuốt lỗi âm thầm. Đã sửa (`generate_mc_instances.py` giờ trỏ `'large'` sang `Shin_instances` đúng), verify bằng `python -m benchmarks.generate_mc_instances --report` (không đổi gì, chỉ đếm) và `tests/test_mc_instances.py::test_large_scale_is_wired_to_shin_instances`.

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

### Bước 2 — Experiment 1: Single-crane (SOTA gốc + baselines + phương pháp đề xuất)

```bash
python -m analysis.run_single_crane_v2
```

- 70 instances × 6 methods (Original Model, **ZeroShot**, Lin2015, Kim2016, Leveling, Durasevic2025) = 420 runs.
- Thời gian ước tính: vài phút trên CPU (Original Model/ZeroShot ~1s/instance; heuristics nhanh hơn).
- Seed cố định cho các heuristic có tie-break ngẫu nhiên (Lin2015) → kết quả reproducible.
- Script tự động assert `max diff(ZeroShot, Original) < 0.01%` trên toàn bộ 70 instance — đây chính là bằng chứng thực nghiệm cho "zero-shot fidelity", không chỉ dựa vào unit test.

### Bước 2b *(tuỳ chọn)* — Experiment 1b: Scale generalization trên `Shin_instances`

```bash
python -m analysis.run_single_crane_large --max_per_scale 3
```

- Mặc định 3 instance/nhóm (8 nhóm bay×tier×type) × 3 method = 72 runs. **Mỗi instance scale lớn (1440-2880 container) có thể mất 20-100+ giây** (đúng như Shin et al. tự báo cáo) — tăng `--max_per_scale` (tối đa 20) chỉ nếu có nhiều thời gian.
- Trả lời câu hỏi: pipeline zero-shot có còn hoạt động tốt ở scale mà multi-crane thực sự cần thiết không (yard nhỏ 1-10 bay không cần 2-3 crane).

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

### Bước 4 — Experiment 3: So sánh với multi-crane baselines (ma trận đầy đủ)

```bash
python -m analysis.run_mc_baselines_v2
```

- 70 layouts × 2 crane-counts × 3 heuristics × **4 strategies** = 1,680 runs (trước đây chỉ chạy với S2 → 420 runs; giờ đầy đủ S1-S4 để trả lời 2 câu: (a) DRL có thắng heuristic khi cùng chiến lược không, và (b) thứ hạng S1-S4 tìm thấy ở Experiment 2 có đúng cho cả heuristic không hay chỉ là đặc thù của DRL). Chạy qua **cùng** `MCEnv` + timing model như ZeroShot — so sánh move-for-move công bằng.
- Thời gian ước tính: 60-120 phút (gấp ~4 lần bản trước do quét đủ 4 strategy thay vì chỉ S2).
- Script tự in ra 2 bảng: "headline" (method × S2, để so trực tiếp với Experiment 2) và "full matrix" (method × strategy đầy đủ).

### Bước 4b *(tuỳ chọn)* — Experiment 4: Multi-crane trên scale lớn

```bash
python -c "from benchmarks.generate_mc_instances import generate_large; generate_large()"
python -m analysis.run_mc_large --n_instances 3
```

- Sinh dataset large-scale multi-crane vào thư mục **riêng** `benchmarks/mc_instances/lee_mc_large/` (không đụng tới `lee_mc/` 70-layout mặc định).
- Mặc định 3 instance × 2 strategy (S1, S2) × C=2 = 6 runs — cố tình thu hẹp vì mỗi run ở scale này chậm hơn nhiều lần Experiment 2. Tăng `--n_instances` nếu muốn phủ rộng hơn (tối đa 160 layout có sẵn).
- Trả lời câu hỏi: lợi thế của S2 (spatial) so với S1 và mức speedup makespan có tăng lên ở yard lớn hơn không — đây là quy mô thực sự cần 2-3 crane, khác với 70 layout nhỏ mặc định.

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
├── analysis/run_single_crane_v2.py       # Experiment 1 (nay có cả ZeroShot)
├── analysis/run_single_crane_large.py    # Experiment 1b (Shin_instances, scale lớn)
├── analysis/run_mc_baselines_v2.py       # Experiment 3 (ma trận đầy đủ 3 heuristic × 4 strategy)
├── analysis/run_mc_large.py              # Experiment 4 (multi-crane scale lớn, subset)
├── analysis/analyze.py                   # Bảng thống kê (metric mới)
├── analysis/visualize_v2.py              # Hình (300 DPI)
│
├── baselines/*.py                        # Heuristic single-crane (Lin2015, Kim2016, Leveling, Durasevic2025)
├── baselines/multi_crane/                # LinDest/KimDest/LevelingDest (destination rule qua cùng driver)
│
├── benchmarks/
│   ├── Lee_instances/                    # 70 instance chuẩn (single-crane + multi-crane mặc định)
│   ├── Shin_instances/                   # benchmark scale cực lớn — dùng cho E1b/E4
│   ├── generate_mc_instances.py          # Sinh dataset multi-crane (1 file/layout); generate_large() cho scale lớn
│   ├── mc_instances/lee_mc/              # 70 layout M-CRP mặc định (small+medium)
│   └── mc_instances/lee_mc_large/        # layout M-CRP scale lớn (sinh riêng, không mặc định)
│
├── tests/                                # Unit tests
├── results/                               # Output thực nghiệm (gitignored)
│   ├── single_crane_v2.csv               # Experiment 1
│   ├── single_crane_large.csv            # Experiment 1b
│   ├── mcrp_experiment_v2_main.csv       # Experiment 2
│   ├── mc_baselines_v2.csv               # Experiment 3
│   ├── mc_large_v2.csv                   # Experiment 4
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
- `benchmarks/generate_mc_instances.py`'s `crane_start_bays()` có thể tạo cấu hình degenerate khi `n_cranes > n_bays` (ví dụ instance 1-bay chạy C=3) — A7 khi đó không thể enforce (`a7_violations` sẽ > 0), đã được log rõ ràng thay vì âm thầm bỏ qua, nhưng bản thân cấu hình này về mặt vật lý không thực tế (3 crane trên 1 bay).
- Experiment 4 (large-scale multi-crane) mặc định chỉ test S1/S2 và C=2 trên vài instance — không phải ma trận đầy đủ như Experiment 2/3, vì chi phí mỗi run ở scale 1440-2880 container cao hơn nhiều. Nếu cần kết quả toàn diện hơn ở scale lớn, tăng dần `--n_instances` và tự thêm S3/S4/C=3 vào `STRATEGY_MAP`/vòng lặp trong `analysis/run_mc_large.py`.

---

## 10. References

- Shin, W.-J., Choi, I., Cho, S.-H., Kim, H.-J. (2026). "Learning to retrieve containers: A scale-diverse deep reinforcement learning approach for the container retrieval problem." *Transportation Research Part C*, 183, 105496.
- Lee, Y., Lee, Y.-J. (2010). "A heuristic for retrieving containers from a yard." *Computers & Operations Research*, 37(6), 1139-1147.
- Lin, D.-Y., Lee, Y.-J., Lee, Y. (2015). "The container retrieval problem with respect to relocation." *Transportation Research Part C*, 52, 132-143.
- Kim, Y., Kim, T., Lee, H. (2016). "Heuristic algorithm for retrieving containers." *Computers & Industrial Engineering*, 101, 352-360.
- Kwon, Y.-D. et al. (2020). "POMO: Policy Optimization with Multiple Optima." *NeurIPS*.
