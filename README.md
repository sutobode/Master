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
- Dataset multi-crane **đúng chuẩn mới** là **1 file / layout** (không có hậu tố `_c2`/`_c3`) — số crane là tham số thực nghiệm, không phải một chiều của dataset. Nếu thư mục `lee_mc/` hiện đang là 140 file định dạng cũ, bạn **phải chạy lại `generate_mc_instances`** trước khi dùng `analysis.run_multi_crane_full` — script này đọc header `# crane_start_bays_c2 = ...` mà định dạng cũ không có.
- Không có script nào ở đây tự động chạy khi bạn mở repo — mọi lệnh dưới đây bạn tự gõ và tự quan sát log.

---

## 4. Ma trận thực nghiệm: 2 setting × 2 scale (đã gộp lại, không còn 5 script rời rạc)

Bài toán có **2 setting khác biệt về bản chất**, không thể trộn lẫn trong 1 bảng:

- **Setting A (single-crane, C=1)**: 6 phương pháp trên 1 dòng — Original Model (SOTA), ZeroShot (đề xuất, phải ≡ Original Model), Lin2015/Kim2016/Leveling/Durasevic2025 (4 baseline).
- **Setting B (multi-crane, C∈{2,3})**: KHÔNG có Original Model (Shin et al. không có phiên bản multi-crane — đây chính là khoảng trống bài báo lấp vào). Baseline hợp lệ duy nhất là 3 heuristic mở rộng multi-crane (M-Lin2015/M-Kim2016/M-Leveling), so với ZeroShot — cả 2 phía đều chạy qua tất cả 4 crane-assignment strategy (S1-S4).

Mỗi setting có **1 script duy nhất**, tham số hoá theo scale dataset (`--dataset`), luôn chạy đủ method set — không còn tình trạng scale nhỏ chạy nhiều method hơn scale lớn:

| Setting | Scale | Dataset | Phương pháp (luôn đủ, không rút gọn theo scale) | Script | Output |
|---|---|---|---|---|---|
| **A** | nhỏ | `benchmarks/Lee_instances/` (70 instance, 50R+20U, 1-10 bay) | OriginalModel, ZeroShot, Lin2015, Kim2016, Leveling, Durasevic2025 | `python -m analysis.run_single_crane_full --dataset lee` | `results/single_crane_lee.csv` |
| **A** | lớn | `benchmarks/Shin_instances/` (20/30 bay, 1440-2880 container) | **cùng 6 phương pháp** như trên | `python -m analysis.run_single_crane_full --dataset shin --max_per_scale N` | `results/single_crane_shin.csv` |
| **B** | nhỏ | `benchmarks/mc_instances/lee_mc/` (70 layout, C∈{2,3}) | ZeroShot, M-Lin2015, M-Kim2016, M-Leveling — mỗi method × cả 4 strategy S1-S4 | `python -m analysis.run_multi_crane_full --dataset small` | `results/multi_crane_small.csv` |
| **B** | lớn | `benchmarks/mc_instances/lee_mc_large/` (20/30 bay, cần sinh trước) | **cùng 4 phương pháp × 4 strategy** như trên | `python -m analysis.run_multi_crane_full --dataset large --max_instances N` | `results/multi_crane_large.csv` |

Cả 4 lần chạy dùng chung `analysis/analyze.py` (bảng thống kê) và `analysis/visualize_v2.py` (hình, 300 DPI).

**Lý do gộp lại:** bản trước đây có 5 script (E1/E1b/E2/E3/E4), mỗi script tự ý rút gọn method set khác nhau ở scale lớn (E1b chỉ 3/6 method, E4 chỉ 2 strategy không heuristic nào) và E2/E3 ghi ra 2 CSV riêng phải tự join bằng tay — khiến 2 script cùng 1 dataset trông như 2 thực nghiệm không so sánh được với nhau. `run_single_crane_full.py`/`run_multi_crane_full.py` thay thế toàn bộ 5 script cũ (nay đã deprecated, tự thoát khi chạy với thông báo trỏ sang script mới), mỗi setting đúng 1 method set áp dụng thống nhất cho cả 2 scale.

**Vì sao dataset `Shin_instances`/`lee_mc_large` (scale lớn) tồn tại:** `benchmarks/generate_mc_instances.py`'s scale `'large'` (bays 20/30) từng trỏ nhầm vào `Lee_instances` (chỉ có tới bay=10) nên luôn sinh ra 0 file, bị `except: continue` nuốt lỗi âm thầm — đã sửa, verify bằng `python -m benchmarks.generate_mc_instances --report` và `tests/test_mc_instances.py::test_large_scale_is_wired_to_shin_instances`.

### Tối ưu tốc độ đã áp dụng: loại bỏ tính toán DRL dư thừa trong Setting B

**Phát hiện:** trong `run_multi_crane_full.py`, `crane_id`/`n_cranes`/`strategy` chỉ ảnh hưởng đến chi phí di chuyển và lịch trình của crane — KHÔNG BAO GIỜ ảnh hưởng container nào di chuyển tới đâu (`MCEnv.step()`'s `crane_id` chỉ feed vào `crane_bays`/`crane_time`, không bao giờ vào `base_env.x`/`target_stack`). Verify bằng thực nghiệm: chuỗi quyết định (target_stack, dest_stack) của ZeroShot **giống hệt bit-for-bit** giữa các tổ hợp (n_cranes, strategy) khác nhau trên cùng 1 instance (462 bước, khớp tuyệt đối giữa C=2/S1, C=3/S2, C=2/S4).

Có nghĩa: script cũ tính DRL forward-pass (phần đắt nhất, chiếm ~90% thời gian 1 run ZeroShot) **8 lần dư thừa mỗi instance** (2 crane-count × 4 strategy), dù chỉ cần tính đúng 1 lần.

**Đã sửa:** thêm `record_zeroshot_trajectory()` (chạy policy đúng 1 lần/instance, ghi lại chuỗi quyết định) + `replay_zeroshot_episode()` (phát lại chuỗi đó qua kế toán crane/chi phí — rẻ như heuristic, không gọi policy nữa) trong `engine/mcrp_inference.py`. Có test đối chứng (`tests/test_engine.py::test_replay_matches_run_mcrp_episode_exactly`) xác nhận `replay` cho kết quả **giống hệt** `run_mcrp_episode` gốc trên 4 tổ hợp (n_cranes, strategy) khác nhau trước khi áp dụng.

**Kết quả đo thật (1 instance mỗi scale, trước/sau tối ưu):**

| Scale | Trước | Sau | Tăng tốc | Kết quả ZeroShot có đổi không |
|---|---|---|---|---|
| Setting B nhỏ, 10 bay (32 run) | 300.4s | 76.0s | **3.95×** | Giống hệt bit-for-bit |
| Setting B lớn, 20 bay (32 run) | 379s | 97s | **3.90×** | Giống hệt bit-for-bit |

### Tối ưu tốc độ #2 đã áp dụng: chạy song song nhiều instance bằng multiprocessing

Cả `run_single_crane_full.py` và `run_multi_crane_full.py` giờ xử lý các instance/layout **song song qua nhiều process** (`--workers N`, mặc định `cpu_count()-2` — máy bạn có 16 luồng nên mặc định dùng 14 process). Mỗi instance hoàn toàn độc lập với instance khác (không chia sẻ state), nên đây là tối ưu an toàn tuyệt đối về mặt kết quả — chỉ cần mỗi worker tự giới hạn `torch.set_num_threads(1)` để 14 process dùng đúng 14 core thay vì mỗi process tự sinh nhiều luồng tranh chấp nhau.

**Verify (test trên vài instance nhỏ, không phải full):**

| Test | Sequential | Song song | Tăng tốc | Kết quả ZeroShot |
|---|---|---|---|---|
| Setting A, 16 instance Lee, `--workers 8` | 29s | 13s | ~2.2× | Giống hệt (diff so sánh bằng `diff`, không lệch dòng nào) |
| Setting B, 4 layout, `--workers 4` | 26.8s | 12.9s | ~2.1× | ZeroShot rows giống hệt bit-for-bit (87.85/87.31, 244.68/412.68...) |

Tốc độ tăng chỉ ~2× ở test nhỏ vì overhead khởi động worker (mỗi worker tự load model ~1-2s) chiếm tỷ trọng lớn khi có ít task. Ở scale thật (70-160 instance >> 14 worker), overhead này không đáng kể so với thời gian tính toán — tốc độ tăng kỳ vọng gần hơn với số worker, nhưng **chưa đo được ở scale đầy đủ** (test full sẽ mất nhiều giờ, không phù hợp để "test nhanh" — sẽ lộ ra khi bạn chạy thật, theo dõi ETA in trực tiếp).

### Ước tính thời gian chạy full (đo THẬT + ngoại suy, sau CẢ 2 tối ưu — trên máy 12th Gen Intel Core i7-1270P @ 2.20GHz, 16 luồng)

| Setting/scale | Trước cả 2 tối ưu | Sau tối ưu #1 (replay) | Sau tối ưu #1+#2 (replay + 14 worker song song, ƯỚC TÍNH) |
|---|---|---|---|
| A, nhỏ (Lee, 70 instance) | ~10-15 phút | ~10-15 phút (không áp dụng #1) | **~3-6 phút** |
| A, lớn (Shin, 160 instance) | ~13-18 giờ | ~13-18 giờ (không áp dụng #1) | **~1.5-3 giờ** |
| B, nhỏ (lee_mc, 70 layout) | ~2-4 giờ | ~30-60 phút | **~5-15 phút** |
| B, lớn (lee_mc_large, 160 layout) | ~20-28 giờ | ~5-8 giờ | **~40 phút-1.5 giờ** |

**Tổng cộng cả 4 dataset: từ 1.5-2 ngày xuống còn ước tính ~2.5-5 giờ.** Cột cuối là NGOẠI SUY từ test nhỏ (~2× ở 4-16 task), không phải đo trực tiếp ở scale đầy đủ — coi là ước tính lạc quan, có thể cao hơn do overhead bộ nhớ/nhiệt độ khi chạy 14 process liên tục nhiều giờ trên laptop. `run_single_crane_full.py`/`run_multi_crane_full.py` đều in ETA trực tiếp khi chạy — con số thật sẽ hiện ra trong vài phút đầu.

**Khuyến nghị:** chạy `--workers` mặc định trước; nếu thấy máy nóng/throttle (ETA tăng dần theo thời gian thay vì ổn định), giảm `--workers` xuống (ví dụ 6-8) để bền hơn cho các lần chạy nhiều giờ.

### Resume — tự động, không cần flag đặc biệt

`run_single_crane_full.py` và `run_multi_crane_full.py` ghi CSV **ngay lập tức** sau mỗi instance/layout hoàn thành (append + `flush()` + `fsync()` — đẩy xuống đĩa thật, không chỉ nằm trong buffer của hệ điều hành) thay vì đợi tới cuối mới ghi. Nếu máy mất điện/crash/bạn lỡ tắt giữa chừng: **chạy lại đúng lệnh cũ, y hệt tham số** — script tự đọc CSV đã có, phát hiện instance/layout nào đã xong, bỏ qua, chỉ chạy tiếp phần còn thiếu.

```bash
python -m analysis.run_single_crane_full --dataset shin      # chạy, giả sử mất điện giữa chừng
python -m analysis.run_single_crane_full --dataset shin      # chạy lại y hệt lệnh trên -> tự resume, in "Resuming from ...: N instances already done"
```

**Đã verify bằng thực nghiệm thật** (không chỉ đọc code): dùng `timeout` để buộc dừng giữa chừng (mô phỏng crash) rồi resume lại 2-3 lần liên tiếp cho cả 2 script — kết quả cuối cùng (`ZeroShot` — cột số liệu chính) **giống hệt bit-for-bit** với 1 lần chạy liền mạch không bị ngắt. Baseline heuristic (M-Kim2016/M-Leveling) có lệch nhẹ giữa các lần chạy do tie-break ngẫu nhiên không set seed — đây là vấn đề đã biết từ trước, không liên quan đến resume.

**Giới hạn cần biết:**
- Resume dựa trên **tên file/instance đã có trong CSV** — nếu bạn đổi `--cranes`/`--strategies` giữa lần chạy bị ngắt và lần resume, `run_multi_crane_full.py` sẽ tự phát hiện (đếm số dòng/instance không khớp số tổ hợp hiện tại) và in cảnh báo — dùng `--fresh` để bỏ qua dữ liệu cũ, chạy lại từ đầu trong trường hợp đó.
- `--fresh`: xoá CSV cũ (nếu có) và chạy lại từ đầu, không resume.
- Mỗi instance/layout được ghi nguyên khối (tất cả method/strategy của nó cùng lúc) — không có tình trạng ghi dở dang giữa chừng 1 instance.

---

## 5. Hướng dẫn chạy theo thứ tự

**Cách nhanh nhất — 1 lệnh chạy hết toàn bộ pipeline** (`run_all.py`, tự động hoá đúng các bước thủ công ở dưới, không thêm thực nghiệm nào mới):

```bash
python run_all.py --dry-run                          # xem trước các bước sẽ chạy, không chạy gì cả
python run_all.py --skip-shin --skip-multi-large      # chỉ scale nhỏ (nhanh) — khuyến nghị chạy trước để xác nhận pipeline chạy đúng
python run_all.py --regenerate-mc                     # FULL: mọi dataset, mọi phương pháp, KHÔNG giới hạn số instance
```

**Mặc định (không truyền `--max-per-scale`/`--max-instances-large`) đã là FULL DATA** — không tự động cắt bớt: Setting A chạy đủ cả 70 Lee_instances lẫn 160 Shin_instances; Setting B chạy đủ cả 70 layout nhỏ lẫn tối đa 160 layout lớn. Mục tiêu là so sánh phương pháp đề xuất (ZeroShot/S1-S4) vượt trội hơn SOTA gốc (Original Model, chỉ có ở Setting A vì Shin et al. không có bản multi-crane) và baseline (4 heuristic ở Setting A, 3 heuristic mở rộng multi-crane ở Setting B) — trên **tất cả** dataset, không phải tập con.

- Tự phát hiện `benchmarks/mc_instances/lee_mc/` đang ở format cũ (140 file) và **dừng lại** với hướng dẫn, trừ khi bạn truyền `--regenerate-mc` (vì đây là thao tác XOÁ file cũ — cố tình không tự động chạy nếu không được yêu cầu rõ).
- Tự sinh `lee_mc_large/` nếu chưa có (an toàn, không đụng thư mục khác).
- Một bước lỗi (ví dụ assert `gap < 0` hoặc backward-compat fail) sẽ **dừng toàn bộ pipeline ngay** — đây là chủ đích, không phải bug, vì lỗi đó nghĩa là lower bound hoặc zero-shot fidelity bị hỏng, không nên chạy tiếp để lấy số liệu tiếp.
- **Cảnh báo thời gian:** bước 4 và 6 (scale lớn) là chậm nhất — full pipeline không skip gì có thể mất **nhiều giờ đến hơn 1 ngày** trên CPU. Chạy `--skip-shin --skip-multi-large` trước để xác nhận mọi thứ hoạt động, sau đó mới chạy full không skip để lấy số liệu thật cho paper.
- Xem `python run_all.py --help` cho đầy đủ flag (`--skip-tests`, `--skip-lee`, `--skip-shin`, `--skip-multi-small`, `--skip-multi-large`, `--skip-analysis`, `--continue-on-error`).

Các bước dưới đây là **chi tiết từng lệnh** mà `run_all.py` gọi tới — đọc để hiểu ý nghĩa, hoặc chạy tay từng bước nếu muốn kiểm soát chặt hơn.

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

### Bước 2 — Setting A, scale nhỏ: Single-crane trên `Lee_instances` (SOTA + 4 baseline + ZeroShot)

```bash
python -m analysis.run_single_crane_full --dataset lee
```

- 70 instances × 6 methods (Original Model, **ZeroShot**, Lin2015, Kim2016, Leveling, Durasevic2025) = 420 runs.
- Thời gian ước tính: vài phút trên CPU (Original Model/ZeroShot ~1s/instance; heuristics nhanh hơn).
- Seed cố định cho các heuristic có tie-break ngẫu nhiên (Lin2015) → kết quả reproducible.
- Script tự động assert `max diff(ZeroShot, Original) < 0.01%` trên toàn bộ 70 instance — đây chính là bằng chứng thực nghiệm cho "zero-shot fidelity", không chỉ dựa vào unit test.

### Bước 2b — Setting A, scale lớn: Single-crane trên `Shin_instances` (CÙNG 6 phương pháp)

```bash
python -m analysis.run_single_crane_full --dataset shin --max_per_scale 20
```

- Mặc định `--max_per_scale 20` → 8 nhóm (bay×tier×type) × 20 × 6 method = 960 runs (tối đa có sẵn). **Mỗi instance scale lớn (1440-2880 container) có thể mất 20-100+ giây/method** → có thể mất nhiều giờ ở `--max_per_scale 20`. Bắt đầu với `--max_per_scale 3` (72×2=144 runs, nhanh) để xem pipeline chạy đúng trước khi mở rộng.
- Trả lời câu hỏi: 6 phương pháp (không chỉ Original+ZeroShot+Lin2015 như bản cũ) có giữ đúng thứ hạng ở scale mà multi-crane thực sự cần thiết không (yard nhỏ 1-10 bay không cần 2-3 crane).

### Bước 3 — Setting B, scale nhỏ: Multi-crane trên `mc_instances/lee_mc/` (ZeroShot vs 3 baseline, cả 4 strategy)

```bash
# smoke test trước (vài phút, subset nhỏ):
python -m analysis.run_multi_crane_full --dataset small --max_instances 3

# full run:
python -m analysis.run_multi_crane_full --dataset small
```

- 70 layouts × 2 crane-counts × 4 strategies × 4 methods (ZeroShot + M-Lin2015 + M-Kim2016 + M-Leveling) = **2,240 runs**, MỘT csv duy nhất — không cần tự join 2 bảng như trước.
- Mỗi run được assert `gap_work >= 0` và `gap_makespan >= 0` — nếu thấy `AssertionError: ... bound invalid`, đó là dấu hiệu lower bound sai ở edge case, **đừng bỏ qua**, báo lại.
- Thời gian ước tính: 60-120 phút.
- Script tự in ra 2 bảng: "headline" (method × S2 only, so trực tiếp) và "full matrix" (method × strategy đầy đủ, trả lời cả câu "thứ hạng S1-S4 có giữ nguyên cho heuristic hay chỉ đúng với DRL").

### Bước 3b — Setting B, scale lớn: Multi-crane trên `mc_instances/lee_mc_large/` (CÙNG 4 phương pháp × 4 strategy)

```bash
python -c "from benchmarks.generate_mc_instances import generate_large; generate_large()"
python -m analysis.run_multi_crane_full --dataset large --max_instances 3     # bắt đầu nhỏ
python -m analysis.run_multi_crane_full --dataset large --max_instances 20    # mở rộng nếu đủ thời gian
```

- Sinh dataset large-scale multi-crane vào thư mục **riêng** `benchmarks/mc_instances/lee_mc_large/` (không đụng tới `lee_mc/` 70-layout mặc định).
- CÙNG 4 method × 4 strategy × 2 crane-count như Bước 3 (không rút gọn method set) — chỉ số instance (`--max_instances`) là tham số co giãn theo thời gian có. `--max_instances 3` → 3×2×4×4=96 runs; `--max_instances 20` → 640 runs, có thể mất nhiều giờ (mỗi instance scale này chậm hơn nhiều lần scale nhỏ).
- Trả lời câu hỏi: lợi thế của S2 (spatial) so với S1/S3/S4 và mức speedup makespan có tăng lên ở yard lớn hơn không — đây là quy mô thực sự cần 2-3 crane, khác với 70 layout nhỏ mặc định.
- Chưa có chế độ batch/resume — nếu cần chạy rất lâu và muốn có thể resume khi bị gián đoạn, báo lại để bổ sung (không tự viết thêm nếu chưa được yêu cầu).

### Bước 4 — Phân tích + hình vẽ

```bash
python -m analysis.analyze results/multi_crane_small.csv
python -m analysis.visualize_v2
```

Output:
- `results/analysis_report.txt` — Table 1 (gap theo strategy, **chỉ tính trên method=ZeroShot** — xem lưu ý dưới), Table 2 (Wilcoxon signed-rank có `sigma_d`), Table 3 (interference + A6 wait + A7 reassignments/violations), Table 4-5 (theo scale/số bay), Table 6 (win count), Table 7 (theo R/U type), **Table 8 (mới): so ZeroShot vs 3 baseline heuristic — 8a headline strategy=S2, 8b ma trận đầy đủ method×strategy**, speedup makespan C=2→C=3, failure modes.
- `results/single_crane_{lee,shin}_report.txt` (mới) — gap theo method + theo R/U type cho Setting A, lưu file thay vì chỉ in ra terminal.
- `results/figures_v2/*.png` — 8 hình: fig1-4 (gap theo strategy/interference/bay/speedup, **chỉ ZeroShot**), fig5 (single-crane theo bay), fig6 (ZeroShot vs 3 baseline heuristic), **fig7 (mới): gap theo Random/Upside-down cho multi-crane**, **fig8 (mới): gap theo Random/Upside-down cho single-crane**.

**Lưu ý quan trọng:** `results/multi_crane_small.csv`/`multi_crane_large.csv` gộp cả ZeroShot lẫn 3 heuristic baseline trong CÙNG 1 file (cột `method`). `analyze.py` tự động lọc `method=='ZeroShot'` cho Table 1-7 (câu hỏi "chiến lược nào tốt nhất cho phương pháp đề xuất") — nếu không lọc, groupby theo `(n_cranes, strategy)` sẽ TRỘN LẪN kết quả DRL và heuristic thành một con số vô nghĩa. Table 8 mới dùng dữ liệu đầy đủ không lọc để trả lời câu hỏi khác: "ZeroShot có thắng heuristic không".

### Bước 5 — Điền số vào paper

Mở `docs/latex/crp_rl_paper_Q1.tex`, thay hết các đoạn `%%FILL_...` bằng số liệu từ 4 file CSV ở trên (qua `analysis/analyze.py`, không gõ tay từ trí nhớ), rồi biên dịch:

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
├── run_all.py                             # Master pipeline: chạy hết mục 5 bằng 1 lệnh (xem --help)
├── model/, env/, trainer.py, main.py     # DRL model gốc (frozen, không train lại)
│
├── mcenv/mcenv.py                        # Multi-crane env: timing model per-crane, A6 wait, A7 enforce
├── policy/zero_shot.py                   # Trích xuất policy (fidelity 0.00% tại C=1)
├── strategies/                           # S1-S4 (phương pháp đề xuất)
├── bounds/lowerbound_mc.py                # LB_work + LB_makespan hợp lệ (dict trả về, không phải tensor)
├── engine/mcrp_inference.py              # Driver loop episode
│
├── analysis/run_single_crane_full.py     # Setting A (C=1): --dataset {lee,shin}, luôn đủ 6 method
├── analysis/run_multi_crane_full.py      # Setting B (C=2,3): --dataset {small,large}, ZeroShot+3 baseline × 4 strategy
├── analysis/analyze.py                   # Bảng thống kê (metric mới)
├── analysis/visualize_v2.py              # Hình (300 DPI)
├── experiment.py                         # nội bộ: parse_instance_file/verify_backward_compatibility dùng chung
│                                          # (CLI --quick vẫn dùng để smoke-test, KHÔNG dùng cho kết quả paper)
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
│   ├── single_crane_lee.csv              # Setting A, scale nhỏ
│   ├── single_crane_shin.csv             # Setting A, scale lớn
│   ├── multi_crane_small.csv             # Setting B, scale nhỏ
│   ├── multi_crane_large.csv             # Setting B, scale lớn
│   ├── analysis_report.txt
│   └── figures_v2/
│
├── compare_mc_baselines.py, analysis/run_comprehensive.py,          # DEPRECATED — xem CLAUDE.md,
│ analysis/supplementary_analysis.py, fix_critical_issues.py,        # exit ngay khi chạy với thông báo
│ run_mc_baselines_extra.py, analysis/visualize.py, run_full_experiment.py,
│ analysis/run_single_crane_v2.py, analysis/run_single_crane_large.py,
│ analysis/run_mc_baselines_v2.py, analysis/run_mc_large.py           # trỏ sang 2 script _full ở trên
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
| C5: Multi-crane baselines công bằng (cùng driver) | `baselines/multi_crane/multi_crane_baseline.py` | chạy `analysis/run_multi_crane_full` và so `gap_makespan` cột `method` |
| C6: Public benchmark dataset | `benchmarks/generate_mc_instances.py` | `tests/test_mc_instances.py` |

---

## 9. Giới hạn đã biết (chưa sửa, cần cân nhắc khi diễn giải kết quả)

- **A6 timing model không khoá hết các bay trung gian khi một chuỗi auto-retrieval đi qua nhiều bay liên tiếp** (`clear()` cascade) — chỉ khoá bay bắt đầu và bay kết thúc của chuỗi. Có thể làm `interference_wait`/`a7_reassignments` bị đánh giá thấp hơn thực tế trong các trường hợp retrieval cascade dài qua nhiều bay (thường gặp ở instance upside-down). Không sửa được nhanh mà không động vào `Env.clear()` (rủi ro phá vỡ invariant C=1=0.00% đã verify).
- `benchmarks/generate_mc_instances.py`'s `crane_start_bays()` có thể tạo cấu hình degenerate khi `n_cranes > n_bays` (ví dụ instance 1-bay chạy C=3) — A7 khi đó không thể enforce (`a7_violations` sẽ > 0), đã được log rõ ràng thay vì âm thầm bỏ qua, nhưng bản thân cấu hình này về mặt vật lý không thực tế (3 crane trên 1 bay).
- `analysis/run_multi_crane_full.py --dataset large` và `analysis/run_single_crane_full.py --dataset shin` chạy đủ method/strategy set như scale nhỏ, nhưng KHÔNG có chế độ batch/resume — một lần chạy dài (`--max_instances`/`--max_per_scale` lớn) bị gián đoạn sẽ mất toàn bộ tiến trình chưa lưu. Bắt đầu với giá trị nhỏ, tăng dần; nếu cần resume thật, đó là việc cần làm thêm (chưa có).
- `requirements.txt` ghim `matplotlib==3.8.x` nhưng môi trường thực tế có thể cài bản mới hơn (đã gặp `3.11.0`, nơi `Axes.boxplot(labels=...)` đã bị xoá hẳn — sửa bằng `set_xticklabels()` sau khi gọi `boxplot()`, xem `analysis/visualize_v2.py:fig_speedup`). Nếu gặp `TypeError: Axes.boxplot() got an unexpected keyword argument` ở API matplotlib khác trong tương lai, đây là cùng một loại vấn đề (version drift giữa file ghim và môi trường cài thật), không phải lỗi logic.

**Đã verify thật trên dữ liệu nhỏ (2026-07-10):** toàn bộ pipeline (`run_single_crane_full.py`, `run_multi_crane_full.py`, `analyze.py`, `visualize_v2.py`) đã chạy thật (không chỉ đọc code) trên 7 instance Lee + 4 layout multi-crane mẫu (sinh vào thư mục tạm, không đụng dataset thật) — 44/45 unit test pass, backward-compat 0.0000%, ZeroShot thắng baseline đúng kỳ vọng, cả 8 hình sinh thành công. Chi tiết: `docs/superpowers/plans/2026-07-10-script-consolidation-handoff.md`.

---

## 10. References

- Shin, W.-J., Choi, I., Cho, S.-H., Kim, H.-J. (2026). "Learning to retrieve containers: A scale-diverse deep reinforcement learning approach for the container retrieval problem." *Transportation Research Part C*, 183, 105496.
- Lee, Y., Lee, Y.-J. (2010). "A heuristic for retrieving containers from a yard." *Computers & Operations Research*, 37(6), 1139-1147.
- Lin, D.-Y., Lee, Y.-J., Lee, Y. (2015). "The container retrieval problem with respect to relocation." *Transportation Research Part C*, 52, 132-143.
- Kim, Y., Kim, T., Lee, H. (2016). "Heuristic algorithm for retrieving containers." *Computers & Industrial Engineering*, 101, 352-360.
- Kwon, Y.-D. et al. (2020). "POMO: Policy Optimization with Multiple Optima." *NeurIPS*.
