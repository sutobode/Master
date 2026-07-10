# Script Consolidation + Verification Handoff — 2026-07-10 (continued)

> Bàn giao tiếp nối `2026-07-10-revision-handoff.md` (đọc file đó trước để hiểu bối cảnh Phase A/B). Phần này ghi lại việc **gộp lại các script thực nghiệm** (sau khi user chỉ ra 5 script E1-E4 bị scatter/không nhất quán) và **verify thật trên dữ liệu nhỏ** (không phải chỉ đọc code) — tìm ra và sửa thêm 3 bug mới trong quá trình verify.

## Vì sao phải gộp lại

User chỉ ra: E1 (`run_single_crane_v2.py`) chỉ chạy trên `Lee_instances`, còn E1b (`run_single_crane_large.py`) chạy trên `Shin_instances` nhưng chỉ với 3/6 phương pháp (thiếu Kim2016/Leveling/Durasevic2025) — 2 script "cùng 1 setting" nhưng method set khác nhau, không so sánh được. Tương tự E2 (`experiment.py`, chỉ ZeroShot) và E3 (`run_mc_baselines_v2.py`, chỉ 3 heuristic) ghi ra 2 CSV riêng phải tự join tay. E4 (`run_mc_large.py`) chỉ test S1/S2, không heuristic nào.

**Nguyên tắc gộp:** đúng 2 setting về bản chất (không thể trộn):
- **Setting A (single-crane, C=1)**: SOTA (Original Model) + ZeroShot (đề xuất) + 4 heuristic — áp dụng như nhau cho MỌI scale.
- **Setting B (multi-crane, C∈{2,3})**: ZeroShot×S1-S4 (đề xuất) vs 3 heuristic multi-crane×S1-S4 — không có "Original Model" vì Shin et al. không có bản multi-crane (đây chính là khoảng trống bài báo lấp vào).

Mỗi setting = **1 script**, tham số hoá theo scale qua `--dataset`.

## Thay đổi code

### File mới
- `analysis/run_single_crane_full.py` — thay `run_single_crane_v2.py` + `run_single_crane_large.py`. `--dataset {lee,shin}`, luôn chạy đủ 6 method. Thêm `--max_instances` (dùng `itertools.islice`) để smoke-test nhanh không cần sửa file. Lưu report riêng `results/single_crane_{dataset}_report.txt` (trước đây bảng theo R/U type chỉ in terminal, không lưu).
- `analysis/run_multi_crane_full.py` — thay `experiment.py`'s sweep + `run_mc_baselines_v2.py` + `run_mc_large.py`. `--dataset {small,large}`, mỗi (instance, n_cranes, strategy) chạy cả ZeroShot lẫn 3 heuristic trong CÙNG 1 CSV (cột `method`), không cần join tay nữa.
- `run_all.py` — master pipeline gọi đúng các script trên theo thứ tự đúng của README mục 5. Tự phát hiện `lee_mc/` format cũ và dừng lại trừ khi có `--regenerate-mc` (không tự ý xoá dữ liệu). Chỉ chuẩn bị dataset multi-crane nếu có bước multi-crane thực sự sẽ chạy (`--skip-multi-small`/`--skip-multi-large` tắt đúng phần chuẩn bị tương ứng, không chặn nhầm khi chỉ chạy Setting A). Mặc định KHÔNG giới hạn số instance (full data) — `--max-per-scale`/`--max-instances-large` chỉ để rút ngắn test.

### File deprecated (tự `sys.exit()` với thông báo khi chạy trực tiếp, KHÔNG xoá)
`analysis/run_single_crane_v2.py`, `analysis/run_single_crane_large.py`, `analysis/run_mc_baselines_v2.py`, `analysis/run_mc_large.py`, `run_full_experiment.py`. `experiment.py` vẫn giữ (không deprecate) vì `--quick` vẫn dùng để smoke-test, và các hàm `parse_instance_file`/`load_instance_tensor`/`verify_backward_compatibility` được `run_multi_crane_full.py` import lại.

### Bug tìm thấy VÀ SỬA trong lúc gộp (đọc code, chưa chạy)
1. **`run_multi_crane_full.py` thiếu cột `interference`** (chỉ có `interference_wait`) khi gộp logic từ 2 script cũ — copy sót 1 dòng. Đã thêm lại `'interference': res.get('n_interference', 0)`.
2. **`analyze.py` group theo `(n_cranes, strategy)` không lọc `method`** — khi CSV mới gộp cả ZeroShot lẫn 3 heuristic, Table 1-7 (vốn để trả lời "strategy nào tốt nhất cho ZeroShot") sẽ **trộn lẫn số của DRL và heuristic** thành một số vô nghĩa nếu không lọc trước. Đã sửa: `MCRPAnalyzer.__init__` tự phát hiện `method` có >1 giá trị (`self.has_baselines`), lọc `self.df` còn ZeroShot cho Table 1-7, giữ `self.full_df` không lọc cho **Table 8 (mới)** — so sánh method thật (headline S2-only + full matrix).
3. **`visualize_v2.py` vẫn trỏ vào tên CSV cũ** (`mcrp_experiment_v2_main.csv`, `single_crane_v2.csv`, `mc_baselines_v2.csv`) — sẽ không tìm thấy file hoặc đọc sai. Đã cập nhật sang `multi_crane_small.csv`/`single_crane_lee.csv`; đơn giản hoá `fig_mc_baselines` (không cần merge 2 CSV nữa); thêm `fig7_mc_gap_by_type.png` + `fig8_single_crane_gap_by_type.png` (Random vs Upside-down, theo yêu cầu user).

### Bug tìm thấy khi CHẠY THẬT (không đọc code phát hiện được)
4. **`ax.boxplot(data, labels=STRAT_ORDER, ...)` trong `fig_speedup()` crash trên matplotlib đang cài (3.11.0)** — API đã xoá hẳn tham số `labels=` (chỉ deprecated ở 3.9, removed ở 3.11; `requirements.txt` ghim 3.8.x nhưng môi trường thật cài 3.11.0). Sửa bằng `ax.boxplot(data, showmeans=True)` + `ax.set_xticklabels(STRAT_ORDER)` sau đó — tương thích ngược mọi bản matplotlib, không phụ thuộc API `labels=`/`tick_labels=` mới đổi tên.

**Bài học:** bug #4 chỉ lộ ra khi chạy thật — 3 bug #1-3 lẽ ra cũng có thể ẩn tới khi user chạy full nhiều giờ rồi mới phát hiện lỗi ở bước phân tích cuối cùng. Từ nay, mọi thay đổi lớn vào các script thực nghiệm nên verify bằng 1 lần chạy thật trên dữ liệu tối thiểu (vài instance, vài giây) trước khi giao cho user chạy full.

## Verify thật đã làm (bằng chứng, không phải claim)

- `pytest tests/ -v`: **44/45 pass**. 1 fail (`test_end_to_end_parse_file`) — không phải lỗi code, mà đúng vấn đề dataset dưới đây.
- `run_single_crane_full.py --dataset lee --max_instances 7` (2 loại R+U, dữ liệu thật từ `Lee_instances/`): ZeroShot ≡ OriginalModel diff **0.0000%**; ranking Lin2015(9.29%) < Kim2016(16.95%) < Durasevic2025(17.63%) < Leveling(29.21%) đúng kỳ vọng thứ hạng heuristic. CSV + report lưu đúng, đọc lại xác nhận nội dung khớp.
- `run_multi_crane_full.py --dataset small` chạy trên **4 layout sinh vào thư mục TẠM** (không đụng `lee_mc/` thật — dùng `generate_all(output_dir=tmp)` giống hệt cách unit test làm), đủ 4 strategy × 2 crane-count: backward-compat pass 0.00%; ZeroShot (37-38% gap) thắng cả 3 heuristic (M-Lin2015 41%, M-Leveling 60%, M-Kim2016 79%) đúng kỳ vọng.
- `analyze.py` + `visualize_v2.py` chạy trên CSV thật vừa sinh: Table 7/8 đúng số; cả 8 hình (kể cả 2 hình R/U mới) sinh ra thành công sau khi sửa bug #4.
- `git status --short benchmarks/` sạch trong suốt quá trình — xác nhận dataset thật KHÔNG bị đụng vào bởi bất kỳ bước verify nào ở trên.

## Trạng thái dataset hiện tại (đã kiểm tra trực tiếp trên đĩa)

| Thư mục | Trạng thái | Cần làm |
|---|---|---|
| `Lee_instances/` | 71 file (51R+20U) | Sẵn sàng |
| `Shin_instances/` | 160 file (80R+80U) | Sẵn sàng |
| `mc_instances/lee_mc/` | **140 file, ĐỊNH DẠNG CŨ** (`_c2`/`_c3` suffix, thiếu header `crane_start_bays_c2`) — git-clean, tracked, an toàn phục hồi | User tự chạy `python -m benchmarks.generate_mc_instances` (tôi bị chặn tự động chạy lệnh này vì nó từng xoá dữ liệu 2 lần trước) |
| `mc_instances/lee_mc_large/` | **Chưa tồn tại** | User tự chạy `generate_large()` |

## Lệnh chạy lại full (đã thay thế hoàn toàn bộ lệnh ở file handoff trước)

```bash
python -m pytest tests/ -v                              # 0. đã pass 44/45 (1 fail hết sau bước 1)
python -m benchmarks.generate_mc_instances               # 1. BẮT BUỘC — regenerate lee_mc/ (140 -> 70 file, format mới)
python -m pytest tests/test_mc_instances.py -v           # xác nhận 5/5 pass

python -m analysis.run_single_crane_full --dataset lee    # 2. Setting A nhỏ, ~vài phút
python -m analysis.run_single_crane_full --dataset shin   # 3. Setting A lớn, CHẬM (giờ)
python -m analysis.run_multi_crane_full --dataset small   # 4. Setting B nhỏ
python -c "from benchmarks.generate_mc_instances import generate_large; generate_large()"
python -m analysis.run_multi_crane_full --dataset large   # 5. Setting B lớn, CHẬM NHẤT (>1 ngày có thể)

python -m analysis.analyze results/multi_crane_small.csv  # 6. bảng thống kê (Table 1-8)
python -m analysis.visualize_v2                           # hình (8 hình, 300 DPI)
```

Hoặc `python run_all.py --help` cho bản gộp 1 lệnh (có `--skip-*` để chia nhỏ từng dataset qua nhiều phiên, `--dry-run` để xem trước).

## Đã verify thêm: chạy thật 1 instance/layout ở CẢ scale lớn (không chỉ scale nhỏ)

Sau khi user hỏi lại "chưa chạy scale lớn dù chỉ 1 lần", đã chạy thật (không skip) đúng 1 instance/layout cho cả 2 dataset scale lớn:

- **`run_single_crane_full.py --dataset shin --max_instances 1`** (thêm cờ `--max_instances`, dùng `itertools.islice`): instance `R201606_1440_001.txt` (20 bay, 1440 container). Kết quả: `ZeroShot=8.34%`, `OriginalModel=8.34%` (diff **0.0000%** — backward-compat giữ vững ở scale lớn), `Leveling=15.84%`, `Lin2015=26.01%`, `Durasevic2025=41.49%`, `Kim2016=52.13%`. **Phát hiện khoa học đáng chú ý:** thứ hạng heuristic ĐỔI so với scale nhỏ — `Leveling` vượt `Lin2015` ở scale lớn (ở scale nhỏ Lin2015 luôn mạnh nhất). Nên đưa vào Discussion của paper, không phải bug.
- **`generate_large()`** chạy thật (an toàn — chỉ ghi thư mục `lee_mc_large/` trống trước đó, không đụng `lee_mc/`): sinh đúng 160 layout (80R+80U) như kỳ vọng.
- **`run_multi_crane_full.py --dataset large --max_instances 1`**: layout `mc_R201606_001.txt` (20 bay), đủ 4 method × 4 strategy × 2 crane-count = 32 run. `gap_makespan` headline (S2): C=2 — ZeroShot=364.70 ≈ M-Leveling=366.00 (rất sát nhau) < M-Lin2015=403.12 < M-Kim2016=777.67; C=3 tương tự. CSV lưu đúng 33 dòng (header + 32).
- `git status --short benchmarks/mc_instances/lee_mc/` sạch trong suốt — xác nhận dataset thật (140 file cũ) không bị đụng.

## Đo thời gian thật để estimate full run (máy: 12th Gen Intel Core i7-1270P @ 2.20GHz)

Đo trực tiếp trên máy user (không phải môi trường khác) bằng cách chạy 1 instance/layout thật ở scale nhỏ nhất và lớn nhất mỗi dataset:

| Dataset | Instance đo | Thời gian đo được |
|---|---|---|
| Lee (Setting A nhỏ) | 1 bay, 70 container | ~1.1s / 6 method |
| Lee (Setting A nhỏ) | 6 bay, 720 container (lớn nhất) | 20.3s / 6 method (OriginalModel 4.9s, ZeroShot 3.6s, Lin2015 2.3s, Kim2016 2.0s, Leveling 4.8s, Durasevic2025 2.6s) |
| Shin (Setting A lớn) | 20 bay, 1440 container | **293s / 6 method** |
| lee_mc (Setting B nhỏ) | 1 bay | ~4.5s / 32 run (4 method×4 strategy×2 crane) |
| lee_mc (Setting B nhỏ) | 10 bay (lớn nhất) | **300.4s / 32 run** (chi tiết theo method: M-Kim2016 5.3s/run, M-Leveling 4.1s/run, M-Lin2015 3.3s/run, **ZeroShot 34.7s/run** — DRL chậm hơn heuristic ~8-10x mỗi run) |
| lee_mc_large (Setting B lớn) | 20 bay/tier6 | **379s / 32 run** (M-Kim2016 5.3s, M-Leveling 4.1s, M-Lin2015 3.3s, ZeroShot 34.7s mỗi run — cùng tỷ lệ như trên) |

Ngoại suy tuyến tính theo số layout còn lại (xem README.md mục 4 "Ước tính thời gian chạy full" cho bảng ước tính đầy đủ):
- Setting A nhỏ (70 instance): **~10-15 phút**
- Setting A lớn (160 instance): **~13-18 giờ**
- Setting B nhỏ (70 layout): **~2-4 giờ**
- Setting B lớn (160 layout): **~20-28 giờ**
- **Tổng cả 4, chạy nối tiếp: ~1.5-2 ngày.**

Đây là ngoại suy TUYẾN TÍNH từ 2 điểm đo (nhỏ nhất + lớn nhất) — coi là cận dưới, vì MCEnv/DRL cost có thể tăng nhanh hơn tuyến tính theo kích thước yard (số bước episode tăng theo số container, không chỉ số bay). Đã thêm ETA trực tiếp vào `run_single_crane_full.py` (in `[N/total] ... | ~Xs remaining` mỗi instance, giống `run_multi_crane_full.py` đã có sẵn) để user theo dõi tiến độ thật khi chạy full, không cần tin số ước tính tĩnh này.

**File `results/*.csv`, `results/*_report.txt`, `results/figures_v2/*.png` hiện tại đang chứa dữ liệu VERIFY nhỏ (4-7 instance/layout)** từ quá trình kiểm nghiệm ở trên — sẽ tự động bị ghi đè khi chạy các lệnh full phía trên.

## Tối ưu tốc độ: loại bỏ tính toán DRL dư thừa trong Setting B (~3.9-3.95× nhanh hơn)

User hỏi "có cách nào cải tiến tốc độ ZeroShot chậm hơn heuristic 8-10 lần không". Phân tích + verify phát hiện: đây không phải chi phí không tránh được, mà là **tính toán dư thừa có thể loại bỏ hoàn toàn, không đánh đổi độ chính xác**.

### Phát hiện gốc rễ

Đọc `mcenv/mcenv.py:step()`: `crane_id` chỉ được dùng để cập nhật `self.crane_bays`/`self.crane_time` (vị trí và lịch của crane, phục vụ tính CHI PHÍ di chuyển) — không bao giờ được truyền vào phần quyết định container nào di chuyển tới đâu (`base_env.x`, `base_env.target_stack`). Nghĩa là: **chuỗi quyết định (target_stack, dest_stack) của policy ZeroShot hoàn toàn không phụ thuộc `n_cranes` hay `strategy`** — chỉ phụ thuộc trạng thái yard (x), vốn tiến triển giống hệt nhau bất kể ai (crane nào) thực thi move.

Verify bằng thực nghiệm (không chỉ đọc code): chạy cùng 1 instance qua 3 tổ hợp `(n_cranes=2, S1)`, `(n_cranes=3, S2)`, `(n_cranes=2, S4)` — cả 3 cho ra đúng 462 bước, **chuỗi (target_stack, dest_stack) giống hệt nhau từng bước một**.

→ `run_multi_crane_full.py` (bản trước khi sửa) gọi `run_mcrp_episode()` — tức chạy lại encoder DRL (phần đắt, ~90% thời gian 1 run ZeroShot) — **8 lần dư thừa mỗi instance** (2 crane-count × 4 strategy), dù chỉ cần đúng 1 lần.

### Đã sửa

`engine/mcrp_inference.py` thêm 2 hàm mới (không đụng `run_mcrp_episode()` gốc):
- `record_zeroshot_trajectory(policy, x, n_bays, n_rows, n_tiers)` — chạy policy đúng 1 lần (dùng MCEnv n_cranes=1 làm reference), trả về danh sách `dest_stack` theo thứ tự quyết định.
- `replay_zeroshot_episode(dest_sequence, env, strategy)` — phát lại đúng chuỗi đó qua 1 MCEnv/strategy cụ thể, chỉ làm phần kế toán crane/chi phí (rẻ, không gọi policy).

`analysis/run_multi_crane_full.py`: gọi `record_zeroshot_trajectory()` đúng 1 lần/instance (ngoài vòng lặp `n_cranes`/`strategy`), rồi `replay_zeroshot_episode()` cho cả 8 tổ hợp thay vì `run_mcrp_episode()`.

**Test đối chứng bắt buộc trước khi tin dùng:** `tests/test_engine.py::test_replay_matches_run_mcrp_episode_exactly` — so `replay_zeroshot_episode` với `run_mcrp_episode` gốc trên 4 tổ hợp (n_cranes, strategy) khác nhau, assert `total_cost`/`makespan`/`n_steps`/`interference_wait`/`a7_reassignments`/`per_crane_cost` khớp CHÍNH XÁC. Pass. `pytest tests/ -v` sau khi sửa: 45/46 pass (1 fail là vấn đề dataset cũ đã biết, không liên quan).

### Kết quả đo thật trước/sau (1 instance, không chạy full — theo đúng yêu cầu user)

| Scale | Trước | Sau | Tăng tốc | ZeroShot numbers |
|---|---|---|---|---|
| Setting B nhỏ, 10 bay (32 run: 2 crane×4 strategy×4 method) | 300.4s | 76.0s | **3.95×** | `gap_makespan` C=2: 371.82/314.06/363.94/290.08 — **giống hệt cả 2 bản** |
| Setting B lớn, 20 bay (32 run) | 379s | 97s | **3.90×** | `gap_makespan` C=2: 433.84/364.70/422.37/344.03 — **giống hệt cả 2 bản** |

Baseline heuristic (M-Kim2016/M-Leveling/M-Lin2015) lệch nhẹ giữa 2 lần đo (ví dụ 784.64 vs 784.47) — do tie-break ngẫu nhiên (`torch.randint`) không set seed trong `run_multi_crane_full.py`, đã tồn tại từ trước, không liên quan đến tối ưu này.

### Ước tính full sau tối ưu (cập nhật từ bảng cũ)

- Setting B nhỏ (70 layout): ~2-4 giờ → **~30-60 phút**
- Setting B lớn (160 layout): ~20-28 giờ → **~5-8 giờ**
- Setting A (nhỏ + lớn): KHÔNG đổi (~10-15 phút + ~13-18 giờ) — không có redundancy giữa các method để loại bỏ (mỗi method chỉ chạy đúng 1 lần/instance sẵn rồi).
- **Tổng cả 4: ~1.5-2 ngày → ~19-27 giờ (~1 ngày).**

### Cải tiến thêm: multiprocessing (đã implement theo yêu cầu user "tận dụng tối đa")

User xác nhận muốn implement. Đã thêm vào CẢ 2 script (`run_single_crane_full.py`, `run_multi_crane_full.py`):

**Thiết kế:**
- Mỗi instance/layout hoàn toàn độc lập (không chia sẻ state) → dùng `multiprocessing.Pool(n_workers, initializer=_init_worker)`.
- `_init_worker()` chạy 1 lần/process: `torch.set_num_threads(1)` (bắt buộc — nếu không, mỗi process tự sinh nhiều luồng nội bộ, N process × nhiều luồng sẽ tranh chấp CPU thay vì mỗi process dùng đúng 1 core) + load model/policy 1 lần/process (không load lại mỗi instance).
- `pool.imap_unordered()` — thu kết quả theo thứ tự hoàn thành (không theo thứ tự nộp), in ETA dựa trên tốc độ hoàn thành thực tế.
- Ghi CSV: **gom hết kết quả về main process rồi ghi 1 lần ở cuối** (không ghi song song từ nhiều process) — tránh hoàn toàn race condition khi ghi file, đơn giản hơn nhiều so với đồng bộ ghi CSV giữa các process.
- `--workers N` (mặc định `cpu_count()-2` = 14 trên máy 16 luồng của user).

**Verify (test nhỏ, không chạy full — đúng yêu cầu user):**

| Test | Sequential | Song song | Tăng tốc | Đúng kết quả? |
|---|---|---|---|---|
| Setting A, 16 Lee instance, `--workers 8` | 29s | 13s | ~2.2× | `diff` giữa 2 file output: không lệch dòng nào |
| Setting B, 4 layout tạm, `--workers 4` | 26.8s | 12.9s | ~2.1× | ZeroShot rows giống hệt bit-for-bit (87.85/87.31 gap_work, 244.68/412.68 gap_makespan — cả 2 bản) |
| Setting A, 8 Shin instance (large), `--workers 2` | (không hoàn thành — bị timeout 10 phút, dừng giữa chừng theo đúng yêu cầu user) | 3 instance xong lúc 218s/301s/576s tích lũy | Khớp lý thuyết chạy đồng thời 2 worker | N/A (chưa xong để so) |

Tăng tốc chỉ ~2× ở test nhỏ (4-16 task) vì overhead khởi động worker (~1-2s/worker để load model) chiếm tỷ trọng lớn so với việc mỗi task chỉ mất vài giây. Ở scale thật (70-160 instance, mỗi instance mất hàng chục giây đến vài trăm giây), overhead này gần như không đáng kể — tăng tốc kỳ vọng tiệm cận gần số worker hơn (ví dụ 14 worker → có thể 6-10× thực tế do CPU có core P/E không đồng nhất, không phải tăng tuyến tính hoàn hảo). **Chưa đo được ở scale đầy đủ** (sẽ mất hàng giờ, không phù hợp "test nhanh") — số ước tính trong README/run_all.py là ngoại suy, không phải đo trực tiếp.

`pytest tests/ -v` sau khi thêm multiprocessing: vẫn 45/46 pass (không có regression mới).

### Tổng kết 2 tối ưu — ước tính thời gian cuối cùng

| Dataset | Ban đầu | Sau tối ưu #1 (replay) | Sau tối ưu #1+#2 (replay + multiprocessing, ƯỚC TÍNH) |
|---|---|---|---|
| Setting A nhỏ (70 instance) | ~10-15 phút | không đổi | ~3-6 phút |
| Setting A lớn (160 instance) | ~13-18 giờ | không đổi | ~1.5-3 giờ |
| Setting B nhỏ (70 layout) | ~2-4 giờ | ~30-60 phút | ~5-15 phút |
| Setting B lớn (160 layout) | ~20-28 giờ | ~5-8 giờ | ~40 phút-1.5 giờ |
| **Tổng cả 4** | **~1.5-2 ngày** | ~19-27 giờ | **~2.5-5 giờ** |

Cột cuối là ngoại suy, coi là lạc quan — con số thật sẽ hiện qua ETA khi user chạy thật. Nếu thấy ETA tăng dần theo thời gian (không ổn định) khi chạy nhiều giờ liên tục, đó là dấu hiệu throttle nhiệt trên laptop — giảm `--workers` (ví dụ 6-8) để bền hơn.

## Resume: chạy tiếp sau khi mất điện/crash giữa chừng

User lo ngại máy có thể mất điện/crash giữa lúc chạy full (hợp lý — full run giờ vẫn mất vài giờ). Yêu cầu: resume để chạy tiếp phần dataset chưa xong, không phải chạy lại từ đầu.

### Thiết kế

- CSV output được ghi **NGAY LẬP TỨC** sau mỗi instance/layout hoàn thành (không đợi tới cuối): `writer.writerow(...)` → `csv_file.flush()` → `os.fsync(csv_file.fileno())`. `fsync` (không chỉ `flush`) là bắt buộc vì user lo về MẤT ĐIỆN thật, không chỉ Ctrl+C — `flush()` chỉ đẩy dữ liệu từ buffer Python xuống buffer của OS, `fsync()` mới đẩy thật xuống đĩa, sống sót qua việc mất điện đột ngột.
- Mỗi instance/layout được worker trả về NGUYÊN KHỐI (toàn bộ dòng của nó cùng lúc) rồi mới ghi — không có tình trạng ghi dở dang giữa chừng 1 instance (atomic ở mức instance/layout).
- Khi khởi động: đọc CSV cũ (nếu có) bằng `pd.read_csv`, lấy set các instance/file đã hoàn thành, lọc bỏ khỏi danh sách task trước khi nộp vào pool — instance đã xong không bao giờ được tính toán lại.
  - `run_single_crane_full.py`: "đã xong" = tên file đã có trong cột `instance` của CSV (mỗi instance luôn có đúng 6 dòng cố định, không phụ thuộc tham số CLI nào).
  - `run_multi_crane_full.py`: "đã xong" = file có ĐÚNG `len(cranes)*len(strategies)*4` dòng trong CSV (khớp đúng số tổ hợp method×strategy×crane hiện tại) — nếu số dòng không khớp (ví dụ user đổi `--cranes`/`--strategies` giữa 2 lần chạy), in cảnh báo và coi là CHƯA xong, chạy lại (chấp nhận rủi ro trùng dòng nhỏ trong trường hợp hiếm này — khuyến nghị dùng `--fresh` nếu đổi tham số).
- Backward-compat assert (`max_zs_diff_pct < 0.01`) được tính lại bao gồm CẢ dữ liệu cũ đã resume (không chỉ dữ liệu mới trong phiên này) — đảm bảo assert vẫn kiểm tra đúng trên TOÀN BỘ dữ liệu đã có, không bỏ sót phần được resume.
- `--fresh`: xoá CSV cũ, chạy lại từ đầu (dùng khi đổi tham số hoặc muốn chạy sạch).

### Verify bằng thực nghiệm thật (dùng `timeout` để buộc dừng giữa chừng, mô phỏng crash/mất điện)

**`run_single_crane_full.py`** (12 instance Lee, `--workers 2`):
1. Chạy liền mạch làm reference: 72 dòng.
2. `timeout 5` buộc dừng sau ~5s → 2 instance xong (12 dòng), tiến trình bị kill, không có process nào sót lại (`tasklist` xác nhận sạch).
3. Chạy lại đúng lệnh cũ → in `Resuming from ...: 2 instances already done` → chỉ chạy 10 instance còn lại → tổng 72 dòng.
4. `diff` (sort trước khi so, vì thứ tự có thể khác do `imap_unordered`) giữa reference và bản resume: **IDENTICAL**.

**`run_multi_crane_full.py`** (8 layout tạm, `--cranes 2 3`, `--workers 2`):
1. Chạy liền mạch làm reference: 256 dòng.
2. `timeout 8` → 2 file xong (64 dòng).
3. Resume lần 1 (bị `timeout 60` cắt giữa chừng lần nữa, mô phỏng crash LẦN THỨ HAI liên tiếp) → tiến độ tăng lên 6 file xong (192 dòng) trước khi bị cắt.
4. Resume lần 2 → hoàn tất 8/8 file, 257 dòng (256 + header).
5. `diff` cột 1-14 (loại cột `time_s` — cột này tự nhiên khác giữa các lần chạy vì đo thời gian thực, không phải bug) giữa reference và bản bị ngắt-resume 2 lần liên tiếp: **các dòng `ZeroShot` giống hệt 100%**; chỉ dòng `M-Kim2016`/`M-Leveling` (heuristic tie-break ngẫu nhiên không set seed, vấn đề đã biết từ trước) lệch nhẹ — không liên quan đến resume.

Kết luận: cơ chế resume đúng và bền, kể cả khi bị ngắt NHIỀU LẦN LIÊN TIẾP giữa chừng, không chỉ 1 lần.
