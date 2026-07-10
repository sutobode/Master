# Revision Handoff — 2026-07-10

> ⚠️ **Đọc `2026-07-10-script-consolidation-handoff.md` TRƯỚC file này nếu bạn cần biết cách chạy lại thực nghiệm ngay bây giờ.** Phần "Cách tự chạy lại" ở cuối file này (script `run_single_crane_v2.py`/`run_mc_baselines_v2.py`/...) đã LỖI THỜI — các script đó đã được gộp lại thành `analysis/run_single_crane_full.py` + `analysis/run_multi_crane_full.py` + `run_all.py` sau khi user chỉ ra 5 script cũ scatter/không nhất quán method set giữa các scale. Nội dung bên dưới vẫn đúng cho phần **giải thích lý do sửa cost-model/lower-bound/dataset** (Phase A/B) — chỉ có phần lệnh chạy ở cuối là cũ.

> Bàn giao sau phiên ultra-review + bắt đầu sửa paper `crp_rl_paper_Q1.tex`. User đã yêu cầu dừng mọi tiến trình nền và tự chạy tiếp — tài liệu này ghi lại chính xác đã sửa gì, tại sao, và cách chạy lại để lấy số liệu.

**Full audit trace (findings gốc từ paper-audit + 3 reviewer agent):** `C:\Users\X1\.claude\plans\vectorized-inventing-plum.md` — đọc file này trước nếu cần hiểu lý do đằng sau mỗi thay đổi bên dưới.

## ⚠️ Sự cố đã xảy ra và đã khắc phục

Trong lúc sửa `benchmarks/generate_mc_instances.py`, tôi đã cho script xoá toàn bộ file cũ trong `benchmarks/mc_instances/lee_mc/` trước khi ghi lại (để đổi sang format mới), rồi tự chạy nó — xoá mất 140 file gốc của bạn. **Đã khôi phục 100%** từ git (`git checkout -- benchmarks/mc_instances/lee_mc/`, commit `61a9344`), verified nội dung giống hệt bản gốc. `git status` cho thư mục này hiện sạch.

**Bài học áp dụng từ giờ:** không cho bất kỳ script nào xoá file trong thư mục dữ liệu đã có sẵn mà không hỏi trước, kể cả khi mục đích là "dọn dẹp format cũ".

## Tổng quan việc đã làm

Đây là **Phase A + một phần Phase B** trong plan gốc (sửa nền tảng: cost model, lower bound, dataset, timing) — KHÔNG phải cosmetic, mà thay đổi cách tính toàn bộ số liệu. Các phase còn lại (C: re-run + fix số liệu paper, D: bibliography + trình bày) mới làm được một phần (bibliography D1 đã sửa trong file `.tex` mới, còn điền số liệu thật thì CHƯA — file `.tex` hiện có các placeholder `%%FILL_...`).

### 1. `mcenv/mcenv.py` — sửa cost model + thêm timing song song
- **Bug đã sửa:** `step()` cũ reset `curr_bay` về vị trí "tracked" của crane mỗi bước thay vì để env gốc tự nhiên tiếp tục từ nơi retrieval cuối dừng lại → đây là nguyên nhân khiến "ZeroShot rẻ hơn Original" (tới 1.95%) dù action sequence giống hệt nhau. Giờ **ZeroShot(C=1) ≡ Original đến 0.0000%** (verify bằng `experiment.py:verify_backward_compatibility`, tolerance giờ là `0.01%` thay vì `2.0%` cũ).
- Thêm mô hình thời gian per-crane thật: mỗi crane có clock riêng (`crane_time`), mỗi bay có `bay_free_time`. **A6 (interference) giờ là một DELAY thật** (crane phải chờ bay rảnh) thay vì cơ chế cũ âm thầm đổi crane thực thi khác với crane strategy chọn. **A7 (non-crossing) giờ được enforce thật** qua `_a7_compatible`/`_enforce_a7` (code cũ định nghĩa `_validate_non_crossing` nhưng không bao giờ gọi nó — dead code).
- Thêm `env.makespan` (= max thời gian hoàn thành của mọi crane) — đây là **objective quan trọng nhất bị thiếu trong paper cũ**: paper nói về "speedup" nhưng objective cũ là SUM (tổng công không giảm khi thêm crane), nên "speedup" vô nghĩa. Giờ có cả `total_cost` (sum, phụ) và `makespan` (chính).
- API thay đổi: `MCEnv.clear()` giờ là method riêng (không dùng `env.base_env.clear()` trực tiếp nữa) — nó gán vị trí/clock ban đầu cho crane 0 để giữ tương thích C=1.

### 2. `bounds/lowerbound_mc.py` — viết lại hoàn toàn Theorem 3
- **Bug đã sửa (fatal):** công thức cũ `LB_retrieval = LB_single_crane_traversal − reloc` rồi chia `LB_relocation` cho `C` → sinh ra **gap âm** (paper cũ tự thú nhận "negative gap observations" ở Discussion, nhưng gap âm nghĩa là bound KHÔNG HỢP LỆ, không phải "not tight"). Nguyên nhân: LB_retrieval dùng chi phí một-crane-đi-tuần-tự-toàn-yard, trong khi multi-crane chia zone đi ít bay-hop hơn hẳn → cost thật có thể < LB cũ.
- Công thức mới (`compute_lb_mc`) trả về **dict `{'work':..., 'makespan':...}`**, không còn trả về 1 tensor. Cả hai bound đều valid theo nghĩa toán học chặt chẽ (proof sketch trong docstring + trong `.tex` Theorem `thm:lb`), verify bằng test mới: **không bao giờ vượt quá cost của một schedule khả thi** (`test_lb_mc_valid_no_negative_gap`, chạy Monte-Carlo trên vài instance nhỏ với chính sách greedy).
- **Mọi nơi gọi `compute_lb_mc(...).item()` trong code cũ ĐỀU HỎNG** — phải sửa thành `.item()` trên `['work']` hoặc `['makespan']`. Tôi đã sửa trong `experiment.py`, nhưng CHƯA rà hết `compare_mc_baselines.py`, `analysis/run_comprehensive.py`, `analysis/supplementary_analysis.py`, `analysis/fix_critical_issues.py` — **các script này sẽ crash nếu chạy nguyên trạng**, cần sửa trước khi dùng lại (search `compute_lb_mc` để tìm hết).

### 3. `benchmarks/generate_mc_instances.py` — bỏ double-count
- **Bug đã sửa:** file cũ sinh 140 file (`_c2`/`_c3` cho mỗi layout) nhưng nội dung yard **byte-identical** giữa 2 file — tức chỉ có 70 layout unique bị đếm đôi thành "140 instances" trong paper. Cộng thêm `experiment.py` cũ chạy MỖI file × cả 2 giá trị crane count → mỗi layout thực chạy 4 tổ hợp trùng lặp.
- Format mới: **1 file/layout** (70 file: 50 R-type + 20 U-type — không phải "70+70" như paper cũ claim), header chứa cả `crane_start_bays_c2` và `crane_start_bays_c3`. Crane count giờ là tham số thực nghiệm, không phải chiều của dataset.
- **⚠️ File bạn cần TỰ CHẠY LẠI:** `python -m benchmarks.generate_mc_instances` sẽ tạo ra 70 file format mới — nhưng **đây chính là lệnh đã xoá dữ liệu của bạn lần trước** vì nó xoá sạch `lee_mc/` trước khi ghi. Tôi ĐÃ khôi phục 140 file gốc (`_c2`/`_c3`) về đúng trạng thái commit. Nếu bạn chạy lại script mới, nó sẽ **xoá 140 file gốc và thay bằng 70 file mới** — đây là hành vi ĐÚNG Ý ĐỒ theo plan sửa lỗi, nhưng bạn cần biết trước khi chạy. Nếu muốn giữ cả hai, hãy backup thư mục `lee_mc/` (hoặc chỉ `git stash`) trước khi chạy.

### 4. `strategies/greedy_optimal.py` (S4) và `strategies/load_balance.py` (S3)
- S4 cũ chỉ là travel-cost tĩnh + penalty phẳng, không hề "simulate" gì (đúng như `FIX_PLAN.md` đã ghi nhận từ trước). Giờ dùng `env.crane_time`/`env._bay_ready` để ước lượng thời gian hoàn thành thật (earliest-finish-time) — đúng nghĩa "1-step lookahead".
- S3 cũ dùng đếm số task đã giao + tie-break theo index → **kết quả identical với S1 (RoundRobin) trên mọi instance**, paper cũ trình bày như 2 strategy riêng biệt (không defensible). Giờ dùng `env.crane_time` (busy time thật) để cân bằng tải — sẽ cho kết quả khác S1 khi tải không đều.
- Cả hai strategy giờ cần `env._a7_compatible`, `env._bay_ready`, `env.crane_time` tồn tại trên object `env` truyền vào — nếu bạn viết code gọi strategy với một mock env tối giản, phải cung cấp đủ các thuộc tính này (xem `tests/test_strategies.py:make_mock_env` làm ví dụ).

### 5. `experiment.py`
- `parse_instance_file()` đổi return signature: `(data_lines, crane_starts_dict)` thay vì `(data_lines, n_cranes, crane_starts_list)` — `crane_starts_dict` map `{2: [...], 3: [...]}`.
- Output CSV đổi cột: `gap` → `gap_work` + `gap_makespan`; thêm cột `makespan`, `lb_work`, `lb_makespan`. Cột `lb_mc` cũ không còn.
- Thêm `assert gap >= 0` — nếu bạn thấy `AssertionError: Negative ... gap` khi chạy, đó là dấu hiệu bound vẫn sai ở edge case nào đó, đừng bỏ qua assert này.

### 6. `analysis/analyze.py` — viết lại cho schema cột mới
- `table1_gap_comparison(gap_col=...)` (mặc định `gap_makespan`), giữ alias `table1_gap_per_strategy` cho code cũ gọi tên cũ.
- `table2_pairwise_wilcoxon` giờ báo cáo thêm `sigma_d` (std của paired differences) — paper cũ tính "power >0.99" dựa trên marginal std sai kỹ thuật, giờ có số đúng để dùng.
- `compute_speedup()` giờ dùng **makespan** (không phải cost sum) — đây mới là speedup có ý nghĩa.
- File CSV cũ (`results/mcrp_experiment_*.csv` sinh trước hôm nay) **không tương thích** với analyzer mới (thiếu cột `gap_makespan`, `makespan`...). Chỉ dùng với CSV sinh từ `experiment.py` sau khi đã sửa.

### 7. `baselines/multi_crane/multi_crane_baseline.py` — viết lại Experiment 3
- **Bug đã sửa:** code cũ (`MultiCraneBaseline.run`) tự chế một luật chọn destination riêng (gần nhất, thấp nhất) — **không phải** heuristic Lin2015/Kim2016/Leveling thật, dù tên biến gợi ý vậy.
- Giờ có `LinDest`, `KimDest`, `LevelingDest` — implement đúng luật `restricted` (chỉ relocate từ target stack, khớp giả định A3) lấy trực tiếp từ logic trong `baselines/lin2015.py`, `kim2016.py`, `leveling.py`, chạy qua `run_mc_heuristic_episode()` — **cùng driver, cùng MCEnv, cùng ZoneSplit** với ZeroShot, đảm bảo so sánh công bằng move-for-move.
- Class `MultiCraneBaseline` cũ **đã bị xoá** khỏi file. Nếu `compare_mc_baselines.py` hoặc script khác còn import nó, sẽ lỗi `ImportError` — cần sửa các nơi gọi sang dùng `run_mc_heuristic_episode` + 3 class Dest mới.

## Việc CHƯA làm (Phase C, D còn lại)

1. **Chưa có số liệu thật cho paper.** File `.tex` mới có các đoạn `%%FILL_...` (placeholder) ở: abstract, Table 1 (single-crane), Experiment 2/3 text+table, phần thống kê, Discussion, Conclusion. Cần chạy 3 script dưới đây rồi điền số thật vào, xoá hết `%%FILL_`.
2. **Script cũ chưa rà theo `compute_lb_mc` API mới:** `compare_mc_baselines.py`, `analysis/run_comprehensive.py`, `analysis/supplementary_analysis.py`, `analysis/fix_critical_issues.py`, `verify_paper.py` — grep `compute_lb_mc` để tìm và sửa `.item()` → `['work'].item()` / `['makespan'].item()`.
3. **`tests/test_analysis.py`, `tests/test_experiment.py`, `tests/test_strategies.py`, `tests/test_lowerbound_mc.py`, `tests/test_mc_instances.py`, `tests/test_mcenv.py`** đã sửa và pass (37/37 khi tôi dừng lại, trừ `test_mc_instances.py` cần dataset 70-file mới để pass — hiện dataset đã bị khôi phục về 140-file cũ nên **test này sẽ FAIL cho tới khi bạn tự quyết định chạy `generate_mc_instances.py` mới**).
4. **D2 (NP-hardness, encoder duplicate, "3×" wording...)** đã sửa trực tiếp trong bản `.tex` mới (viết lại from scratch, không phải patch).
5. **D3 (author info thật, GitHub link, CRediT...), D4 (compile + re-audit)** chưa làm.
6. Có vài file/script không phải do tôi tạo xuất hiện untracked trong `git status` (`analysis/phase1_shin_zo.py`, `analysis/run_all_comprehensive.py`, `analysis/run_missing.py`, `logs/*`, `results/phase1/`) — tôi **không đụng vào** vì không chắc chắn nguồn gốc, bạn kiểm tra lại xem có phải việc của bạn từ trước không.

## Cách tự chạy lại (theo đúng thứ tự)

```bash
# 0. Xác nhận trạng thái sạch trước khi bắt đầu
git status --short

# 1. (TÙY CHỌN, sẽ xoá 140 file _c2/_c3 cũ) Sinh lại dataset 70-layout format mới
#    Backup trước nếu muốn giữ cả 2:
#    cp -r benchmarks/mc_instances/lee_mc benchmarks/mc_instances/lee_mc.bak
python -m benchmarks.generate_mc_instances
python -m pytest tests/test_mc_instances.py -v   # phải pass 4/4 sau bước này

# 2. Chạy full test suite (kỳ vọng pass hết, đã pass 37/37 lúc tôi dừng)
python -m pytest tests/ -v

# 3. Experiment 1: single-crane, 70 instance chuẩn Lee, có seed cố định
python -m analysis.run_single_crane_v2
#    -> results/single_crane_v2.csv

# 4. Experiment 2: 4 strategy x 70 layout x C={2,3} = 560 runs
python experiment.py --cranes 2 3 --strategies S1 S2 S3 S4 --output results/mcrp_experiment_v2_main.csv

# 5. Experiment 3: M-Lin2015/M-Kim2016/M-Leveling vs ZeroShot+S2, cùng driver
python -m analysis.run_mc_baselines_v2
#    -> results/mc_baselines_v2.csv

# 6. Phân tích + figures
python -m analysis.analyze results/mcrp_experiment_v2_main.csv
python -m analysis.visualize_v2
#    -> results/analysis_report.txt, results/figures_v2/*.png

# 7. Điền số vào docs/latex/crp_rl_paper_Q1.tex (thay hết %%FILL_...)
#    rồi biên dịch:
cd docs/latex && pdflatex -interaction=nonstopmode crp_rl_paper_Q1.tex && pdflatex -interaction=nonstopmode crp_rl_paper_Q1.tex
```

Mọi con số trong `%%FILL_...` nên lấy từ 3 file CSV ở bước 3-5 (`single_crane_v2.csv`, `mcrp_experiment_v2_main.csv`, `mc_baselines_v2.csv`) qua `analysis/analyze.py`, không gõ tay từ trí nhớ.
