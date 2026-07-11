# PLAN NGHIÊN CỨU & THỰC NGHIỆM CUỐI CÙNG — M-CRP

**Ngày:** 2026-07-11. **Thay thế** `2026-07-11-q1-research-strategy.md` (v2) và
`2026-07-11-q1-hybrid-strategy.md` (v3). Plan này hợp nhất hai bản đó thành chiến
lược 2 lớp, với nguyên tắc: **bài nền luôn tồn tại, không có bước nào phải cầu may**.

## Chiến lược 2 lớp

- **Lớp 1 (đảm bảo có bài):** empirical/problem-extension paper từ dữ liệu ĐÃ CÓ —
  "First DRL for M-CRP via zero-shot transfer: exact fidelity, small-scale dominance,
  and the scale boundary where simple heuristics take over". Target: **C&OR / EAAI**
  (Q1). Mọi claim đều đã được dữ liệu chống lưng; chỉ cần hoàn tất dữ liệu + phân
  tích + viết lại.
- **Lớp 2 (nâng cấp nếu thành công):** HGP fusion (score DRL + penalty khoảng
  cách/chiều cao + zone mask, inference-only) — nếu thắng chặt cả ZS lẫn Leveling
  ở scale lớn thì nâng khung thành method paper, target **TR-C / TR-E**. Nếu thất
  bại → ghi nhận là ablation trong bài Lớp 1 (fallback_rate cao tự nó là finding).

Xác suất trung thực (một lần nộp đầu): Lớp 1 ~35–50% tại C&OR/EAAI; Lớp 1+2 thành
công ~25–40% tại TR-C, ~60–75% tại Q1 rộng. Tính cả chuỗi resubmit 12–18 tháng:
~70–85% có bài Q1. KHÔNG có kịch bản "chắc chắn".

---

## PHASE 0 — Hoàn tất nền dữ liệu (0.5 ngày việc + ~1 ngày máy)

| # | Việc | Lệnh / file | Output |
|---|---|---|---|
| 0.1 | Chạy nốt 26 instance Shin thiếu (6 R + 20 U nhóm 30-bay) | `python -m analysis.run_single_crane_full --dataset shin --workers 8` (resume tự động; 8 worker tránh contention đã đo) | `results/single_crane_shin.csv` đủ 160×6=960 dòng |
| 0.2 | Verify 4 CSV đủ: 420 / 960 / 2240 / 5120 dòng; chạy analyze + visualize | `python -m analysis.analyze ...`, `python -m analysis.visualize_v2` | báo cáo + figures nền |
| 0.3 | Đối chiếu cost ZeroShot với Table A.8 bài gốc per-instance (đã trích PDF) | script mới `analysis/compare_published.py` | bảng reproduction-match (bằng chứng rigor) |
| 0.4 | Commit + push toàn bộ code/docs hiện có (việc tồn đọng) | git — stage chọn lọc, không add results/ | repo sạch |

**Gate G0:** không viết bất kỳ kết luận scale-lớn nào trước khi 0.1 xong (nhóm
30×8 U-type hiện 0/20 — kết luận có thể đổi chiều).

## PHASE 1 — Phân tích cho bài nền (2–3 ngày, cần cho MỌI khung)

| # | Việc | Chi tiết |
|---|---|---|
| 1.1 | **Behavioral analysis (bằng chứng cơ chế — figure chính của paper)** | Script mới `analysis/behavioral_analysis.py`: per-step ghi bay-distance của destination, tỉ lệ cùng-bay, tỉ lệ cùng-zone cho ZS/Leveling/Lin trên cả 4 bộ. Vẽ: cross-bay-rate theo scale đặt cạnh gap theo scale → hai đường cong trùng pha = cơ chế được chứng minh định lượng |
| 1.2 | **Nâng cấp thống kê** | Wilcoxon + hiệu chỉnh Holm cho đa so sánh, Cohen's d, σ_d paired, win-rate cho mọi cặp chính. XOÁ power-analysis circular cũ |
| 1.3 | **Complementarity map** | Phương pháp nào thắng ở đâu: scale × type(R/U) × C × strategy. Đây là bảng động cơ cho Lớp 2 và là finding độc lập |
| 1.4 | **Xử lý LB makespan lỏng** | Báo cáo cả gap_work (bound chặt hơn) lẫn gap_makespan, kèm đoạn giải thích per-bay term không bind; mọi kết luận makespan dùng so sánh tương đối |

## PHASE 2 — HGP fusion: thí nghiệm nâng cấp (3–5 ngày, có go/no-go)

| # | Việc | Chi tiết |
|---|---|---|
| 2.1 | Implement fusion trong `policy/zero_shot.py::get_scores()` | `logits' = logits − λ·travel_norm − μ·height_norm`, thêm zone_mask option khi C>1. KHÔNG sửa trọng số → giữ claim training-free. Điểm chạm thứ 2: `engine/mcrp_inference.py` truyền zone |
| 2.2 | Protocol chống overfit | Tách validation 20%/scale (khai báo trong paper), tune (λ, μ) grid nhỏ trên val, test một lần trên phần còn lại |
| 2.3 | **Thí nghiệm quyết định (Gate G2)** | Chạy fusion trên đúng 2 chỗ đang thua: Shin 30×6 (vs Leveling 12.65%) và MC-large makespan (vs M-Leveling 357.9). GO nếu fusion ≤ Leveling có ý nghĩa thống kê ở cả 2; NO-GO nếu không |
| 2.4a | Nếu GO | Re-run full matrix 4 bộ với fusion (+ portfolio guard, log `fallback_rate`, `winner_component`); tận dụng record/replay + multiprocessing + resume sẵn có |
| 2.4b | Nếu NO-GO | Giữ kết quả như một ablation trung thực trong bài Lớp 1 ("distance-penalty correction không đủ khôi phục — gợi ý giới hạn nằm sâu hơn trong representation") — vẫn là nội dung dùng được |
| 2.5 | Ablation thật thay sensitivity giả | Sweep λ, μ, k, zone_mask on/off — thay thế đoạn sensitivity-không-có-data cũ trong .tex |

## PHASE 3 — Viết lại paper (1–2 tuần — phần tốn thời gian thật)

**3.1 Chọn khung theo Gate G2:**
- GO → method paper: "Hybrid inference-time correction cho zero-shot M-CRP" → TR-C/TR-E
- NO-GO → empirical study: "Zero-shot DRL transfer to M-CRP: scale boundaries" → C&OR/EAAI

**3.2 Cấu trúc RQ (khớp phong cách bài gốc):**
- RQ1: Fidelity — trích xuất zero-shot có tái tạo đúng SOTA tại C=1? (0.0000%)
- RQ2: ZS vs heuristics ở scale nhỏ-vừa, cả 2 setting (thắng áp đảo)
- RQ3: Ranh giới scale — ở đâu, và cơ chế gì (behavioral 1.1 + đối chiếu Section 5.2.5 bài gốc)
- RQ4: [GO] fusion khôi phục được không / [NO-GO] correction đơn giản có đủ không (ablation 2.4b)
- RQ5: Chiến lược gán crane S1–S4 + interference/A7 + speedup C=2→3 (báo trung thực 3–11%)

**3.3 Checklist sửa .tex bắt buộc (từ audit — mọi mục phải FULLY_ADDRESSED):**
- [ ] Theorem 3 viết lại theo `bounds/lowerbound_mc.py` hiện tại (max(work/C, per-bay)); nêu điều kiện t_pd
- [ ] Objective: makespan chính, total work phụ (khớp code)
- [ ] Dataset: 70 unique layouts (không phải 140); mô tả header crane_start_bays
- [ ] Mô tả A6=delay, A7=reassign đúng semantics mcenv.py hiện tại; bỏ "S2 enforces by construction"
- [ ] Bỏ claim "ZeroShot tốt hơn Original" (artifact đã fix — giờ là ≡ 0.0000%)
- [ ] Sửa 6 lỗi bibliography (forster=tree search, kim2016 CIE, lee2010 COR, caserta EJOR, cifuentes ASC, durasevic EAAI) + chuyển .bib
- [ ] Xoá đoạn encoder duplicate; sửa NP-hardness thành embedding argument
- [ ] Sửa misquote bài gốc: 70 Lee instances, 7.8% greedy trên 50 R-type, TS 44–108%, GRASP 37–67%
- [ ] Mọi số liệu từ 4 CSV mới; xoá 6.56/6.03/7.06/10.46 và mọi số cũ
- [ ] Sensitivity = sweep thật từ 2.5
- [ ] Limitations khai báo: sequential simulator (A3 global order), LB makespan lỏng, không TS/GRASP/GP, không exact baseline, offline setting

**3.4 Figures:** visualize_v2 + behavioral figures (1.1) + complementarity heatmap (1.3), 300 DPI.

## PHASE 4 — Gate chất lượng trước nộp (2–3 ngày)

- [ ] `python -m pytest tests/ -v` sạch; test mới: gap≥0 toàn CSV, backward-compat 0.0000%, [GO] HGP ≤ min components toàn CSV
- [ ] Re-audit: chạy paper-audit re-audit mode đối chiếu báo cáo cũ trong `review_results/`
- [ ] Self-review 3 persona + citation-check
- [ ] Cover letter nêu thẳng quan hệ với bài gốc (extend future work Section 6 của họ; lấp khoảng trống so sánh Leveling-ở-scale-lớn của họ)

## Tổng thời gian & mốc quyết định

| Phase | Thời gian | Gate |
|---|---|---|
| 0 | 0.5 ngày việc + ~1 ngày máy | G0: dữ liệu đủ 100% |
| 1 | 2–3 ngày | — |
| 2 | 3–5 ngày | **G2: GO/NO-GO quyết định khung + venue** |
| 3 | 1–2 tuần | — |
| 4 | 2–3 ngày | G4: mọi audit item FULLY_ADDRESSED |
| **Tổng** | **~3–4.5 tuần** | |

## Ngoài phạm vi (khai báo là limitation, KHÔNG làm trong vòng này)

- Viết lại simulator thành event-driven concurrent thật (trả lời triệt để phê
  bình MCSP) — đây là paper tiếp theo, không nhét vào vòng này
- Reimplement TS/GRASP/GP, exact solver trên instance nhỏ — chỉ làm nếu reviewer
  đòi ở vòng revision
- Fine-tune policy — chỉ cân nhắc nếu TR-C revision yêu cầu "learning contribution"

## Rủi ro chính

| Rủi ro | Đối sách |
|---|---|
| 26 instance Shin mới đổi chiều kết luận 30×8 | G0 chặn viết trước khi có đủ; nếu đổi chiều thì đó là data — cập nhật claim |
| G2 NO-GO | Đã có đường 2.4b; bài Lớp 1 không phụ thuộc G2 |
| Reviewer MCSP đánh sequential simulator | Limitations khai báo thẳng + trích A3 của chính bài gốc làm cơ sở giả định |
| Viết lại .tex lâu hơn dự kiến | Checklist 3.3 đóng khung phạm vi; không thêm nội dung ngoài RQ đã chốt |
