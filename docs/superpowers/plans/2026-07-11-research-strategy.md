# Chiến lược nghiên cứu (Research Strategy) — M-CRP Zero-shot Transfer

**Ngày:** 2026-07-11. Đây là **tầng chiến lược** (câu hỏi nghiên cứu, thesis,
đóng góp, claim, định vị so với bài gốc, rủi ro). Tầng **code/thực nghiệm** (script
nào chạy khi nào, file nào sửa, lệnh gì) là việc phái sinh từ tài liệu này — bạn
tự dựng theo nhu cầu, tham khảo `2026-07-11-final-research-plan.md` (phase-by-phase
chi tiết) nếu cần một bản đã có sẵn.

Tài liệu này thay thế phần "chiến lược" rải rác trong `2026-07-11-q1-research-strategy.md`
(v2) và `2026-07-11-q1-hybrid-strategy.md` (v3) — hai file đó giữ lại làm phụ lục
kỹ thuật (v2: chi tiết cơ chế SLC; v3: chi tiết công thức fusion + Mệnh đề 1),
nhưng quyết định chiến lược nằm ở đây.

---

## 1. Định vị nghiên cứu (research gap)

**Bài gốc** (Shin et al. 2026, TR-C): DRL single-crane CRP, SOTA đã verify, tự
xác nhận 2 giới hạn của chính họ:
- Multi-crane là **future work chưa làm** (Section 6).
- Ở scale cực lớn (offline, full-information) họ **chỉ so với GP/Lin, chưa từng
  so với Leveling**; ở online setting họ tự ghi nhận lợi thế so với Leveling
  **thu hẹp ở scale lớn** (Section 5.2.5).

**Khoảng trống cụ thể, có thể lấp:** chưa có nghiên cứu nào (kể cả bài gốc) trả
lời "policy DRL single-crane đã train có transfer được sang multi-crane không,
và transfer đó — nếu có — hoạt động tốt đến scale nào, vì cơ chế gì". Đây là câu
hỏi nghiên cứu của bạn, không phải một câu hỏi bạn tự đặt ra để né tránh so sánh
khó — nó nối tiếp trực tiếp giới hạn bài gốc tự nêu.

## 2. Câu hỏi nghiên cứu trung tâm

> **RQ chính:** Một policy DRL single-crane, không huấn luyện lại, transfer sang
> multi-crane container retrieval hiệu quả đến đâu — và điều gì quyết định ranh
> giới đó?

Ba câu hỏi con, mỗi câu ánh xạ tới một mảng bằng chứng bạn **đã có**:
- RQ-A (fidelity): trích xuất zero-shot có trung thành với model gốc? → đã trả lời: có, 0.0000%.
- RQ-B (hiệu năng theo scale): zero-shot so với heuristic multi-crane thế nào theo scale? → đã trả lời: thắng áp đảo scale nhỏ, thua ở scale lớn (2 setting độc lập).
- RQ-C (cơ chế): tại sao lại có ranh giới đó? → giả thuyết đã có (mất tính cục bộ không gian), cần bằng chứng định lượng (behavioral analysis).

Một câu hỏi con thứ tư, **có điều kiện** (chỉ theo đuổi nếu muốn nâng cấp venue):
- RQ-D (khắc phục): ranh giới ở RQ-B có khắc phục được bằng hiệu chỉnh inference-time không cần train lại?

## 3. Thesis (một câu, cho từng kịch bản)

- **Nếu chỉ trả lời RQ-A/B/C** (kịch bản nền, không phụ thuộc thí nghiệm mới):
  *"Zero-shot transfer của policy DRL single-crane sang multi-crane là khả thi và
  vượt trội heuristic ở scale thực tế phổ biến, nhưng có một ranh giới scale rõ
  ràng, đo lường được, và giải thích được bằng cơ chế mất tính cục bộ không gian
  — nhất quán độc lập ở cả single- và multi-crane."*
- **Nếu RQ-D thành công** (kịch bản nâng cấp):
  *"...và ranh giới đó khôi phục được hoàn toàn bằng hiệu chỉnh tại thời điểm suy
  luận, không cần huấn luyện lại — biến zero-shot transfer thành một giải pháp
  multi-crane thực dụng ở mọi scale."*

Chọn thesis nào để viết paper là quyết định **sau khi có bằng chứng RQ-D**, không
phải trước — đây chính là gate chiến lược (§6).

## 4. Đóng góp (contribution set) — tách theo mức độ chắc chắn

| # | Đóng góp | Phụ thuộc thí nghiệm mới? | Mức độ chắc chắn |
|---|---|---|---|
| C1 | Người đầu tiên đưa DRL (qua zero-shot) vào M-CRP — lấp đúng future-work bài gốc tự nêu | Không | Chắc chắn — đã có |
| C2 | Backward-compatibility chính xác tuyệt đối (0.0000%), không phải "gần đúng" như hầu hết claim zero-shot-transfer khác | Không | Chắc chắn — đã có |
| C3 | Mô hình chi phí/lower-bound multi-crane hợp lệ (đã sửa 2 lỗi toán học từ bản trước) + timing model đủ để đo makespan thật | Không | Chắc chắn — đã có |
| C4 | Đặc trưng hoá ranh giới scale của zero-shot transfer + cơ chế nguyên nhân (mất cục bộ không gian), xác nhận chéo với dữ liệu online của chính bài gốc | Cần Phase phân tích (behavioral), không cần thí nghiệm mới rủi ro | Cao — chỉ cần phân tích thêm dữ liệu đã có |
| C5 (tuỳ chọn) | Cơ chế khắc phục ranh giới bằng hiệu chỉnh inference-time, không train lại | Cần thí nghiệm mới, có gate go/no-go | Không chắc chắn — 50-70% |

**Nguyên tắc chọn đóng góp cho bản nộp đầu tiên:** C1–C4 đủ để tạo một bài Q1
hoàn chỉnh, không phụ thuộc bất kỳ kết quả thí nghiệm chưa biết nào. C5 là phần
thưởng thêm, không phải điều kiện.

## 5. Định vị so với bài gốc (novelty framing)

Nguyên tắc: **không cạnh tranh trên sân của họ** (single-crane, đã ≡ chính họ),
mà **mở rộng sân** (multi-crane) và **lấp khoảng trống đánh giá họ tự để lại**
(Leveling ở scale lớn). Bảng đối chiếu dùng khi viết Related Work / Discussion:

| Khía cạnh | Bài gốc | Bài này |
|---|---|---|
| Phạm vi | Single-crane | Multi-crane (C1) |
| So sánh ở scale cực lớn | Chỉ GP, Lin | + Leveling, + đa chiến lược gán crane |
| Objective multi-crane | Không có | Makespan hợp lệ (C3) |
| Ranh giới scale | Nêu ở online, không đo ở offline | Đo, lý giải cơ chế, cả 2 setting (C4) |
| Xác nhận zero-shot | Không áp dụng (không phải bài zero-shot) | Chính xác tuyệt đối (C2), hiếm trong literature transfer learning |

## 6. Chiến lược 2 lớp + gate quyết định

- **Lớp nền (không điều kiện):** viết bài từ C1–C4. Target: **C&OR / EAAI /
  ESWA** (Q1, không phải flagship transportation). Không phụ thuộc bất kỳ gate nào.
- **Lớp nâng cấp (có điều kiện, một gate duy nhất — "Gate G"):** thử nghiệm cơ
  chế khắc phục (C5). Chạy **trước khi cam kết khung paper cuối**, trên đúng 2
  điểm dữ liệu đang thua (Shin 30×6 single-crane; multi-crane-large makespan).
  - **G = GO** (cơ chế khắc phục thắng cả 2 endpoint có ý nghĩa thống kê) → nâng
    khung thành method paper, target **TR-C / TR-E**.
  - **G = NO-GO** → giữ khung Lớp nền; kết quả thất bại của cơ chế khắc phục vẫn
    dùng được (thêm 1 ablation, củng cố C4: ranh giới sâu hơn một correction đơn giản).

Đây là **gate duy nhất** trong toàn chiến lược. Không có gate nào khác quyết
định "có bài hay không" — chỉ gate này quyết định "bài mạnh cỡ nào / venue nào".

## 7. Ma trận claim — cái được nói, cái không được nói

| Được claim (có bằng chứng) | Không được claim (không có bằng chứng / sai) |
|---|---|
| Zero-shot ≡ SOTA gốc tại C=1, chính xác | Zero-shot "tốt hơn" SOTA gốc |
| Zero-shot thắng áp đảo heuristic ở scale nhỏ-vừa, cả 2 setting | Zero-shot thắng ở mọi scale |
| Multi-crane qua zero-shot khả thi, chi phí train = 0 | Đã giải multi-crane scheduling thật (MCSP) |
| Ranh giới scale đo được, có cơ chế giải thích | Ranh giới đã "khắc phục hoàn toàn" (trừ khi Gate G = GO và verify) |
| So sánh tương đối gap_makespan giữa các method hợp lệ | Gap_makespan tuyệt đối (200-600%) phản ánh "tệ gấp X lần" — LB lỏng |
| Tốc độ song song hoá C=2→3 khiêm tốn (3–11%), báo trung thực | "Multi-crane scaling tốt" |

## 8. Đánh giá rủi ro trung thực (không tô hồng)

- **Không có kế hoạch nào "chắc chắn" Q1** — quyết định thuộc về reviewer ẩn danh,
  không phải chất lượng thực nghiệm một mình quyết định.
- Xác suất ước lượng (một lần nộp, sau khi hoàn thành Lớp nền + thử Gate G):
  - Lớp nền riêng, tại C&OR/EAAI: ~35–50%.
  - Nếu Gate G = GO, tại TR-C: ~25–40%; tại Q1 rộng: ~60–75%.
  - Tính cả chuỗi resubmit 12–18 tháng (đổi venue nếu bị từ chối): ~70–85% có
    một bài Q1 nào đó.
- Rủi ro cấu trúc không tự sửa được trong ngắn hạn: mô phỏng vẫn tuần tự-có-timing
  (không phải concurrent event-driven thật) — nếu reviewer từ cộng đồng MCSP
  chấm, đây là điểm sẽ bị hỏi; xử lý bằng khai báo Limitations thẳng thắn, không
  che giấu, không hứa hẹn quá mức ở Discussion.
- Rủi ro dữ liệu: kết luận scale cực lớn (30×8) hiện dựa trên mẫu chưa đầy đủ
  (thiếu 26/160 instance) — bắt buộc hoàn tất trước khi khoá bất kỳ con số nào
  vào bản thảo.

## 9. Tiêu chí thành công của TỪNG lớp (để tự chấm trước khi nộp)

- Lớp nền: mọi claim trong §7 cột trái đều verify được bằng một câu lệnh chạy
  lại trên CSV kết quả; không còn số liệu nào trong bản thảo không truy được
  nguồn; toàn bộ audit findings trước đó (bibliography, Theorem 3, dataset
  count...) đã FULLY_ADDRESSED.
- Lớp nâng cấp: Gate G có kết quả rõ ràng (không mập mờ) trước khi viết Discussion
  cuối cùng — không được vừa viết vừa chờ kết quả.

## 10. Việc cần làm để chuyển sang tầng code/thực nghiệm

Từ đây, chiến lược code/thực nghiệm cụ thể (script, hàm, lệnh chạy, thứ tự) là
việc bạn tự dựng hoặc yêu cầu riêng — nó cần trả lời tối thiểu các câu hỏi do
chiến lược này đặt ra:
1. Làm sao lấp nốt dữ liệu thiếu (26 instance) mà không phá vỡ CSV đã có?
2. Behavioral analysis (C4) cần đo gì per-step, trên toàn bộ 4 bộ dữ liệu?
3. Cơ chế khắc phục (C5, nếu thử Gate G) cắm vào đâu trong pipeline hiện có mà
   không sửa trọng số model, và tách validation/test thế nào để không bị nghi
   overfit?
4. Bản .tex cần đồng bộ với danh sách nào (đối chiếu checklist audit đã có)?

`2026-07-11-final-research-plan.md` đã có một bản trả lời chi tiết cho 4 câu
trên (phase 0–4, file:line cụ thể) — dùng làm điểm khởi đầu, điều chỉnh theo
đúng gate/thesis bạn chọn ở §6 của tài liệu này.
