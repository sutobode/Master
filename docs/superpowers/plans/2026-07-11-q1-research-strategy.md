# Chiến lược nghiên cứu Q1 (v2) — M-CRP Zero-shot Transfer

**Ngày:** 2026-07-11
**Trạng thái bằng chứng làm nền:** Setting A nhỏ (70/70), A lớn (134/160 — thiếu 26 instance nhóm 30-bay), B nhỏ (70/70), B lớn (160/160). Backward-compat 0.0000% ở mọi scale.

---

## 1. Chẩn đoán: vì sao bản hiện tại chưa đủ Q1 (TR-C)

Phát hiện xuyên suốt 4 bộ thực nghiệm: ZeroShot thắng áp đảo ở scale nhỏ nhưng
**thua heuristic Leveling khi scale đủ lớn**, ở CẢ hai setting độc lập:

| Bằng chứng | Số liệu |
|---|---|
| Single-crane, Shin 30×6 (2160 cont.) | ZS 14.36% vs Leveling 12.65%, ZS chỉ thắng 14/40 instance |
| Multi-crane large, makespan (best strategy) | M-Leveling 357.9 < ZS 375.7, Wilcoxon p≈3e-13, ZS thắng 31% |
| Multi-crane large, work | ZS thắng sát nút (p=0.0043, 52.5% win) |

Cơ chế (đã đối chiếu code): policy chấm điểm **argmax toàn cục** trên mọi stack
(`policy/zero_shot.py:64-68`), không có ràng buộc cục bộ không gian; Leveling có
ràng buộc cứng **ưu tiên cùng-bay** (`baselines/leveling.py:79-92`). Ở yard nhỏ,
"tốt nhất toàn cục" hiếm khi ở xa; ở yard 20–30 bay, lựa chọn toàn cục trả giá
di chuyển lớn — và phân bố scale này nằm ngoài phân bố huấn luyện của model gốc
(OOD). Hệ quả kép: (i) suy giảm gốc ở single-crane scale lớn; (ii) cộng hưởng
thêm ở multi-crane vì destination lệch zone phá phối hợp crane.

Điểm tựa từ chính bài gốc (Shin et al. 2026, TR-C):
- Section 5.2.5: chính họ ghi nhận "gap so với leveling rõ hơn ở scale nhỏ" (online setting) — xu hướng ta phát hiện KHỚP với dữ liệu của họ.
- Section 5.2.4 / Table 5: ở benchmark cực lớn họ CHỈ so với GP và Lin, **chưa từng so với Leveling** — khoảng trống đánh giá thật.
- Section 6: multi-crane là future work họ tự nêu.

→ Kết quả hiện tại là một **negative result có cơ chế rõ** — đủ cho Q1 nhánh rộng
(C&OR/EAAI/ESWA) nếu trình bày trung thực, nhưng **chưa đủ cho TR-C** vì thiếu một
đóng góp phương pháp khắc phục.

## 2. Thesis mới (một câu)

> *Zero-shot transfer của policy DRL single-crane sang multi-crane thất bại ở
> scale lớn vì mất tính cục bộ không gian — và có thể khôi phục hoàn toàn bằng
> các hiệu chỉnh tại thời điểm suy luận (inference-time), không cần huấn luyện lại.*

Người phản biện sẽ hỏi: "nếu inference-time correction đủ tốt, sao không dùng
luôn Leveling?" — Trả lời bằng số: ZS+correction phải thắng Leveling ở MỌI scale
(kế thừa chất lượng xếp chồng của DRL ở scale nhỏ + tính cục bộ ở scale lớn),
điều Leveling một mình không làm được (thua 6-16pp ở scale nhỏ).

## 3. Ba đóng góp mới (bổ sung vào 3 đóng góp cũ)

**N1 — Spatial Locality Correction (SLC), training-free.** Họ các hiệu chỉnh
inference-time trên policy đóng băng:
- (a) *Zone masking* (multi-crane): mask destination ngoài zone của crane phụ
  trách target; fallback toàn cục khi zone hết chỗ (mirror logic fallback của
  LevelingDest). Dùng cơ chế `invalid_mask` sẵn có — không sửa model.
- (b) *Distance-penalized scoring* (cả hai setting): `logits' = logits − λ·travel_time(target, dest)`
  (chuẩn hoá), λ chọn trên tập validation tách riêng (không đụng test).
- (c) *k-nearest-bay candidate restriction*: chỉ chấm điểm các stack trong k bay
  gần nhất; k lớn → hồi quy về bản gốc (tham số nội suy giữa "toàn cục" và "cục bộ").
- Kết hợp a+b/a+c và ablation từng thành phần.

**N2 — Ranh giới scale của zero-shot transfer, đo lường được.** Behavioral
analysis: phân bố khoảng cách bay của lựa chọn destination (ZS vs Leveling) theo
scale; tỉ lệ move cùng-bay/cùng-zone; hồi quy gap ~ cross-bay-rate. Biến lời giải
thích thành bằng chứng định lượng. Đối chiếu chéo với xu hướng online của chính
Shin et al. (5.2.5).

**N3 — Chi phí của zero-shot (fine-tune ablation, gọn).** Fine-tune policy trên
rollout multi-crane/scale-lớn với budget nhỏ (vd. 5–10 epoch, chỉ scorer hoặc
LoRA-style trên decoder weights) để định vị: zero-shot / zero-shot+SLC /
fine-tuned trên trục "chi phí huấn luyện ↔ chất lượng". Đây là mảnh "learning
contribution" reviewer TR-C sẽ đòi. (Rủi ro cao nhất — xem Phase D; có thể bỏ
nếu N1 đã đủ mạnh, hạ target xuống C&OR vẫn ổn.)

Đóng góp giữ nguyên từ bản cũ: (C1) hiện thực hoá future-work multi-crane của
bài gốc, zero-training; (C2) MCEnv timing model + LB hợp lệ + backward-compat
0.0000%; (C3) khảo sát chiến lược gán crane S1–S4.

## 4. Kế hoạch thực thi theo phase

### Phase A — Hoàn tất nền dữ liệu (0.5–1 ngày máy chạy)
- A1. Resume 26 instance Shin còn thiếu (`python -m analysis.run_single_crane_full --dataset shin`) — bắt buộc trước khi viết bất kỳ kết luận scale-lớn nào.
- A2. Đối chiếu cost ZeroShot/Original với Table A.8 của bài gốc trên từng instance trùng — bằng chứng reproduction khớp số công bố (rigor hiếm có).

### Phase B — SLC implementation + thí nghiệm quyết định (2–4 ngày)
- B1. Implement 3 biến thể SLC (zone mask / distance penalty / k-nearest). Điểm chạm: `policy/zero_shot.py` (thêm tham số mask/penalty vào `get_scores`), `engine/mcrp_inference.py` (truyền zone của crane được strategy chọn). KHÔNG sửa trọng số model — giữ nguyên claim zero-shot.
- B2. Chọn λ, k trên validation: tách 20% instance mỗi scale làm validation, phần còn lại là test (tránh tuning-on-test — reviewer sẽ soi).
- B3. **Thí nghiệm quyết định (go/no-go):** ZS+SLC vs Leveling trên Shin 30×6 (chỗ đang thua nặng nhất, single-crane) và multi-crane large makespan. 
  - Nếu ZS+SLC ≥ Leveling ở mọi scale → giữ thesis mục 2, target TR-C/TR-E.
  - Nếu chỉ thu hẹp mà không đảo được → chuyển khung "boundary study + partial mitigation", target C&OR/EAAI (vẫn Q1).
- B4. Re-run full matrix với biến thể SLC thắng cuộc (tận dụng record/replay + multiprocessing + resume sẵn có; ước tính tương đương 1 vòng chạy full hiện tại).

### Phase C — Behavioral analysis N2 (1–2 ngày)
- C1. Script trích per-step: khoảng cách bay của destination, cùng-bay?, cùng-zone?, cho ZS / ZS+SLC / Leveling / Lin trên toàn bộ instance.
- C2. Figure chính của paper: cross-bay-rate và mean bay-distance theo scale, cạnh gap theo scale — cho thấy đường cong "mất cục bộ" trùng pha với đường cong "mất lợi thế".
- C3. Wilcoxon + Cohen's d + σ_d cho mọi so sánh cặp chính (thay power-analysis circular cũ).

### Phase D — Fine-tune ablation N3 (3–5 ngày, TUỲ CHỌN theo kết quả B3)
- D1. Fine-tune scorer trên instance lớn (train split riêng, không đụng benchmark test), REINFORCE tiếp trên `trainer.py` sẵn có.
- D2. Bảng 3 cột: ZeroShot / +SLC / +fine-tune — chi phí (giờ GPU/CPU) vs gap từng scale.

### Phase E — Viết lại paper (4–6 ngày)
- E1. Đồng bộ toàn bộ .tex với code/data đã fix (Theorem 3 mới, makespan objective, 70 unique layouts, bỏ sensitivity không có data, sửa 6 lỗi bibliography, bỏ đoạn encoder duplicate, sửa NP-hardness embedding argument) — theo checklist trong `2026-07-10-revision-handoff.md` + bản audit.
- E2. Cấu trúc lại Experiments theo RQ (mirror phong cách bài gốc):
  - RQ1: Fidelity — zero-shot có tái tạo đúng SOTA ở C=1? (0.0000%)
  - RQ2: Small-scale — ZS vs heuristics, cả 2 setting (thắng áp đảo)
  - RQ3: Ranh giới scale — ở đâu và vì sao lợi thế đảo chiều (N2)
  - RQ4: SLC khôi phục được không? (N1 — kết quả B3/B4)
  - RQ5: Zero-shot vs fine-tune trade-off (N3, nếu làm)
- E3. Discussion: đóng khung phát hiện scale-boundary như finding chính, trích 5.2.5 và Section 6 của bài gốc làm điểm tựa; caveat LB-makespan lỏng (so sánh tương đối hợp lệ, tuyệt đối thì không).

### Phase F — Gate trước submit
- F1. `python -m pytest tests/ -v` sạch; assert gap≥0 toàn bộ; backward-compat 0.0000%.
- F2. Chạy lại paper-audit (re-audit mode) đối chiếu báo cáo cũ — mọi finding FATAL/MAJOR phải FULLY_ADDRESSED.
- F3. Self-review 3 persona (skill self-review) trước khi nộp.

## 5. Venue ladder (quyết định sau B3)

| Kịch bản kết quả B3 | Venue chính | Dự phòng |
|---|---|---|
| SLC đảo chiều hoàn toàn + có N3 | **TR-C** | TR-E |
| SLC đảo chiều, không N3 | TR-E / **C&OR** | EAAI |
| SLC chỉ thu hẹp | **C&OR** (khung boundary-study) | EAAI, ESWA |

Tổng thời gian ước tính: 2–3 tuần (không tính thời gian máy chạy thực nghiệm,
đã có resume + parallel + replay tối ưu sẵn).

## 6. Rủi ro & đối sách

| Rủi ro | Đối sách |
|---|---|
| SLC không đảo được ở 30×6 | Khung boundary-study vẫn là Q1 hợp lệ (C&OR); N2 trở thành đóng góp chính |
| Fine-tune không hội tụ (bài gốc từng báo scale-diverse ablation fail) | N3 là optional; báo cáo trung thực như thêm một bằng chứng khó |
| λ/k tuning bị nghi overfit | Tách validation/test từ đầu, khai báo protocol trong paper |
| Reviewer TR-C đòi parallel/online setting | Ghi rõ trong Limitations; MCEnv timing model đã event-based đủ đo makespan |
| 26 instance Shin thiếu đổi chiều kết luận 30×8 | Chạy A1 TRƯỚC khi viết — không viết kết luận trên mẫu thiếu |

## 7. INSIGHT collection (từ dữ liệu thật, dùng khi viết)

- I1: Backward-compat exact (0.0000%) ở mọi scale — trụ validity, giữ nguyên.
- I2: Lợi thế ZS co lại đơn điệu theo scale ở cả 2 setting → hiện tượng hệ thống, không phải nhiễu.
- I3: 30×6 là điểm đảo chiều single-crane hiện quan sát được (14/40 win vs Leveling).
- I4: Multi-crane: cùng phát hiện lặp lại trên makespan (p≈3e-13) — hai bằng chứng độc lập cùng cơ chế.
- I5: S2/S4 (spatial) > S1/S3 (non-spatial) nhất quán mọi scale → tín hiệu sớm rằng "không gian" là biến quyết định, nhất quán với chẩn đoán SLC.
- I6: Speedup C=2→3 chỉ 3–11% — A6/A7 giới hạn song song hoá; báo cáo trung thực, đừng bán "multi-crane scaling".
- I7: LB makespan = work/C trên mọi instance test (per-bay term không bind) — hợp lệ nhưng lỏng; chỉ dùng so sánh tương đối.
