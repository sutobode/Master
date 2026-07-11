# MASTER PLAN — M-CRP Q1 Strategy (bản chốt trước khi code)

**Ngày:** 2026-07-11. Đây là bản **CHỐT**, hợp nhất và thay thế toàn bộ 4 file
plan trước đó trong phiên này (`2026-07-11-q1-research-strategy.md`,
`2026-07-11-q1-hybrid-strategy.md`, `2026-07-11-final-research-plan.md`,
`2026-07-11-research-strategy.md`) — giữ lại 3 file cũ làm phụ lục kỹ thuật
(công thức, cơ chế chi tiết), nhưng **mọi quyết định thực thi tra ở đây**.

**Chưa code gì cả.** Đây là điểm dừng để duyệt trước khi bắt tay vào Phase 0.

---

## 1. Nguyên tắc chỉ đạo (bất biến, không đổi theo kết quả thí nghiệm)

1. **Không bước nào được phép là điều kiện để "có bài".** Luôn có một con đường
   dùng dữ liệu đã có, không phụ thuộc bất kỳ thí nghiệm mới nào thành công hay không.
2. **Mọi thí nghiệm mới có time-box và tiêu chí go/no-go chốt TRƯỚC khi chạy** —
   không nhìn số ra rồi mới quyết định "thế nào là thắng".
3. **Không tô hồng, không giấu giới hạn.** Claim trong paper phải khớp chính xác
   với bằng chứng — kể cả khi bằng chứng không đẹp.
4. **Việc rẻ/an toàn làm trước, việc đắt/rủi ro làm sau và có cổng kiểm soát.**

## 2. Câu hỏi nghiên cứu & thesis (không đổi bởi kết quả HGP)

**RQ chính:** Một policy DRL single-crane, không train lại, transfer sang
multi-crane hiệu quả đến đâu, và điều gì quyết định ranh giới đó?

- RQ-A Fidelity: đã trả lời (0.0000%, không phụ thuộc thí nghiệm mới)
- RQ-B Hiệu năng theo scale: đã trả lời (thắng nhỏ, thua lớn, cả 2 setting)
- RQ-C Cơ chế: giả thuyết có, cần bằng chứng định lượng (behavioral analysis — rẻ, không rủi ro)
- RQ-D Khắc phục (HGP): **có điều kiện**, cổng go/no-go ở §5

## 3. Đóng góp (contribution set) — không đổi bởi HGP

| # | Đóng góp | Trạng thái |
|---|---|---|
| C1 | Đầu tiên đưa DRL vào M-CRP qua zero-shot, lấp future-work bài gốc | Đã có |
| C2 | Backward-compat chính xác tuyệt đối (0.0000%) | Đã có |
| C3 | Lower-bound + timing model multi-crane hợp lệ | Đã có |
| C4 | Ranh giới scale + cơ chế (mất tính cục bộ không gian), đối chiếu chéo với chính dữ liệu online của bài gốc | Cần Phase 1 (phân tích, không rủi ro) |
| C5 | Cơ chế khắc phục ranh giới (HGP fusion), inference-time, không train lại | **Có điều kiện — Gate G** |

C1–C4 = bài nền, KHÔNG phụ thuộc C5.

## 4. Trình tự thực thi (thứ tự bắt buộc — việc rẻ/chắc trước)

```
PHASE 0 (nền dữ liệu, ~1-2 ngày, không rủi ro)
  0.1 Resume 26 instance Shin còn thiếu
  0.2 Verify 4 CSV đủ dòng, chạy analyze + visualize
  0.3 Đối chiếu cost với Table A.8 bài gốc (bằng chứng reproduction)
  0.4 Commit/push tồn đọng
        |
        v
PHASE 1 (phân tích cho bài nền, ~2-3 ngày, không rủi ro — làm bất kể HGP ra sao)
  1.1 Behavioral analysis: bay-distance / same-bay-rate / same-zone-rate theo
      scale cho ZS và Leveling -> figure cơ chế chính (bằng chứng C4)
  1.2 Thống kê: Wilcoxon + Holm correction, Cohen's d, sigma_d cho mọi cặp chính
  1.3 Complementarity map (method thắng ở đâu: scale x type x C x strategy)
        |
        v
PHASE 2 (HGP fusion — THÍ NGHIỆM RẺ TRƯỚC, ~1 ngày, có Gate G)
  2.1 CHỈ code Tầng 1 (fusion). Thêm 1 biến thể endpoint ĐÚNG cho Leveling
      (bộ lọc cứng same-bay trước, không chỉ phạt mềm theo chiều cao -- xem
      §6 lỗ hổng kỹ thuật) để "chứa cả 2 endpoint" là claim đúng, không chỉ
      gần đúng.
  2.2 Tune (lambda, mu) trên validation tách riêng (20%/scale)
  2.3 Chạy THỬ (không full matrix) trên 2 điểm quyết định:
      - Shin 30x6 single-crane (~40 instance, đang thua Leveling 14/40)
      - MC-large C=2, best strategy, makespan (đang thua M-Leveling, p~3e-13)
  2.4 GATE G (tiêu chí chốt trước, xem §5) -> GO hoặc NO-GO
        |
        +-- GO ------------------------> PHASE 2B (mở rộng)
        |                                  - Build Tang 3 (portfolio guard,
        |                                    gần miễn phí, dùng CSV đã có)
        |                                  - Cân nhắc Tang 2 (lookahead) nếu
        |                                    cần thêm lực
        |                                  - Re-run full matrix 4 bộ
        |                                  - Khung paper -> method paper (TR-C/TR-E)
        |
        +-- NO-GO ---------------------> Ghi nhận là 1 ablation trong bài nền
                                           ("distance/height correction đơn giản
                                           không đủ -- ranh giới sâu hơn representation")
                                           Khung paper -> empirical study (C&OR/EAAI)
        |
        v (cả 2 nhánh hội tụ về đây)
PHASE 3 (viết lại paper, ~1-2 tuần)
  3.1 Chọn khung theo kết quả Gate G
  3.2 Cấu trúc theo RQ (A/B/C bắt buộc, D theo kết quả G)
  3.3 Checklist sửa .tex (Theorem 3, bibliography, dataset count, số liệu mới,
      bỏ claim sai, Limitations trung thực) -- xem final-research-plan.md §3.3
      cho danh sách đầy đủ
        |
        v
PHASE 4 (gate chất lượng, ~2-3 ngày)
  4.1 pytest sạch + test mới (gap>=0, backward-compat, [nếu GO] Menh de 1
      verify trên toàn CSV)
  4.2 Re-audit (paper-audit re-audit mode)
  4.3 Self-review 3 persona + citation-check
```

**Điểm mấu chốt của trình tự này:** Phase 0–1 luôn chạy, không phụ thuộc Phase
2. Phase 2 chỉ tốn ~1 ngày để có câu trả lời GO/NO-GO (không phải 3-5 ngày build
đầy đủ 3 tầng rồi mới biết) — tối thiểu hoá rủi ro thời gian trước khi biết
hướng đi có đáng đầu tư tiếp không.

## 5. Gate G — tiêu chí chốt TRƯỚC khi chạy (không đổi sau khi thấy số)

**GO nếu ĐỒNG THỜI:**
- Fusion thắng Leveling có ý nghĩa thống kê (Wilcoxon p<0.05, Holm-corrected)
  trên CẢ HAI điểm quyết định (Shin 30×6 và MC-large C=2 makespan)
- Fusion không làm giảm quá 2pp lợi thế trung bình của ZS so với Leveling ở
  scale nhỏ (Lee 70 instance, Shin 20-bay) — tức không "hy sinh" phần đang thắng
  để cứu phần đang thua

**NO-GO nếu một trong hai điều kiện trên không đạt.** Không có vùng xám — nếu
kết quả mập mờ (vd thắng 1/2 điểm), mặc định NO-GO và ghi nhận như phát hiện
("correction một phần, không toàn diện").

## 6. Lỗ hổng kỹ thuật cần xử lý trong Phase 2 (đã phát hiện, chưa sửa)

Công thức `score = α·DRL − λ·travel − μ·height` với α=0 KHÔNG tái tạo chính xác
Leveling, vì Leveling dùng **bộ lọc cứng ưu tiên cùng-bay trước** (`baselines/
leveling.py:79-92`), không phải phạt mềm theo chiều cao toàn cục. Trước khi tin
"công thức chứa cả 2 endpoint", cần trong Phase 2.1: thêm biến thể `same_bay_mask`
tuỳ chọn trong fusion (ưu tiên cứng cùng-bay giống hệt Leveling khi bật) để
endpoint thứ hai là Leveling thật, không phải một xấp xỉ.

## 7. Ma trận claim (dùng khi viết, không đổi bởi Gate G — chỉ RQ-D bật/tắt)

| Luôn được claim | Chỉ claim nếu Gate G = GO | Không bao giờ claim |
|---|---|---|
| ZS ≡ SOTA tại C=1 (0.0000%) | Ranh giới scale khắc phục được bằng inference-time correction | ZS "tốt hơn" SOTA |
| ZS thắng áp đảo heuristic scale nhỏ-vừa | HGP không thua bất kỳ baseline nào (Mệnh đề 1, verify thực nghiệm) | ZS thắng mọi scale |
| Multi-crane qua zero-shot, chi phí train = 0 | | Đã giải MCSP thật |
| Ranh giới scale đo được + cơ chế giải thích (C4) | | Gap_makespan tuyệt đối phản ánh đúng "tệ gấp X lần" |
| So sánh tương đối gap_makespan hợp lệ | | Multi-crane scaling tốt (chỉ 3-11% speedup C2->3) |

## 8. Rủi ro & xác suất (nhắc lại, không đổi)

- Không có kịch bản nào "chắc chắn" Q1.
- Bài nền (không cần Gate G) tại C&OR/EAAI: ~35-50%.
- Nếu Gate G = GO: tại TR-C ~25-40%, tại Q1 rộng ~60-75%.
- Chuỗi resubmit 12-18 tháng: ~70-85% có một bài Q1.
- Rủi ro không tự sửa trong vòng này: simulator vẫn tuần tự-có-timing (không
  phải concurrent event-driven thật) — khai báo ở Limitations, không né tránh.

## 9. Việc làm NGAY nếu bạn duyệt plan này

Thứ tự bắt tay vào (đợi bạn xác nhận trước khi tôi code):
1. Phase 0.1 — resume 26 instance Shin thiếu (chạy nền, không cần giám sát)
2. Phase 0.4 — commit/push tồn đọng (việc quản lý, tách biệt hoàn toàn với 0.1)
3. Song song: bắt đầu viết script Phase 1.1 (behavioral analysis) — không phụ
   thuộc 0.1 hoàn tất, có thể code trên dữ liệu hiện có rồi chạy lại khi 0.1 xong

**Chưa động vào Phase 2 (HGP) cho đến khi Phase 0-1 xong** — đúng nguyên tắc
"việc rẻ/chắc trước".

---

Bạn duyệt plan này chứ? Nếu có điều gì muốn đổi thứ tự, đổi tiêu chí Gate G, hay
bỏ hẳn nhánh HGP để tập trung 100% vào bài nền — nói trước khi tôi bắt đầu Phase 0.
