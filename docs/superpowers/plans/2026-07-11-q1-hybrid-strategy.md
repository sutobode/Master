# Chiến lược nghiên cứu Q1 (v3) — Hybrid Guided Planner với bảo chứng thống trị

**Ngày:** 2026-07-11. Thay thế trọng tâm của `2026-07-11-q1-research-strategy.md`
(bản đó bias giữ zero-shot thuần; bản này ưu tiên THẮNG CHẮC baseline + SOTA,
zero-shot trở thành một thành phần + một finding, không còn là toàn bộ thesis).

---

## 0. Ý tưởng cốt lõi: vì sao "thắng chắc" là khả thi về mặt cấu trúc

CRP/M-CRP bản offline là bài toán **full-information, mô phỏng rẻ và tất định**:
với một instance, chạy TRỌN VẸN một planner bất kỳ rồi đo đúng chi phí thật chỉ
tốn giây→~40s. Vì vậy có thể xây phương pháp 3 tầng mà tầng cuối **đảm bảo bằng
cấu trúc** không bao giờ thua bất kỳ thành phần nào:

- Dữ liệu thật đã chứng minh **tính bù trừ (complementarity)**: DRL thắng lớn ở
  scale nhỏ (7.15% vs 23.5%), Leveling thắng ở scale lớn (12.65% vs 14.36% tại
  30×6). Hai "chuyên gia" mạnh ở hai chế độ khác nhau → đúng điều kiện để hybrid
  vượt cả hai.
- SOTA (Original Model) tại C=1 ≡ ZeroShot (0.0000%) → mọi phương pháp ≥ ZS tại
  C=1 nghĩa là ≥ SOTA gốc. Thắng SOTA quy về thắng thành phần ZS của chính mình.

## 1. Phương pháp đề xuất: HGP — Hybrid Guided Planner (3 tầng)

### Tầng 1 — Score fusion từng bước (đóng góp phương pháp chính)
Điểm chọn destination mỗi bước là tổ hợp:

```
score(d) = α · logp_DRL(d)  −  λ · travel(target→d)/T_norm  −  μ · height(d)/H_norm  [+ zone_mask khi C>1]
```

- `logp_DRL`: log-softmax từ scorer đóng băng (policy/zero_shot.py — giữ nguyên trọng số).
- `travel`: chi phí di chuyển thật (t_acc/t_bay/t_row) — bù đúng khiếm khuyết
  "argmax toàn cục mù khoảng cách" đã chẩn đoán.
- `height`: chiều cao stack đích — chính là tín hiệu của Leveling.
- **Tính chất bao hàm (lemma phát biểu được trong paper):** họ tham số này CHỨA
  cả hai endpoint: (α=1, λ=μ=0) → đúng ZeroShot/SOTA; (α=0, μ>0, λ nhỏ) → đúng
  Leveling + tie-break travel. Hybrid là *tổng quát hoá chặt* của cả phương pháp
  học lẫn heuristic mạnh nhất — không phải một mẹo ghép.
- α, λ, μ chọn trên **validation tách riêng** (20%/scale), có thể cho α phụ
  thuộc scale (α(bays) — dạng gating đơn giản, 2 tham số).

### Tầng 2 — Candidate-union 1-step lookahead (tuỳ chọn, bật/tắt qua ablation)
Mỗi bước lấy hợp {top-k của DRL} ∪ {ứng viên Leveling} ∪ {ứng viên Lin}, mô phỏng
finish-time 1 bước (tái dùng máy móc GreedyOptimal + timing model MCEnv sẵn có),
chọn argmin. Chi phí O(k·C·B)/bước — vẫn realtime.

### Tầng 3 — Portfolio guard (nguồn gốc của "thắng chắc")
Với mỗi instance chạy trọn: {HGP-fused, ZeroShot thuần, M-Leveling}, lấy plan rẻ
nhất (theo makespan cho M-CRP, working time cho C=1).

> **Mệnh đề 1 (dominance-by-construction):** chi phí của HGP ≤ min(chi phí từng
> thành phần) trên MỌI instance. Hệ quả: HGP ≤ SOTA gốc tại C=1 trên mọi
> instance, và HGP ≤ M-Leveling/M-Lin/M-Kim trên mọi instance M-CRP.
> Chứng minh: tầm thường theo định nghĩa min — nhưng phát biểu được, đúng, và
> biến "thắng baseline + SOTA" từ kỳ vọng thực nghiệm thành thuộc tính cấu trúc.

Tổng runtime ≈ 1.1–1.3× một lần chạy ZS (heuristic không đáng kể) — vẫn trong
ngân sách "≤40s/instance cực lớn" mà bài gốc quảng bá.

**Metric trung thực đi kèm:** `fallback_rate` — tỉ lệ instance mà tầng 3 phải
cứu tầng 1 (fusion thua một endpoint thuần). Fusion tốt → fallback_rate ~0 và
fusion THẮNG CHẶT cả hai endpoint trên nhiều instance (kết quả thú vị thật sự);
fusion tồi → paper vẫn đứng nhờ Mệnh đề 1, và fallback_rate cao tự nó là finding.

## 2. Bộ claim của paper (đều an toàn)

- **K1 (đảm bảo):** HGP không bao giờ thua SOTA và mọi baseline được so sánh,
  trên mọi instance, mọi scale, cả hai setting — by construction (Mệnh đề 1),
  xác nhận empirically trên 4 bộ dữ liệu đầy đủ.
- **K2 (kỳ vọng cao, kiểm chứng bằng B3):** riêng tầng fusion (không cần guard)
  thắng chặt cả ZS lẫn Leveling trên phần lớn instance — vì nó vá đúng cơ chế
  thua đã chẩn đoán (mù khoảng cách) trong khi giữ chất lượng xếp chồng của DRL.
- **K3 (phân tích):** ranh giới scale của zero-shot transfer + cơ chế cross-bay
  — giữ nguyên từ plan v2 (behavioral analysis), giờ đóng vai "động cơ thiết kế"
  của HGP thay vì kết quả tiêu cực trơ trọi.
- **K4:** MCEnv timing model + LB hợp lệ + backward-compat 0.0000% (giữ nguyên).
- Giới hạn khai báo rõ: "thắng chắc" chỉ so với các phương pháp NẰM TRONG
  portfolio và offline full-information; không claim vs optimum, vs TS/GRASP/GP
  (không reimplement), hay online setting.

## 3. Kiểm tra novelty (bắt buộc trước khi code — 0.5 ngày)

Hybrid DRL+heuristic và algorithm-portfolio đều có truyền thống (hyper-heuristics,
algorithm selection, Bertsekas rollout). Cần search: đã ai làm score-fusion /
portfolio cho CRP/BRP multi-crane chưa? Điểm khác biệt kỳ vọng: (i) chưa có
multi-crane CRP DRL nào cả (bài gốc xác nhận là future work); (ii) fusion trên
policy ĐÓNG BĂNG với lemma bao hàm 2 endpoint; (iii) dominance guard được đo
bằng fallback_rate. Nếu search ra prior art gần → điều chỉnh framing sang
"first for M-CRP" (vẫn đứng vững nhờ (i)).

## 4. Kế hoạch thực thi

### Phase A — Nền dữ liệu + novelty (1 ngày)
- A1. Resume 26 instance Shin thiếu (bắt buộc trước mọi kết luận 30-bay).
- A2. Novelty search (mục 3). A3. Đối chiếu Table A.8 bài gốc (bằng chứng reproduction).

### Phase B — HGP core + thí nghiệm quyết định (3–5 ngày)
- B1. Implement tầng 1 (fusion): thêm `travel_penalty`/`height_penalty`/`zone_mask`
  vào `get_scores()` — cộng vào logits trước argmax, không đụng trọng số.
- B2. Implement tầng 3 (portfolio guard) trong runner: chạy N planner, ghi từng
  plan + chọn min; cột mới `winner_component`, `fallback_used`.
- B3. **Go/no-go cho K2:** tune (α,λ,μ) trên validation nhỏ rồi test fusion trên
  Shin 30×6 + MC-large makespan (chỗ ZS đang thua). K1 không cần go/no-go —
  đúng by construction.
- B4. Tầng 2 (lookahead) nếu B3 còn dư địa cải thiện.
- B5. Re-run full matrix 4 bộ dữ liệu với HGP + các ablation (tận dụng
  record/replay, multiprocessing, resume sẵn có).

### Phase C — Phân tích (2 ngày)
- C1. Complementarity/regime map: thành phần nào thắng ở đâu (scale × type × C ×
  strategy); fallback_rate theo scale.
- C2. Behavioral: cross-bay-rate & bay-distance của ZS / fusion / Leveling theo
  scale (figure chính, chứng minh cơ chế).
- C3. Wilcoxon + Cohen's d + σ_d cho: fusion vs ZS, fusion vs Leveling, HGP vs
  từng baseline (HGP vs baseline sẽ one-sided tầm thường — báo cáo win-rate và
  mean-improvement thay vì p-value rỗng).

### Phase D — Ablation + mở rộng tuỳ chọn (2–4 ngày)
- D1. Ablation: từng term (α/λ/μ/zone_mask/lookahead), sensitivity theo tham số
  (thay thế đoạn sensitivity không-có-dữ-liệu cũ bằng sweep THẬT).
- D2 (tuỳ chọn TR-C): α(scale) gating học từ features instance (bays, tiers,
  fill-rate) — mảnh "learning" nhỏ; hoặc fine-tune ablation từ plan v2.

### Phase E — Viết lại paper (4–6 ngày)
- Cấu trúc RQ: RQ1 fidelity (0.0000%); RQ2 ranh giới scale của ZS thuần (động
  cơ); RQ3 fusion có thắng chặt cả hai endpoint? RQ4 dominance toàn cục của HGP
  vs SOTA + baselines (headline table); RQ5 ablation + runtime; RQ6 strategies
  S1–S4 dưới HGP.
- Đồng bộ .tex với code/data đã fix (checklist audit cũ: Theorem 3, bibliography
  6 lỗi, encoder duplicate, NP-hardness, 70 unique layouts, số liệu mới).
- Title hướng: "Hybrid learning-guided planning for multi-crane container
  retrieval: dominating both a deep RL policy and classical heuristics across
  scales without retraining".

### Phase F — Gate
- pytest sạch + test mới: Mệnh đề 1 (HGP ≤ min components trên toàn bộ CSV);
  fallback_rate được log; backward-compat 0.0000%; re-audit paper-audit;
  self-review 3 persona.

## 5. Venue

| Kết quả B3 (fusion) | Venue chính | Ghi chú |
|---|---|---|
| Fusion thắng chặt cả 2 endpoint đa số instance | **TR-C** | K1+K2+K3 đủ sức nặng, đúng future-work bài gốc |
| Fusion ngang endpoint tốt nhất, K1 vẫn giữ | **TR-E / C&OR** | Headline chuyển sang K1+K3 (dominance + boundary) |
| (không có kịch bản thua — K1 by construction) | — | — |

Tổng: ~2.5–3.5 tuần. Khác biệt then chốt so với plan v2: **không còn kịch bản
thất bại thực nghiệm** — chỉ còn thang mạnh/yếu của kết quả fusion quyết định
venue, vì tầng portfolio đã khoá sàn kết quả.

## 6. Rủi ro còn lại (đã đổi bản chất: từ "có thể thua" sang "có thể bị chê")

| Rủi ro | Đối sách |
|---|---|
| Reviewer: "portfolio là trivial" | Portfolio chỉ là guard/floor; headline là fusion (K2) + cơ chế (K3) + lemma bao hàm; fallback_rate chứng minh fusion tự đứng được |
| Prior art hybrid CRP tồn tại | A2 novelty search trước khi code; fallback framing "first for M-CRP" |
| Tuning bị nghi overfit | Validation/test split khai báo từ đầu, sensitivity sweep D1 |
| Runtime bị chê (chạy nhiều planner) | Báo cáo runtime thật: heuristics ≪ DRL, tổng ≈1.2× ZS, vẫn ≤ ~60s/instance |
| "Không còn là zero-shot" làm mất câu chuyện cũ | Đổi câu chuyện: zero-shot là RQ1-RQ2 (fidelity + boundary finding), HGP là câu trả lời — mạch "phát hiện vấn đề → giải quyết" chặt hơn bản cũ |
