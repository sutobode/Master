# 📋 PROFESSOR-LEVEL REVIEW REPORT
## Paper: Zero-shot Transfer of Single-Crane DRL Policies for M-CRP

---

## TỔNG QUAN

| Aspect | Rating | Ghi chú |
|--------|--------|---------|
| **Novelty** | ⭐⭐⭐⭐ | First M-CRP study, but single-crane part là reproduce |
| **Method** | ⭐⭐⭐⭐ | Clear, well-structured, missing complexity analysis |
| **Experiments** | ⭐⭐⭐ | Rộng (1120 runs) nhưng thiếu depth ở single-crane |
| **Analysis** | ⭐⭐⭐ | Đã có cost decomposition, baselines, stats. Cần thêm case study multi-bay |
| **Writing** | ⭐⭐⭐ | Good structure, cần tightening ở vài chỗ |
| **Q1 Readiness** | ⭐⭐⭐ | Gần được, cần 6 fixes trước khi submit |

---

## 🔴 CRITICAL: Phải sửa trước khi submit

### 1. Single-crane benchmark only 4 instances

**Vấn đề:** Table 1 chỉ test trên 4 instances (2 configs × 2 seeds). Reviewer sẽ hỏi ngay: "Tại sao không test full 36-instance Lee benchmark?"

**Yêu cầu TR-C:** Phải chạy trên full Lee benchmark để so sánh công bằng với Shin et al. (2026).

**Fix:** Chạy `compare_all.py` trên tất cả configs: bay ∈ {1,2,4,6,8,10}, tier ∈ {6,8}, id ∈ {1..5}, cả random và upside-down. ~36 instances.

### 2. Backward compatibility only 1 instance

**Vấn đề:** Backward compat verified trên chỉ 1 instance. Cần verify on 5-10 diverse instances để đảm bảo scorer extraction đúng.

**Fix:** Thêm backward compat test trên nhiều instances khác nhau.

### 3. Case study không convincing

**Vấn đề:** Case study trên 2-bay instance cho thấy S1 = S2 về cost. Không chứng minh được superiority của S2.

**Fix:** Chọn 4-bay hoặc 6-bay instance nơi S2 thực sự vượt trội.

---

## 🟡 HIGH: Nên sửa

### 4. Missing complexity analysis

**Vấn đề:** Paper nói O(B), O(C²) cho strategies nhưng không phân tích tổng thể system complexity (DRL inference cost + strategy cost).

**Fix:** Thêm subsection "Computational Complexity Analysis".

### 5. LB_interference formula chưa chặt

**Vấn đề:** LB_interference chỉ phạt per-bay imbalance, không tính crane travel delay do interference. Dẫn đến negative gaps.

**Fix:** Thảo luận về LB tightness và hướng cải thiện.

### 6. No optimal C analysis

**Vấn đề:** Paper recommend S2 nhưng không nói nên dùng 2 hay 3 crane cho layout nào.

**Fix:** Thêm discussion: "When to use 2 vs 3 cranes".

---

## 🟢 MEDIUM: Có thể bổ sung

### 7. Notation table
### 8. Figure references in body text
### 9. Speedup analysis in main paper (not appendix)
### 10. Author info placeholder

---

## PRIORITY FIX LIST

| # | Priority | Fix | Effort | Impact |
|---|----------|-----|--------|--------|
| 1 | 🔴 Critical | Full single-crane benchmark (36 instances) | 30 phút chạy + 1 giờ integrate | Reviewer sẽ hỏi chắc chắn |
| 2 | 🔴 Critical | Multi-bay case study | 5 phút | Chứng minh S2 superiority |
| 3 | 🔴 Critical | Backward compat trên 5+ instances | 5 phút | Tăng độ tin cậy |
| 4 | 🟡 High | Complexity analysis | 30 phút | Reviewer expectation |
| 5 | 🟡 High | LB tightness discussion | 15 phút | Khoa học hơn |
| 6 | 🟢 Medium | Notation table | 20 phút | Clarity |

---

## Tổng kết

Paper hiện tại: **22 trang elsarticle, novelty tốt (first M-CRP study), experiments rộng (1120 runs), analysis đầy đủ (baselines, stats, cost decomposition).**

Điểm yếu nhất: **Single-crane verification chỉ 4 instances** — đây là vấn đề lớn nhất với TR-C reviewers.

Suggest: Fix critical issues (1-3) trước, sau đó submit.

Bạn muốn tôi fix các critical issues ngay bây giờ không?
