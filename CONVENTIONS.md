# CONVENTIONS.md — Quy ước nhanh cho dự án INT1418

> **Dự án:** Hệ CSDL lưu trữ và tìm kiếm tiếng động vật  
> **Môn:** Cơ sở dữ liệu đa phương tiện (INT1418)  
> **Mục đích:** File tham chiếu nhanh — AI Agent đọc file này đầu tiên khi bắt đầu mỗi phiên làm việc.

---

## 🔗 Tài liệu chi tiết

| File | Nội dung |
|---|---|
| [context/problem.md](context/problem.md) | Đề bài gốc (KHÔNG chỉnh sửa) |
| [context/rules.md](context/rules.md) | **8 nhóm quy tắc bắt buộc** (R-01 → R-08) |
| [context/skills.md](context/skills.md) | **6 kỹ năng chuyên môn** (S-01 → S-06) |

---

## ⚡ Tóm tắt quy tắc quan trọng nhất

1. **Python ≥ 3.10**, chạy trên CPU, không thư viện trả phí
2. **Dataset ≥ 500 files**, mỗi file = 1 loài, chuẩn hóa `.wav` mono 22050Hz
3. **Đặc trưng:** MFCC + Mel Spectrogram + ZCR + Spectral Centroid + Chroma → vector 310-D
4. **Tìm kiếm:** Cosine Similarity qua Faiss, **luôn trả đúng top 5**, sắp xếp giảm dần
5. **Lưu kết quả trung gian** (`.npy`, spectrogram images) vào `features/intermediate/`
6. **Demo UI** phải có: Upload, Audio Player, Waveform/Spectrogram, Bảng Top-5, Audio Player kết quả
7. **Tài liệu:** Giải trình đặc trưng + Sơ đồ khối Mermaid + Báo cáo kết quả trung gian
8. **Test 2 kịch bản:** File có trong CSDL (expect rank 1 = chính nó) + File không có trong CSDL

---

## 📁 Cấu trúc thư mục

```
int1418/
├── context/          # Đề bài + Rules + Skills
├── data/raw/         # Audio gốc phân loài (GITIGNORE)
├── data/processed/   # Audio chuẩn hóa (GITIGNORE)
├── features/         # Vector DB + index + intermediate
├── docs/             # Giải trình + Sơ đồ + Báo cáo
├── src/              # Source code Python
├── app/              # Demo UI (Streamlit)
├── tests/            # Unit tests
└── requirements.txt
```

---

## 🚦 Workflow thực thi

```
Phase 1: Data → Phase 2: Features → Phase 3: Search Engine → Phase 4: Demo UI
```

Khi bắt đầu bất kỳ task nào, **đọc `context/rules.md`** để đảm bảo tuân thủ quy tắc, và tham chiếu `context/skills.md` để áp dụng đúng kỹ thuật.
