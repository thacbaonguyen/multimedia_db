# 📝 BÁO CÁO ĐÁNH GIÁ ĐỒ ÁN INT1418
## Hệ CSDL lưu trữ và tìm kiếm tiếng động vật

> **Vai trò:** Giảng viên chấm bài  
> **Tiêu chuẩn:** Đối chiếu 100% với đề bài (`problem.md`), quy tắc (`rules.md` R-01→R-08), kỹ năng (`skills.md` S-01→S-06)

---

## 🏆 TỔNG ĐIỂM: 52/100 — CHƯA ĐẠT YÊU CẦU

---

## I. ĐÁNH GIÁ THEO TỪNG YÊU CẦU ĐỀ BÀI (`problem.md`)

### Yêu cầu 1: Bộ dữ liệu ≥ 500 files (15/15 điểm)

| Tiêu chí | Đánh giá | Điểm |
|---|---|---:|
| Số lượng file ≥ 500 | ✅ 1,051 files — vượt yêu cầu | 5/5 |
| Mỗi file = 1 loài | ✅ Cấu trúc phân loài rõ ràng | 3/3 |
| Đa dạng loài | ✅ 8 nhãn (Dog, Cat, Cow, Frog, Sheep, Monkey, Hen, Rooster) | 4/4 |
| Nguồn đa dạng | ✅ ESC-50, AnimalQA, DynamicSuperb, SoundDino, local | 3/3 |

> [!TIP]
> Đây là phần mạnh nhất của bài. Dataset vượt yêu cầu cả về lượng lẫn đa dạng nguồn.

---

### Yêu cầu 2: Bộ thuộc tính + Giải trình lý do (3/20 điểm)

| Tiêu chí | Đánh giá | Điểm |
|---|---|---:|
| Có bộ đặc trưng hợp lệ | ✅ 58D vector (MFCC, Chroma, Spectral, ZCR, RMS) | 3/5 |
| Tài liệu giải trình (`docs/feature_justification.md`) | ❌ **KHÔNG TỒN TẠI** | 0/8 |
| Biểu đồ minh họa ≥ 2 loài | ❌ **KHÔNG CÓ** | 0/4 |
| Giải thích tại sao phù hợp tiếng động vật | ❌ Chỉ có comment ngắn trong code, không có tài liệu | 0/3 |

> [!CAUTION]
> **Vi phạm nghiêm trọng R-06.1:** Đề bài yêu cầu rõ ràng "Trình bày cụ thể về lý do lựa chọn và giá trị thông tin của các thuộc tính." — Bài làm hoàn toàn thiếu tài liệu này. Docstring trong code KHÔNG thay thế được tài liệu giải trình.

**Sai lệch đặc trưng so với rules.md:**

| Đặc trưng (R-04.1 bắt buộc) | Trong bài | Vấn đề |
|---|---|---|
| MFCC | ✅ 26D (mean+std) | OK |
| **Mel Spectrogram** | ❌ **THIẾU** | **Bắt buộc theo R-04.1, không có trong vector 58D** |
| Spectral Centroid | ✅ 1D (chỉ mean) | Thiếu std, R-04 yêu cầu mean+std |
| ZCR | ✅ 2D | OK |
| Chroma | ✅ 24D | OK |
| Spectral Bandwidth | ➕ Thêm (không bắt buộc) | Bonus |
| Spectral Rolloff | ➕ Thêm | Bonus |
| Spectral Flatness | ➕ Thêm | Bonus |
| RMS Energy | ➕ Thêm | Bonus |

> [!WARNING]
> **Mel Spectrogram — đặc trưng bắt buộc theo R-04.1 — hoàn toàn bị bỏ qua.** Đây là yêu cầu cứng trong rules.md. Vector đúng phải là 310D (theo CONVENTIONS.md) nhưng bài chỉ có 58D.

---

### Yêu cầu 3a: Sơ đồ khối hệ thống (0/10 điểm)

| Tiêu chí | Đánh giá | Điểm |
|---|---|---:|
| File `docs/system_diagram.md` | ❌ **KHÔNG TỒN TẠI** | 0/5 |
| Sơ đồ Mermaid Indexing Pipeline | ❌ KHÔNG CÓ | 0/2.5 |
| Sơ đồ Mermaid Query Pipeline | ❌ KHÔNG CÓ | 0/2.5 |

> [!CAUTION]
> **Vi phạm R-06.2:** Đề bài yêu cầu "Trình bày sơ đồ khối của hệ thống và quy trình thực hiện." — Hoàn toàn vắng mặt.

---

### Yêu cầu 3b: Kết quả trung gian (0/15 điểm)

| Tiêu chí | Đánh giá | Điểm |
|---|---|---:|
| File `docs/search_results_report.md` | ❌ **KHÔNG TỒN TẠI** | 0/5 |
| Hiển thị quá trình input → đặc trưng → khoảng cách → ranking | ❌ KHÔNG CÓ | 0/5 |
| Ảnh spectrogram/waveform query + 5 kết quả | ❌ KHÔNG CÓ | 0/3 |
| Thư mục `features/intermediate/` chứa spectrogram images | ❌ KHÔNG CÓ (chỉ có .npy files) | 0/2 |

> [!CAUTION]
> **Vi phạm R-04.3 & R-06.3:** Không lưu kết quả trung gian (spectrogram images), không có báo cáo kết quả trung gian. Đây là yêu cầu đề bài: "Trình bày các kết quả trung gian của quá trình tìm kiếm."

---

### Yêu cầu 4: Demo hệ thống + Đánh giá kết quả (15/20 điểm)

| Tiêu chí | Đánh giá | Điểm |
|---|---|---:|
| Upload file audio | ✅ Gradio `gr.Audio(sources=["upload", "microphone"])` | 2/2 |
| Nút Search | ✅ `gr.Button("Search")` | 1/1 |
| Audio Player cho query | ✅ Gradio tự động | 2/2 |
| Waveform/Spectrogram cho query | ⚠️ Chỉ có Mel Spectrogram, không có Waveform | 1.5/3 |
| Bảng Top-5 (Rank, File, Species, Score) | ✅ HTML table đẹp | 3/3 |
| Audio Player cho từng kết quả | ❌ **KHÔNG CÓ** — chỉ hiển thị text, không thể nghe | 0/3 |
| Biểu đồ so sánh query vs results | ⚠️ Chỉ so sánh query vs Top-1, không phải 5 | 1.5/3 |
| Có evaluation metrics (Hit@1, P@5) | ✅ Có evaluate.py | 2/2 |
| Hoạt động thực tế (không mock data) | ✅ Pipeline thực | 2/2 |

> [!WARNING]
> **Vi phạm R-07.1:** Thiếu Audio Player cho từng file trong Top 5 — đây là yêu cầu bắt buộc. Người chấm không thể nghe thử kết quả trả về.

---

### Yêu cầu kiểm thử 2 kịch bản (4/5 điểm)

| Kịch bản | Đánh giá | Điểm |
|---|---|---:|
| File CÓ trong CSDL → rank 1 = chính nó | ✅ `evaluate.py` leave-one-out, Hit@1 = 87.35% | 2/2.5 |
| File KHÔNG CÓ trong CSDL → vẫn trả 5 | ✅ Có cơ chế `is_unknown` + `assess_unknown()` | 2/2.5 |

> [!NOTE]
> Hit@1 = 87.35% là chấp nhận được nhưng không cao. Một số loài yếu (Cow 72.5%, Cat 77.4%). Cơ chế unknown detection là điểm sáng.

---

## II. ĐÁNH GIÁ THEO QUY TẮC (`rules.md`)

### R-01: Ngôn ngữ & Công nghệ (8/10)

| Yêu cầu | Thực tế | Điểm |
|---|---|---:|
| Python ≥ 3.10 | ✅ Dùng type hints `list[str]`, `tuple[bool, str]` | 2/2 |
| librosa, soundfile | ✅ | 2/2 |
| numpy, scipy | ✅ numpy; scipy trong requirements nhưng không import trực tiếp | 1/1 |
| faiss-cpu hoặc sklearn | ⚠️ Dùng sklearn SVM, **KHÔNG dùng Faiss** — cosine tính thủ công | 1/2 |
| Streamlit hoặc Gradio | ✅ Gradio | 2/2 |
| Không trả phí, chạy CPU | ✅ | 1/1 |

> [!IMPORTANT]
> **Không sử dụng Faiss** — quy tắc R-01 ghi ưu tiên `faiss-cpu`. CONVENTIONS.md ghi rõ "Cosine Similarity **qua Faiss**". Bài dùng numpy thủ công thay vì Faiss IndexFlatIP.

---

### R-02: Cấu trúc thư mục (3/10)

| Yêu cầu | Thực tế | Vấn đề |
|---|---|---|
| `data/raw/` | `data/balanced8_raw/` | ⚠️ Đổi tên |
| `data/processed/` | `data/balanced8_processed/` | ⚠️ Đổi tên |
| `data/metadata.csv` | `data/balanced8_metadata.csv` | ⚠️ Đổi tên, thiếu cột `file_id`, `filepath`, `duration_sec`, `sample_rate` |
| `features/feature_db.npy` | Không có (lưu .npy riêng lẻ) | ❌ Thiếu |
| `features/file_index.json` | Không có (dùng SQLite) | ⚠️ Thay thế |
| `features/intermediate/` | Không có | ❌ Thiếu |
| `docs/` | **KHÔNG TỒN TẠI** | ❌ Thiếu hoàn toàn |
| `tests/` | **KHÔNG TỒN TẠI** | ❌ Thiếu hoàn toàn |
| `app/demo.py` | `demo/app.py` | ⚠️ Đổi tên |

> [!CAUTION]
> Cấu trúc thư mục sai lệch đáng kể so với R-02. Không có `docs/`, `tests/`, `features/intermediate/`. Metadata CSV thiếu nhiều cột bắt buộc theo R-03.3.

---

### R-03: Quản lý dữ liệu (4/10)

| Tiêu chí | Đánh giá |
|---|---|
| `.gitignore` | ❌ **KHÔNG TỒN TẠI** |
| 1 File = 1 Loài | ✅ |
| Metadata đủ cột bắt buộc | ❌ Thiếu `file_id`, `filepath`, `duration_sec`, `sample_rate`, `source` (có nhưng khác schema) |
| Chuẩn hóa mono + 22050Hz + .wav | ✅ |
| Trim silence | ❌ Không dùng `librosa.effects.trim()` — dùng center-crop thay vì trim |
| Normalize amplitude | ✅ RMS normalization |

> [!WARNING]
> **Không có `.gitignore`** — vi phạm R-03.1. Nếu commit lên Git sẽ push hàng GB audio lên repository.
> 
> **Không trim silence** (vi phạm S-01): `preprocess.py` dùng center-crop (`normalize_length` lấy đoạn giữa), KHÔNG dùng `librosa.effects.trim()` để loại silence. Skills.md cảnh báo rõ: "Luôn dùng `librosa.effects.trim()` để loại bỏ đoạn im lặng."

---

### R-04: Feature Extraction (5/10)

| Tiêu chí | Đánh giá |
|---|---|
| 5 đặc trưng bắt buộc | ❌ Thiếu Mel Spectrogram |
| Vector 1-D cố định | ✅ 58D |
| mean + std aggregation | ⚠️ Spectral features chỉ có mean, thiếu std |
| `feature_db.npy` (N×D) | ❌ Không có — lưu riêng lẻ 1,051 file .npy |
| `file_index.json` | ❌ Không có — dùng SQLite thay thế |
| Intermediate results | ❌ Không lưu spectrogram images |

---

### R-05: Search Engine (7/10)

| Tiêu chí | Đánh giá |
|---|---|
| Luôn trả top_k=5 | ✅ |
| Output schema đúng | ⚠️ Gần đúng, dùng `hybrid_score` thay `similarity_score`, thiếu `distance` |
| Cosine Similarity | ✅ Có cosine, nhưng kết hợp hybrid score |
| Sắp xếp giảm dần | ✅ |
| Test 2 kịch bản | ✅ |

> [!NOTE]
> Hệ thống search vượt yêu cầu đề bài: thêm SVM classifier + hybrid reranking. Tuy nhiên, output schema không khớp R-05.2 (thiếu `distance`, thêm nhiều field không yêu cầu).

---

### R-06: Tài liệu & Báo cáo (0/10)

| Tài liệu bắt buộc | Tồn tại? |
|---|---|
| `docs/feature_justification.md` | ❌ |
| `docs/system_diagram.md` | ❌ |
| `docs/search_results_report.md` | ❌ |

> [!CAUTION]
> **0/3 tài liệu bắt buộc.** Folder `docs/` không tồn tại. Đây là lỗi mất điểm lớn nhất.

---

### R-07: Demo UI (6/10) — Đã chấm ở trên

---

### R-08: Code Quality (6/10)

| Tiêu chí | Đánh giá |
|---|---|
| Docstring mọi function | ✅ Hầu hết có docstring |
| Type hints | ⚠️ Một số file có (search.py, train_classifier.py), một số thiếu (preprocess.py, database.py) |
| Error handling | ⚠️ Có try/except trong indexing nhưng dùng generic `Exception` |
| `random_seed` cố định | ✅ `RANDOM_STATE = 42` trong train_classifier.py |
| Phiên bản thư viện trong requirements.txt | ❌ **Không pin version** — `librosa` thay vì `librosa==0.10.1` |
| Unit tests | ❌ **Không có thư mục `tests/`** |

---

## III. BẢNG TỔNG HỢP ĐIỂM

| Hạng mục | Trọng số | Điểm | Chi tiết |
|---|---:|---:|---|
| **YC1:** Dataset ≥ 500 | 15% | **15/15** | Xuất sắc — 1,051 files, 8 nhãn |
| **YC2:** Bộ thuộc tính + Giải trình | 20% | **3/20** | Thiếu Mel Spec, thiếu tài liệu giải trình |
| **YC3a:** Sơ đồ khối | 10% | **0/10** | Hoàn toàn vắng mặt |
| **YC3b:** Kết quả trung gian | 15% | **0/15** | Hoàn toàn vắng mặt |
| **YC4:** Demo + Đánh giá | 20% | **15/20** | Demo hoạt động nhưng thiếu audio player cho kết quả |
| **Kiểm thử 2 kịch bản** | 5% | **4/5** | Có cả 2 kịch bản |
| **Code Quality (R-08)** | 10% | **6/10** | Thiếu tests, thiếu version pinning |
| **Cấu trúc + Governance** | 5% | **3/5** | Sai cấu trúc, thiếu .gitignore |
| **TỔNG** | **100%** | **46/100** | |

**Điểm thưởng (bonus):** +6 điểm cho các sáng tạo vượt yêu cầu:
- SVM classifier + hybrid reranking (+3)
- Unknown detection mechanism (+1.5)
- SQLite database với search_log (+1)
- Dataset build automation từ HuggingFace (+0.5)

### **TỔNG CỘNG: 52/100**

---

## IV. CÁC LỖI NGHIÊM TRỌNG NHẤT (Ưu tiên sửa)

### 🔴 Mức Nghiêm trọng (Mất ≥ 10 điểm mỗi lỗi)

1. **Thiếu toàn bộ folder `docs/`** — Mất 35 điểm cộng dồn
   - `feature_justification.md` — Giải trình đặc trưng
   - `system_diagram.md` — Sơ đồ khối Mermaid  
   - `search_results_report.md` — Báo cáo kết quả trung gian

2. **Thiếu Mel Spectrogram trong vector đặc trưng** — Đặc trưng bắt buộc R-04.1

3. **Thiếu intermediate results** — Không lưu spectrogram images vào `features/intermediate/`

### 🟡 Mức Trung bình (Mất 3-8 điểm)

4. **Demo thiếu Audio Player cho Top-5** — Không thể nghe kết quả
5. **Không dùng Faiss** — CONVENTIONS.md ghi rõ "qua Faiss"
6. **Thiếu `.gitignore`** — Vi phạm R-03.1
7. **Thiếu `tests/`** — Không có unit test nào
8. **`requirements.txt` không pin version** — Vi phạm R-08.3

### 🟢 Mức Nhẹ (Mất 1-2 điểm)

9. Cấu trúc thư mục đổi tên so với R-02
10. Metadata CSV thiếu cột bắt buộc
11. Preprocess không dùng `librosa.effects.trim()`
12. Output schema search khác R-05.2

---

## V. NHẬN XÉT TỔNG THỂ

### Điểm mạnh
- Pipeline xử lý hoạt động end-to-end
- Dataset chất lượng, đa nguồn, vượt yêu cầu
- Hybrid search (cosine + SVM) là ý tưởng sáng tạo
- Code có tổ chức module rõ ràng
- Có evaluation metrics (leave-one-out)

### Điểm yếu
- **Thiếu hoàn toàn phần tài liệu** — đây là phần chiếm 45% yêu cầu đề bài
- Không tuân thủ cấu trúc thư mục quy định
- Demo chưa đủ chức năng bắt buộc
- Thiếu đặc trưng Mel Spectrogram (bắt buộc)
- Không có unit tests

> **Kết luận:** Bài có nền tảng kỹ thuật tốt (pipeline, SVM, evaluation) nhưng **bỏ qua gần như toàn bộ phần tài liệu và báo cáo** — phần mà đề bài yêu cầu rõ ràng và chiếm tỷ trọng lớn. Nếu bổ sung đầy đủ docs + sửa các lỗi 🔴, bài có thể đạt **75-85 điểm**.
