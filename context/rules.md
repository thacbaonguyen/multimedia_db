# 📜 Rules – Quy tắc bắt buộc cho AI Agent

> Tài liệu này là **bộ luật cứng** (hard constraints) mà AI Agent phải tuân thủ **tuyệt đối** trong toàn bộ vòng đời dự án "Hệ CSDL lưu trữ và tìm kiếm tiếng động vật".
> Mọi vi phạm sẽ dẫn đến kết quả sai yêu cầu đề bài.

---

## R-01: Ngôn ngữ & Công nghệ

| Hạng mục | Giá trị bắt buộc |
|---|---|
| Ngôn ngữ lập trình | **Python ≥ 3.10** |
| Quản lý package | `pip` + `requirements.txt` (hoặc `pyproject.toml`) |
| Âm thanh I/O | `librosa`, `soundfile` |
| Tính toán số | `numpy`, `scipy` |
| Vector DB / Similarity | `faiss-cpu` (ưu tiên) hoặc `scikit-learn` |
| Trực quan hóa | `matplotlib`, `librosa.display` |
| Web Demo | `Streamlit` hoặc `Gradio` |
| Backend API (nếu cần) | `FastAPI` |

**Cấm:** Không sử dụng thư viện trả phí, không yêu cầu GPU bắt buộc (dự án phải chạy được trên CPU).

---

## R-02: Cấu trúc thư mục dự án (Project Structure)

```
int1418/
├── context/               # Đề bài, rules, skills (KHÔNG chỉnh sửa)
│   ├── problem.md
│   ├── rules.md
│   └── skills.md
├── data/
│   ├── raw/               # File âm thanh gốc, phân theo loài
│   │   ├── cat/
│   │   ├── dog/
│   │   └── ...
│   ├── processed/          # File đã chuẩn hóa (mono, 22050Hz, .wav)
│   └── metadata.csv        # Bảng ánh xạ: filename → species → duration → source
├── features/
│   ├── feature_db.npy      # Ma trận đặc trưng toàn bộ dataset (N × D)
│   ├── file_index.json     # Ánh xạ index → filepath + metadata
│   └── intermediate/       # Kết quả trung gian (spectrogram images, vector từng file)
├── docs/
│   ├── feature_justification.md   # Giải trình lý do chọn đặc trưng (Yêu cầu 2)
│   ├── system_diagram.md          # Sơ đồ khối hệ thống (Yêu cầu 3a)
│   └── search_results_report.md   # Báo cáo kết quả trung gian (Yêu cầu 3b)
├── src/
│   ├── __init__.py
│   ├── preprocessing.py    # Chuẩn hóa audio
│   ├── feature_extraction.py # Trích xuất đặc trưng
│   ├── indexing.py          # Xây dựng CSDL đặc trưng
│   ├── search_engine.py     # Core tìm kiếm tương tự
│   ├── visualization.py     # Vẽ waveform, spectrogram, so sánh
│   └── utils.py             # Hàm tiện ích dùng chung
├── app/
│   └── demo.py              # Streamlit/Gradio demo UI
├── tests/
│   ├── test_preprocessing.py
│   ├── test_feature_extraction.py
│   └── test_search_engine.py
├── notebooks/               # Jupyter notebooks thử nghiệm (tùy chọn)
├── requirements.txt
├── .gitignore
├── CONVENTIONS.md            # Tổng hợp quy ước nhanh
└── README.md
```

**Bắt buộc:** Luôn giữ đúng cấu trúc trên. Không đặt code xử lý vào thư mục `data/`. Không đặt tệp dữ liệu vào `src/`.

---

## R-03: Quản lý Dữ liệu (Dataset Governance)

### R-03.1: Gitignore
```gitignore
# Audio files — KHÔNG commit lên Git
data/raw/
data/processed/
features/*.npy
features/intermediate/
*.wav
*.mp3
*.flac
*.ogg
```
Agent phải tạo `.gitignore` với nội dung trên **ngay khi khởi tạo dự án**.

### R-03.2: Quy tắc 1 File = 1 Loài
- Mỗi file âm thanh **chỉ chứa tiếng của đúng 1 loài động vật**.
- Nếu file có tạp âm nền (tiếng gió, tiếng người nói...) quá nhiều → loại bỏ hoặc xử lý lọc nhiễu trước khi đưa vào pipeline.

### R-03.3: Metadata bắt buộc
File `data/metadata.csv` phải tồn tại và có cấu trúc tối thiểu:

| Cột | Kiểu | Mô tả |
|---|---|---|
| `file_id` | string | ID duy nhất của file |
| `filename` | string | Tên file (ví dụ: `cat_001.wav`) |
| `species` | string | Tên loài (ví dụ: `cat`, `dog`, `rooster`) |
| `filepath` | string | Đường dẫn tương đối từ root (ví dụ: `data/processed/cat/cat_001.wav`) |
| `duration_sec` | float | Độ dài file (giây) |
| `sample_rate` | int | Sample rate sau chuẩn hóa |
| `source` | string | Nguồn gốc file (kaggle, freesound, self-recorded) |

### R-03.4: Chuẩn hóa Audio
Tất cả file trước khi đưa vào feature extraction **bắt buộc** phải qua các bước:
1. Convert sang **mono channel**
2. Resample về **22050 Hz** (hoặc giá trị thống nhất do User quyết định)
3. Convert định dạng về **`.wav`** (PCM 16-bit)
4. Cắt/padding về độ dài chuẩn (ví dụ: 5 giây) nếu cần thiết cho vector đều kích thước

---

## R-04: Trích xuất Đặc trưng (Feature Extraction Constraints)

### R-04.1: Bộ đặc trưng bắt buộc
AI Agent phải trích xuất **ít nhất** các đặc trưng sau cho mỗi file:

| Đặc trưng | Thư viện / Hàm | Mục đích |
|---|---|---|
| **MFCC** (Mel-Frequency Cepstral Coefficients) | `librosa.feature.mfcc()` | Đặc trưng âm sắc chính, mô phỏng cảm nhận tần số của tai người |
| **Mel Spectrogram** | `librosa.feature.melspectrogram()` | Biểu diễn năng lượng trên thang tần số Mel theo thời gian |
| **Spectral Centroid** | `librosa.feature.spectral_centroid()` | "Trọng tâm" phổ tần – cho biết âm thanh "sáng" hay "tối" |
| **Zero Crossing Rate (ZCR)** | `librosa.feature.zero_crossing_rate()` | Tốc độ đổi dấu – phân biệt tiếng kêu liên tục vs. ngắt quãng |
| **Chroma Features** | `librosa.feature.chroma_stft()` | Đặc trưng cao độ (pitch class) – hữu ích cho tiếng chim, ếch |

### R-04.2: Vector đặc trưng phải là 1-D cố định
- Mỗi file audio → đầu ra là **1 vector 1-D** có chiều dài cố định `D`.
- Cách tổng hợp: Với mỗi đặc trưng dạng ma trận (n_features × n_frames), tính **mean** và **std** theo trục thời gian → concat tất cả thành 1 vector duy nhất.
- **Ghi rõ** giá trị `D` trong file `docs/feature_justification.md`.

### R-04.3: Lưu trữ đặc trưng
- Ma trận tổng hợp: `features/feature_db.npy` — shape `(N, D)` với `N` = số file.
- Index mapping: `features/file_index.json` — dict `{0: {"filepath": "...", "species": "..."}, ...}`.
- **Kết quả trung gian** (spectrogram image, từng vector riêng lẻ) lưu vào `features/intermediate/` để phục vụ báo cáo kết quả trung gian (Yêu cầu 3b).

---

## R-05: Hệ thống Tìm kiếm (Search Engine Rules)

### R-05.1: Top-K cố định
- Hàm tìm kiếm **luôn trả về chính xác `top_k = 5`** kết quả.
- Nếu database có ít hơn 5 file, trả về toàn bộ + cảnh báo.

### R-05.2: Output Schema
Kết quả trả về phải đúng cấu trúc sau (JSON-like):
```python
[
    {
        "rank": 1,
        "filepath": "data/processed/cat/cat_042.wav",
        "species": "cat",
        "similarity_score": 0.9821,     # Điểm tương đồng (0–1)
        "distance": 0.0179              # Khoảng cách raw
    },
    # ... 4 kết quả tiếp theo, giảm dần theo similarity_score
]
```

### R-05.3: Metric tương đồng
- **Metric mặc định:** Cosine Similarity (sau khi L2-normalize vector).
- Nếu dùng Euclidean Distance, phải convert sang điểm tương đồng: `similarity = 1 / (1 + distance)`.
- Kết quả **sắp xếp giảm dần** theo `similarity_score`.

### R-05.4: Kiểm thử 2 kịch bản bắt buộc
1. **File CÓ trong CSDL:** Query file từ chính dataset → kết quả rank 1 phải là chính nó (similarity ≈ 1.0).
2. **File KHÔNG CÓ trong CSDL:** Query file hoàn toàn mới → hệ thống vẫn phải trả về 5 file gần nhất, **không crash, không trả rỗng**.

---

## R-06: Tài liệu & Báo cáo (Documentation Rules)

### R-06.1: Giải trình đặc trưng (docs/feature_justification.md)
Phải bao gồm:
- Bảng liệt kê từng đặc trưng + lý do chọn + giá trị thông tin.
- Giải thích tại sao bộ đặc trưng này phù hợp cho **tiếng động vật** (không phải tiếng người / nhạc cụ).
- Ví dụ minh họa bằng biểu đồ (spectrogram/waveform) cho ít nhất 2 loài khác nhau.

### R-06.2: Sơ đồ khối (docs/system_diagram.md)
- Dùng **Mermaid.js** vẽ 2 sơ đồ:
  1. **Indexing Pipeline:** Audio Input → Preprocessing → Feature Extraction → Vector DB.
  2. **Query Pipeline:** Query Audio → Preprocessing → Feature Extraction → Similarity Search → Top-5 Results.
- Mỗi khối phải ghi rõ công nghệ/thư viện sử dụng.

### R-06.3: Kết quả trung gian (docs/search_results_report.md)
- Hiển thị toàn bộ quá trình từ input → đặc trưng → khoảng cách → ranking.
- Bao gồm ảnh chụp spectrogram/waveform của query file và 5 kết quả trả về.

---

## R-07: Demo UI (Application Rules)

### R-07.1: Chức năng tối thiểu
UI Demo **bắt buộc** phải có:
1. ☐ Nút **Upload** file audio (.wav)
2. ☐ Nút **Search** kích hoạt tìm kiếm
3. ☐ Hiển thị **Audio Player** cho file query đầu vào
4. ☐ Hiển thị **Waveform / Spectrogram** cho file query
5. ☐ Bảng kết quả **Top 5** gồm: Rank, Tên file, Loài, Similarity Score
6. ☐ **Audio Player** cho từng file trong Top 5
7. ☐ **Biểu đồ so sánh** waveform/spectrogram giữa query và kết quả

### R-07.2: Không dùng dữ liệu giả
- Mọi dữ liệu hiển thị trên UI phải được tính toán thực từ pipeline.
- **Cấm** hardcode kết quả, cấm dùng random mock data.

---

## R-08: Code Quality

### R-08.1: Docstring & Comments
- Mọi function phải có **docstring** mô tả: mục đích, tham số, kiểu trả về.
- Sử dụng **type hints** cho tất cả function signatures.
- Comment giải thích cho các đoạn logic phức tạp (thuật toán trích xuất, tính khoảng cách).

### R-08.2: Error Handling
- Xử lý lỗi file không tồn tại, file không phải audio, file bị hỏng.
- Trả thông báo lỗi rõ ràng, phân biệt (không dùng generic `Exception`).

### R-08.3: Reproducibility
- Set `random_seed` cố định ở mọi nơi có yếu tố ngẫu nhiên.
- Ghi rõ phiên bản thư viện trong `requirements.txt` (ví dụ: `librosa==0.10.1`).
