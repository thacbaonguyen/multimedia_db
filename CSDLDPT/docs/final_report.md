# BÁO CÁO ĐỒ ÁN MÔN HỌC
# CƠ SỞ DỮ LIỆU ĐA PHƯƠNG TIỆN (INT1418)

## HỆ THỐNG TRUY VẤN ÂM THANH ĐỘNG VẬT
### Animal Sound Retrieval System

---

**Giảng viên hướng dẫn:** [Tên giảng viên]

**Sinh viên thực hiện:**
- [Họ tên 1] — MSSV: [MSSV1]
- [Họ tên 2] — MSSV: [MSSV2]

**Ngày nộp:** [DD/MM/YYYY]

---

## MỤC LỤC

1. [Giới thiệu](#1-giới-thiệu)
2. [Thu thập dữ liệu](#2-thu-thập-dữ-liệu)
3. [Trích xuất đặc trưng](#3-trích-xuất-đặc-trưng)
4. [Xây dựng hệ thống tìm kiếm](#4-xây-dựng-hệ-thống-tìm-kiếm)
5. [Kết quả trung gian](#5-kết-quả-trung-gian)
6. [Demo và kiểm thử](#6-demo-và-kiểm-thử)
7. [Kết luận](#7-kết-luận)
8. [Tài liệu tham khảo](#8-tài-liệu-tham-khảo)
9. [Phụ lục](#9-phụ-lục)

---

## 1. Giới thiệu

### 1.1 Bối cảnh

Content-Based Audio Retrieval (CBAR) là bài toán tìm kiếm file âm thanh dựa trên nội dung thay vì metadata. Ứng dụng rộng rãi trong giám sát hệ sinh thái, nhận dạng loài, và quản lý kho dữ liệu đa phương tiện.

### 1.2 Mục tiêu

Xây dựng hệ thống truy vấn âm thanh tiếng kêu động vật:
- Thu thập ≥ 500 file âm thanh tiếng kêu động vật
- Trích xuất vector đặc trưng 310 chiều
- Xây dựng hệ thống tìm kiếm top-5 tương tự nhất
- Demo giao diện web cho phép upload/thu âm và tra cứu

### 1.3 Phạm vi

- **8 loài:** Cat, Cow, Dog, Frog, Hen, Monkey, Rooster, Sheep
- **1042 files** sau audit (loại bỏ 9 files multi-species/non-vocalization)
- **Phương pháp:** Pure Cosine Similarity via Faiss IndexFlatIP
- **Công nghệ:** Python 3.10+, PostgreSQL, Faiss, Gradio

---

## 2. Thu thập dữ liệu

### 2.1 Nguồn dữ liệu

| Nguồn | Số files | Mô tả |
|---|---|---|
| local | 442 | Thu thập thủ công |
| ESC-50 | 269 | Environmental Sound Classification dataset |
| DynamicSuperb | 90 | Dataset benchmark cho audio understanding |
| AnimalQA | 80 | Animal audio question answering dataset |
| SoundDino | 36 | Sound effects curated (sau audit) |
| Khác | 125 | Các nguồn bổ sung |
| **Tổng** | **1042** | |

### 2.2 Phân bố theo loài

| Loài | Số files | Tỷ lệ |
|---|---|---|
| Dog | 160 | 15.4% |
| Cat | 159 | 15.3% |
| Monkey | 150 | 14.4% |
| Sheep | 130 | 12.5% |
| Cow | 123 | 11.8% |
| Frog | 120 | 11.5% |
| Hen | 100 | 9.6% |
| Rooster | 100 | 9.6% |

### 2.3 Data Audit

Đã thực hiện audit toàn bộ 45 SoundDino files:
- **9 files bị loại:** multi-species (tiếng kêu ≥ 2 loài trộn lẫn) hoặc non-vocalization
- Kết quả chi tiết: `data/excluded_files.csv` và `docs/data_audit.md`
- Dataset sau audit: 1042 files (vẫn > 500 yêu cầu tối thiểu)

### 2.4 Tiền xử lý

Pipeline chuẩn hóa cho mọi audio file:

```
1. Load audio      → mono, 22050 Hz (librosa)
2. Trim silence    → librosa.effects.trim(top_db=20)
3. Zero-pad/Truncate → 2 giây = 44100 samples
4. Normalize amplitude → RMS target -20 dB
5. Save             → WAV, 16-bit PCM
```

> **Lưu ý:** Sử dụng zero-pad (không tile/repeat) để tránh tạo artifact giả khi file ngắn hơn 2 giây.

---

## 3. Trích xuất đặc trưng

### 3.1 Vector 310D

| Đặc trưng | Mô tả | Số chiều |
|---|---|---|
| **MFCC** | Mel-Frequency Cepstral Coefficients (13 mean + 13 std) | 26 |
| **Mel Spectrogram** | Phân bố năng lượng trên 128 Mel bands (128 mean + 128 std) | 256 |
| **Chroma STFT** | 12 pitch classes (12 mean + 12 std) | 24 |
| **Spectral Centroid** | Trung tâm khối phổ tần (mean + std) | 2 |
| **ZCR** | Zero-Crossing Rate (mean + std) | 2 |
| | **Tổng** | **310** |

### 3.2 Cơ sở lựa chọn

Mỗi đặc trưng capture một khía cạnh khác nhau của âm thanh:
- **MFCC:** Timbre (âm sắc) — phân biệt vocal tract
- **Mel Spectrogram:** Temporal patterns — burst vs. continuous vs. harmonic
- **Chroma:** Pitch structure — melodic vs. noise-like
- **Centroid:** Brightness — loài nhỏ (cao) vs. loài lớn (thấp)
- **ZCR:** Voiced/unvoiced — thú (thấp) vs. côn trùng (cao)

Chi tiết giải trình: `docs/feature_justification.md`

### 3.3 Tham số kỹ thuật

```python
SAMPLE_RATE = 22050   # Hz
N_FFT       = 2048    # FFT window
HOP_LENGTH  = 512     # Hop giữa các frames
N_MFCC      = 13      # Số hệ số MFCC
N_MELS      = 128     # Số Mel bands
N_CHROMA    = 12      # Số pitch classes
```

---

## 4. Xây dựng hệ thống tìm kiếm

### 4.1 Sơ đồ kiến trúc

Chi tiết: `docs/system_diagram.md`

Kiến trúc 3 tầng:
1. **Storage:** PostgreSQL (metadata) + NumPy .npy (vectors) + Faiss (index)
2. **Processing:** Preprocessing → Feature extraction → z-score → feature weights → L2-norm
3. **Search:** Faiss IndexFlatIP (= Cosine Similarity) → Top-5

### 4.2 Cơ sở dữ liệu

- **PostgreSQL 16-alpine** (Docker container) cho metadata
- **3 tables:** `audio_files`, `species_stats`, `search_log`
- **Indexes:** species, filename, file_id
- Tệp `.npy` cho vector đặc trưng (tránh BLOB trong DB)

### 4.3 Indexing pipeline

```
PostgreSQL + .npy files
        ↓
feature_db.npy (1042 × 310)
        ↓
z-score normalize (fit trên toàn DB)
        ↓
feature weights (MFCC×3, Mel×1, Chroma×2, Centroid×2, ZCR×2)
        ↓
L2 normalize
        ↓
Faiss IndexFlatIP (Inner Product = Cosine trên L2-normed vectors)
```

### 4.4 Search pipeline

```
Query audio → preprocess (cùng pipeline với indexing)
           → extract 310D vector
           → z-score (dùng scaler đã fit)
           → apply feature weights
           → L2 normalize
           → Faiss search (top-5)
           → scores: similarity = clip(cosine, 0, 1)
           → output: {rank, filepath, species, similarity_score, distance}
```

### 4.5 Output schema (R-05.2)

```json
[
  {
    "rank": 1,
    "filepath": "data/processed/cat_local_xxx.wav",
    "species": "cat",
    "similarity_score": 0.9046,
    "distance": 0.0954
  }
]
```

---

## 5. Kết quả trung gian

### 5.1 Feature examples

Minh họa waveform, Mel spectrogram, MFCC cho 4 loài (cat, dog, frog, cow):
- Thư mục: `features/intermediate/feature_examples/`
- Mỗi loài: 3 hình (waveform, spectrogram, mfcc)

### 5.2 Canonical artifacts

| Artifact | Đường dẫn | Kích thước |
|---|---|---|
| Feature matrix | `features/feature_db.npy` | (1042, 310) |
| Z-score scaler | `features/feature_scaler.npz` | mean + std (310D) |
| Faiss index | `features/faiss.index` | 1042 vectors |
| File index | `features/file_index.json` | 1042 entries |
| Metadata | `data/metadata.csv` | 1042 rows, 7 cột |

---

## 6. Demo và kiểm thử

### 6.1 Demo UI

- **Framework:** Gradio 6.14
- **URL:** `http://localhost:7860`
- **Chức năng:**
  - Upload file hoặc thu âm trực tiếp
  - Hiển thị waveform + spectrogram query
  - Audio player cho **từng file** Top-5
  - Similarity bar chart so sánh
  - Database stats panel

### 6.2 Kịch bản kiểm thử

#### Kịch bản 1: File CÓ trong CSDL

- Query: `cat_local_B_ANI01_MC_FN_SIM01_101.wav`
- **Kết quả:** Rank 1 = chính file query, sim = 1.0000
- **Precision@5 = 100%** (5/5 đều là cat)

#### Kịch bản 2: File KHÔNG trong CSDL

- Query: `query_external_dog.wav` (dog + noise)
- **Kết quả:** Top-5 scores trong [0.8496, 0.8572], không có self-match
- Xác nhận hệ thống không false positive

Chi tiết: `docs/search_results_report.md`

### 6.3 Unit Tests

```
42 passed in 3.54s
├── test_preprocessing.py      (11 tests)
├── test_feature_extraction.py (17 tests)
└── test_search_engine.py      (14 tests)
```

---

## 7. Kết luận

### 7.1 Kết quả đạt được

- ✅ Thu thập **1042 files** tiếng kêu **8 loài** động vật (> 500 yêu cầu)
- ✅ Vector đặc trưng **310D** với 5 nhóm feature bổ sung lẫn nhau
- ✅ Hệ thống tìm kiếm **Pure Cosine Similarity** qua Faiss, self-match = 1.0
- ✅ Feature-space evaluation: Top-1 cùng loài 90.31%, Precision@5 73.01%
- ✅ Demo UI hoàn chỉnh với audio players và visualization
- ✅ **42/42 tests** passed, documentation đầy đủ
- ✅ PostgreSQL cho metadata, reproducible pipeline

### 7.2 Hạn chế

- External query với noise mạnh có thể trả về sai loài
- Chưa xử lý real-world noise (traffic, wind) ở mức production
- Dataset chưa đủ lớn cho deep learning approaches

### 7.3 Hướng phát triển

- Thêm loài mới và mở rộng dataset
- Tích hợp deep audio embeddings (VGGish, AudioMAE)
- Real-time streaming search
- Mobile application

---

## 8. Tài liệu tham khảo

1. Davis, S., & Mermelstein, P. (1980). Comparison of parametric representations for monosyllabic word recognition. *IEEE Trans. ASSP*.
2. Logan, B. (2000). Mel Frequency Cepstral Coefficients for Music Modeling. *ISMIR*.
3. Johnson, J., Douze, M., & Jégou, H. (2019). Billion-scale similarity search with GPUs. *IEEE Trans. Big Data*.
4. McFee, B., et al. (2015). librosa: Audio and Music Signal Analysis in Python. *SciPy*.
5. Piczak, K. J. (2015). ESC: Dataset for Environmental Sound Classification. *ACM MM*.

---

## 9. Phụ lục

### A. Cấu trúc thư mục

```
CSDLDPT/
├── app/demo.py              # Demo UI entry point
├── data/
│   ├── raw/ → balanced8_raw
│   ├── processed/ → balanced8_processed
│   ├── metadata.csv
│   └── excluded_files.csv
├── features/
│   ├── feature_db.npy
│   ├── feature_scaler.npz
│   ├── faiss.index
│   ├── file_index.json
│   └── intermediate/
├── src/                     # Source code
├── tests/                   # pytest suite
├── docs/                    # Documentation
├── scripts/init.sql         # PostgreSQL schema
├── docker-compose.yml
└── requirements.txt
```

### B. Hướng dẫn chạy

```bash
# 1. Setup
pip install -r requirements.txt
cp .env.example .env
docker compose up -d

# 2. Build pipeline
python src/index_balanced8.py
python src/build_canonical.py

# 3. Run demo
python app/demo.py
# → http://localhost:7860

# 4. Run tests
python -m pytest tests/ -v
```

### C. Tài liệu chi tiết

| Tài liệu | Đường dẫn |
|---|---|
| Giải trình đặc trưng | `docs/feature_justification.md` |
| Đánh giá feature space | `docs/feature_space_evaluation.md` |
| Sơ đồ hệ thống | `docs/system_diagram.md` |
| Kết quả tìm kiếm | `docs/search_results_report.md` |
| Data audit | `docs/data_audit.md` |
