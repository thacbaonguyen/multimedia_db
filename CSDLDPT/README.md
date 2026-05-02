# CSDLDPT — Hệ Thống Truy Vấn Âm Thanh Động Vật

Animal Sound Retrieval System — Đồ án môn Cơ Sở Dữ Liệu Đa Phương Tiện (INT1418)

## Tổng Quan

Hệ thống truy vấn âm thanh dựa trên nội dung (Content-Based Audio Retrieval) cho tiếng kêu động vật.
- **Dataset:** 1042 files, 8 loài (cat, cow, dog, frog, hen, monkey, rooster, sheep)
- **Feature vector:** 310D (MFCC + Mel Spectrogram + Chroma + Spectral Centroid + ZCR)
- **Search:** Pure Cosine Similarity via Faiss IndexFlatIP
- **Database:** PostgreSQL (metadata) + NumPy .npy (vectors) + Faiss (index)

## Setup

### 1. Cài đặt dependencies

```bash
pip install -r requirements.txt
```

### 2. Khởi động PostgreSQL

```bash
cp .env.example .env
docker compose up -d
```

### 3. Kiểm tra kết nối

```bash
python -c "from src.database import check_connection; check_connection(); print('OK')"
```

### 4. Build / Rebuild toàn bộ pipeline

```bash
python src/index_balanced8.py      # Preprocess + Index → PostgreSQL
python src/build_canonical.py      # Faiss + scaler + file_index
```

### 5. Chạy Demo

```bash
python app/demo.py
# Mở http://localhost:7860
```

### Reset DB

```bash
docker compose down -v
docker compose up -d
```

## Cấu Trúc Thư Mục

```
CSDLDPT/
├── app/demo.py                 # Entry point demo UI
├── data/
│   ├── raw/ → balanced8_raw    # Audio gốc (symlink)
│   ├── processed/ → balanced8_processed  # Audio đã chuẩn hóa
│   ├── metadata.csv            # Metadata 7 cột
│   └── excluded_files.csv      # Files loại bỏ
├── features/
│   ├── feature_db.npy          # Ma trận (N, 310)
│   ├── feature_scaler.npz      # z-score scaler
│   ├── faiss.index             # Faiss IndexFlatIP
│   ├── file_index.json         # Metadata index
│   └── intermediate/           # Ảnh + JSON cho báo cáo
├── scripts/init.sql            # PostgreSQL schema
├── src/
│   ├── preprocess.py           # Trim + zero-pad + normalize
│   ├── feature.py              # 310D feature extraction
│   ├── database.py             # PostgreSQL operations
│   ├── search_engine.py        # Pure cosine search
│   ├── build_canonical.py      # Build Faiss + artifacts
│   ├── build_metadata.py       # Build metadata.csv
│   ├── index_balanced8.py      # Full indexing pipeline
│   ├── visualization.py        # Plot functions
│   ├── generate_search_examples.py  # Intermediate results
│   └── exceptions.py           # Custom exceptions
├── tests/                      # pytest test suite
├── docs/                       # Documentation
├── docker-compose.yml          # PostgreSQL container
├── .env.example                # DB config template
└── requirements.txt            # Pinned dependencies
```

## Chạy Tests

```bash
python -m pytest tests/ -v
```

## Vector 310D

| Đặc trưng | Chiều | Mô tả |
|---|---|---|
| MFCC | 26 | 13 mean + 13 std |
| Mel Spectrogram | 256 | 128 mean + 128 std |
| Chroma | 24 | 12 mean + 12 std |
| Spectral Centroid | 2 | mean + std |
| ZCR | 2 | mean + std |
| **Tổng** | **310** | |
