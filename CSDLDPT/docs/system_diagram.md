# Sơ Đồ Khối Hệ Thống

## 1. Kiến trúc tổng quan

```mermaid
graph TB
    subgraph Input["📁 Input Layer"]
        RAW["data/raw/<br>Audio gốc (.wav)"]
        UPLOAD["🎤 Upload/Record<br>Query Audio"]
    end

    subgraph Preprocess["⚙️ Preprocessing Pipeline"]
        LOAD["load_audio()<br>mono, 22050 Hz"]
        TRIM["trim silence<br>librosa.effects.trim"]
        PAD["zero-pad / truncate<br>2s = 44100 samples"]
        NORM["normalize amplitude<br>RMS → -20 dB"]
    end

    subgraph Feature["🔬 Feature Extraction (310D)"]
        MFCC["MFCC<br>26D (13 mean+std)"]
        MEL["Mel Spectrogram<br>256D (128 mean+std)"]
        CHROMA["Chroma STFT<br>24D (12 mean+std)"]
        CENT["Spectral Centroid<br>2D (mean+std)"]
        ZCR["ZCR<br>2D (mean+std)"]
        CONCAT["Concatenate → 310D"]
    end

    subgraph Storage["💾 Storage Layer"]
        PG["PostgreSQL<br>Metadata (audio_files,<br>species_stats, search_log)"]
        NPY["features/*.npy<br>Vectors (N×310)"]
        SCALER["feature_scaler.npz<br>z-score mean/std"]
        FAISS["faiss.index<br>IndexFlatIP"]
        FINDEX["file_index.json<br>Metadata map"]
    end

    subgraph Search["🔍 Search Engine"]
        ZSCORE["z-score normalize<br>(query vector)"]
        FWEIGHT["feature weights<br>MFCC×3, Mel×1, Chroma×2,<br>Centroid×2, ZCR×2"]
        L2NORM["L2 normalize"]
        COSINE["Faiss IndexFlatIP<br>= Cosine Similarity"]
        TOP5["Top-5 Results<br>rank, filepath, species,<br>similarity_score, distance"]
    end

    subgraph Output["📊 Output Layer"]
        DEMO["Gradio Demo UI<br>localhost:7860"]
        INTER["features/intermediate/<br>Ảnh + JSON + CSV"]
        DOCS["docs/<br>Báo cáo"]
    end

    RAW --> LOAD
    UPLOAD --> LOAD
    LOAD --> TRIM --> PAD --> NORM

    NORM --> MFCC
    NORM --> MEL
    NORM --> CHROMA
    NORM --> CENT
    NORM --> ZCR
    MFCC --> CONCAT
    MEL --> CONCAT
    CHROMA --> CONCAT
    CENT --> CONCAT
    ZCR --> CONCAT

    CONCAT -->|"Indexing"| NPY
    CONCAT -->|"Indexing"| PG
    NPY --> SCALER --> FAISS
    NPY --> FINDEX

    CONCAT -->|"Query"| ZSCORE
    SCALER -.->|"load scaler"| ZSCORE
    ZSCORE --> FWEIGHT --> L2NORM --> COSINE
    FAISS -.->|"load index"| COSINE
    FINDEX -.->|"load metadata"| TOP5
    COSINE --> TOP5

    TOP5 --> DEMO
    TOP5 --> INTER
    TOP5 --> DOCS
```

## 2. Pipeline chi tiết

### 2.1 Indexing Pipeline (offline)

```mermaid
sequenceDiagram
    participant U as User
    participant PP as preprocess.py
    participant FE as feature.py
    participant DB as database.py<br>(PostgreSQL)
    participant BC as build_canonical.py
    participant FS as Faiss

    U->>PP: python index_balanced8.py
    PP->>PP: load_excluded_filenames()
    PP->>PP: trim + zero-pad + normalize (1042 files)
    PP->>FE: extract_all(y) → 310D vector
    FE-->>DB: insert_record(file_id, filename, species, ...)
    FE-->>DB: save .npy vector
    U->>BC: python build_canonical.py
    BC->>DB: load_all_vectors() → (1042, 310)
    BC->>BC: fit z-score scaler
    BC->>BC: apply feature weights
    BC->>BC: L2 normalize
    BC->>FS: IndexFlatIP.add(scaled_vectors)
    BC-->>FS: write faiss.index
    BC-->>DB: write file_index.json
```

### 2.2 Search Pipeline (online)

```mermaid
sequenceDiagram
    participant U as User / Demo UI
    participant PP as preprocess.py
    participant FE as feature.py
    participant SE as search_engine.py
    participant FS as Faiss Index
    participant FI as file_index.json
    participant DB as PostgreSQL

    U->>PP: Upload audio query
    PP->>PP: trim + zero-pad + normalize
    PP->>FE: extract_all(y) → 310D
    FE->>SE: engine.search(query_vec, top_k=5)
    SE->>SE: z-score scale (dùng DB scaler)
    SE->>SE: apply feature weights
    SE->>SE: L2 normalize
    SE->>FS: index.search(q, k=5)
    FS-->>SE: scores, indices
    SE->>FI: lookup metadata
    SE-->>U: [{rank, filepath, species, similarity_score, distance}]
    SE->>DB: log_search(query, top1, score)
```

## 3. Công nghệ sử dụng

| Layer | Công nghệ | Version |
|---|---|---|
| Language | Python | 3.10+ |
| Audio I/O | librosa, soundfile | 0.10+, 0.12+ |
| Feature | librosa (MFCC, Mel, Chroma, Centroid, ZCR) | 0.10+ |
| Vector Search | Faiss (IndexFlatIP) | 1.9+ |
| Metadata DB | PostgreSQL (via Docker) | 16-alpine |
| Python DB Driver | psycopg2-binary | 2.9+ |
| Demo UI | Gradio | 6.14+ |
| Testing | pytest | 8.3+ |
| Container | Docker Compose | v2 |

## 4. Cấu trúc Database

### PostgreSQL — 3 tables

```sql
audio_files    : id, file_id, filename, species, filepath, duration_sec,
                 sample_rate, source, quality, indexed_at, feat_path
species_stats  : species, file_count, avg_duration_sec, updated_at
search_log     : id, query_file, top1_result, top1_score, searched_at
```

### File-based storage

```
features/feature_db.npy       — (1042, 310) float32
features/feature_scaler.npz   — mean (310,), std (310,), weights (310,)
features/faiss.index           — Faiss IndexFlatIP, 1042 vectors
features/file_index.json       — {idx: {file_id, filepath, species, ...}}
```
