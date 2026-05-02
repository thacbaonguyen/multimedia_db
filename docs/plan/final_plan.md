# Implementation Plan v6 — INT1418

> **Mục tiêu:** 52 → 92-95/100 nếu implement đúng và demo chạy ổn định  
> **Revision:** v5 + fix toàn bộ lỗi PostgreSQL/Docker nhỏ nhưng dễ bị trừ điểm.

## Nguyên tắc phạm vi

- Giữ pipeline hiện có, thêm artifact chuẩn theo `CONVENTIONS.md`.
- Core chấm điểm: dataset sạch, feature 310D, pure cosine/Faiss, top-5 đúng schema, docs/demo/tests đủ.
- SVM/hybrid chỉ là bonus tùy chọn trong demo.
- Không sinh ảnh trung gian cho toàn bộ 1000+ file; chỉ lưu vector toàn bộ + ảnh cho báo cáo.
- **PostgreSQL qua Docker** thay SQLite cho metadata. Faiss vẫn giữ cho vector search.
- Docker chỉ dùng cho DBMS, không dockerize toàn bộ app để tránh quá tay với đồ án 2 người.

---

## Phase Order

```
P0:   Data Audit          → exclude_list, docs/data_audit.md
P1:   Preprocess/Structure → tile→pad, trim, exceptions, .gitignore, requirements, canonical dirs
P1.5: PostgreSQL Migration → docker-compose, refactor database.py, init schema
P2:   Feature 310D        → sửa feature.py, KHÔNG re-index
P3:   Full Rebuild        → re-index + metadata + scaler + Faiss + search_engine + train + evaluate
P4:   Intermediate Results → vectors toàn DB, images cho report
P5:   Demo UI             → audio players, comparison, README
P6:   Documentation       → 3 docs bắt buộc + final_report
```

> [!IMPORTANT]
> Full rebuild chỉ chạy **1 lần duy nhất ở P3**, sau khi P0-P2 hoàn tất.

---

## Phase 0: Data Audit

*(Giữ nguyên v4)*

1. **[NEW] `data/excluded_files.csv`** — filename, species, source, reason, decision, reviewer
2. **Audit:** Quét SoundDino (~45 files) nghe + spectrogram. Sample 5 files/nguồn khác (seed=42).
3. **[NEW] `docs/data_audit.md`** — Audit log chi tiết
4. **Filter mechanism:** Helper `load_excluded_filenames()`, mọi script preprocessing/indexing/metadata/canonical skip `decision=excluded`
5. **Verify:** Tổng files sau exclude ≥ 500

---

## Phase 1: Preprocessing + Structure + Infrastructure

*(Giữ nguyên v4, trừ requirements)*

### 1.1 [NEW] src/exceptions.py

```python
class AudioFileNotFoundError(FileNotFoundError): ...
class AudioFormatError(ValueError): ...
class AudioProcessingError(RuntimeError): ...
```

### 1.2 [MODIFY] src/preprocess.py

- `load_audio()`: raise custom exceptions, type hints
- `normalize_length()`: `librosa.effects.trim()` + zero-pad (không tile), xử lý `len(y)==0`
- `preprocess_audio_for_features()`: shared pipeline cho indexing + query
- `preprocess_all()`: skip excluded files

### 1.3 Canonical Structure

```bash
ln -s balanced8_raw data/raw
ln -s balanced8_processed data/processed
mkdir -p features/intermediate docs tests/fixtures app
cp demo/app.py app/demo.py
```

### 1.4 Requirements

```
librosa==0.10.2
soundfile==0.12.1
numpy==1.26.4
scipy==1.13.1
scikit-learn==1.5.2
pandas==2.2.3
matplotlib==3.9.2
gradio==4.44.1
faiss-cpu==1.9.0.post1
pytest==8.3.3
huggingface_hub==0.26.5
pyarrow==18.1.0
psycopg2-binary==2.9.9
python-dotenv==1.0.1
```

### 1.5 [NEW] .python-version → `3.10`

### 1.6 [NEW] .gitignore

```gitignore
data/raw/
data/processed/
data/balanced8_raw/
data/balanced8_processed/
features/*.npy
features/faiss.index
features/intermediate/
*.wav
*.mp3
*.flac
*.ogg
!tests/fixtures/*.wav
__pycache__/
*.pyc
.pytest_cache/
.env
```

### 1.7 [NEW] tests/fixtures/

- `query_external.wav`, `not_audio.txt`, `silence.wav`

---

## Phase 1.5: PostgreSQL Migration 🆕

### 1.5.1 [NEW] docker-compose.yml

```yaml
services:
  db:
    image: postgres:16-alpine
    container_name: csdldpt_postgres
    environment:
      POSTGRES_DB: ${DB_NAME:-animal_sounds}
      POSTGRES_USER: ${DB_USER:-csdldpt}
      POSTGRES_PASSWORD: ${DB_PASSWORD:-csdldpt123}
    ports:
      - "${DB_PORT:-5432}:5432"
    volumes:
      - pgdata:/var/lib/postgresql/data
      - ./scripts/init.sql:/docker-entrypoint-initdb.d/init.sql
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ${DB_USER:-csdldpt} -d ${DB_NAME:-animal_sounds}"]
      interval: 5s
      timeout: 3s
      retries: 5

volumes:
  pgdata:
```

> `scripts/init.sql` chỉ tự chạy khi volume `pgdata` được tạo lần đầu. Nếu sửa schema hoặc muốn reset DB sạch, dùng `docker compose down -v` rồi `docker compose up -d`.

### 1.5.2 [NEW] .env.example

```env
DB_HOST=localhost
DB_PORT=5432
DB_NAME=animal_sounds
DB_USER=csdldpt
DB_PASSWORD=csdldpt123
```

- Commit `.env.example`, không commit `.env`.
- README hướng dẫn tạo env local:
  ```bash
  cp .env.example .env
  ```

### 1.5.3 [NEW] scripts/init.sql

```sql
CREATE TABLE IF NOT EXISTS audio_files (
    id          SERIAL PRIMARY KEY,
    file_id     TEXT NOT NULL UNIQUE,
    filename    TEXT NOT NULL UNIQUE,
    species     TEXT NOT NULL,
    filepath    TEXT NOT NULL,
    duration_sec REAL,
    sample_rate INTEGER DEFAULT 22050,
    source      TEXT,
    quality     TEXT DEFAULT 'kept',
    indexed_at  TIMESTAMP DEFAULT NOW(),
    feat_path   TEXT
);

CREATE TABLE IF NOT EXISTS species_stats (
    species          TEXT PRIMARY KEY,
    file_count       INTEGER,
    avg_duration_sec REAL,
    updated_at       TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS search_log (
    id          SERIAL PRIMARY KEY,
    query_file  TEXT,
    top1_result TEXT,
    top1_score  REAL,
    searched_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_audio_species ON audio_files(species);
CREATE INDEX IF NOT EXISTS idx_audio_filename ON audio_files(filename);
CREATE INDEX IF NOT EXISTS idx_audio_file_id ON audio_files(file_id);
```

### 1.5.4 [MODIFY] src/database.py — Refactor SQLite → PostgreSQL

**Thay đổi chính:**

| Trước (SQLite) | Sau (PostgreSQL) |
|---|---|
| `import sqlite3` | `import psycopg2` + `from dotenv import load_dotenv` |
| `sqlite3.connect(db_path)` | `psycopg2.connect(host, port, dbname, user, password)` |
| Placeholder `?` | Placeholder `%s` |
| `AUTOINCREMENT` | `SERIAL` |
| `DB_PATH = "...db"` | `get_connection()` → reads `.env` |
| Mỗi function mở/đóng file | Connection pool hoặc context manager |

**Code mẫu:**

```python
import os
import psycopg2
from contextlib import contextmanager
from dotenv import load_dotenv

load_dotenv()

DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": int(os.getenv("DB_PORT", 5432)),
    "dbname": os.getenv("DB_NAME", "animal_sounds"),
    "user": os.getenv("DB_USER", "csdldpt"),
    "password": os.getenv("DB_PASSWORD", "csdldpt123"),
}

FEATURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'features')
SCALER_PATH  = os.path.join(FEATURES_DIR, 'feature_scaler.npz')
FEATURE_DIM  = 310

@contextmanager
def get_connection():
    conn = psycopg2.connect(**DB_CONFIG)
    try:
        yield conn
    finally:
        conn.close()

def check_connection() -> None:
    """Kiểm tra PostgreSQL sẵn sàng cho pipeline/demo."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT 1")
            cur.fetchone()

def init_db():
    """PostgreSQL tables — cũng có thể dùng scripts/init.sql khi docker compose up"""
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS audio_files (
                id SERIAL PRIMARY KEY,
                file_id TEXT NOT NULL UNIQUE,
                filename TEXT NOT NULL UNIQUE,
                species TEXT NOT NULL,
                filepath TEXT NOT NULL,
                duration_sec REAL,
                sample_rate INTEGER DEFAULT 22050,
                source TEXT,
                quality TEXT DEFAULT 'kept',
                indexed_at TIMESTAMP DEFAULT NOW(),
                feat_path TEXT
            )
        """)
        # ... species_stats, search_log tương tự
        conn.commit()

def insert_record(file_id, filename, species, filepath, duration_sec, source,
                  feature_vec, quality="kept", features_dir=FEATURES_DIR):
    os.makedirs(features_dir, exist_ok=True)
    feat_name = filename.replace('.wav', '.npy')
    feat_path = os.path.join(features_dir, feat_name)
    np.save(feat_path, feature_vec.astype(np.float32))

    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO audio_files
                (file_id, filename, species, filepath, duration_sec, sample_rate, source, quality, feat_path)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (filename) DO UPDATE SET
                file_id=EXCLUDED.file_id,
                species=EXCLUDED.species,
                filepath=EXCLUDED.filepath,
                duration_sec=EXCLUDED.duration_sec,
                sample_rate=EXCLUDED.sample_rate,
                source=EXCLUDED.source,
                quality=EXCLUDED.quality,
                feat_path=EXCLUDED.feat_path,
                indexed_at=NOW()
        """, (file_id, filename, species, filepath, duration_sec, 22050, source, quality, feat_path))
        conn.commit()

def load_all_vectors(scaled=False, scaler_path=SCALER_PATH):
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT id, file_id, filename, species, filepath, duration_sec, source, feat_path
            FROM audio_files
            WHERE quality = 'kept'
            ORDER BY id
        """)
        rows = cur.fetchall()
    # ... load .npy files giống hiện tại
```

**Các function cần sửa (tất cả trong database.py):**

| Function | Thay đổi |
|---|---|
| `init_db()` | `sqlite3.connect` → `get_connection()`, `?` → `%s` |
| `check_connection()` | [NEW] dùng cho README/tests để verify PostgreSQL |
| `insert_record()` | `INSERT OR REPLACE` → `ON CONFLICT DO UPDATE`, thêm `file_id`, `duration_sec`, `source`, `quality` |
| `get_all_records()` | Context manager |
| `update_species_stats()` | `INSERT OR REPLACE` → `ON CONFLICT DO UPDATE` |
| `get_db_stats()` | Context manager |
| `log_search()` | Context manager, `%s` |
| `load_all_vectors()` | Bỏ `db_path` param → dùng `get_connection()`, chỉ lấy `quality='kept'` |
| `save_feature_scaler()` | Giữ nguyên (chỉ dùng numpy, không liên quan DB) |
| `load_feature_scaler()` | Giữ nguyên |

### 1.5.5 [MODIFY] README.md — Thêm hướng dẫn setup

```markdown
## Setup

### 1. Khởi động PostgreSQL
cp .env.example .env
docker compose up -d

### 2. Cài dependencies
pip install -r requirements.txt

### 3. Kiểm tra kết nối
python -c "from src.database import check_connection; check_connection(); print('OK')"

### Reset DB khi đổi schema hoặc muốn rebuild sạch
docker compose down -v
docker compose up -d
```

### 1.5.6 Verify

```bash
docker compose up -d
docker compose exec db psql -U csdldpt -d animal_sounds -c "\dt"  # 3 tables
python -c "from src.database import init_db, check_connection; init_db(); check_connection(); print('DB OK')"
```

---

## Phase 2: Feature Extraction 310D

*(Giữ nguyên v4)*

### [MODIFY] src/feature.py

- Thêm `extract_mel_spectrogram()` (256D)
- Sửa `extract_spectral_centroid()` (2D: mean+std)
- Sửa `extract_all()`: MFCC(26) + Mel(256) + Chroma(24) + Centroid(2) + ZCR(2) = **310D**
- Bỏ bandwidth/rolloff/flatness/RMS khỏi vector
- Sửa `extract_from_file()`: dùng `preprocess_audio_for_features()` cho consistency
- `FEATURE_DIM = 310`, `assert len(FEATURE_NAMES) == 310`

### [MODIFY] src/database.py

- `FEATURE_DIM = 310`

---

## Phase 3: Full Rebuild

*(Giữ nguyên v4, PostgreSQL thay SQLite)*

### 3.1 Re-index

```bash
docker compose up -d                  # Đảm bảo PostgreSQL đang chạy
python src/index_balanced8.py         # Insert metadata → PostgreSQL, vectors → .npy
```

- Clear tables trước khi re-insert (`TRUNCATE audio_files, species_stats, search_log`)
- Dùng `TRUNCATE audio_files, species_stats, search_log RESTART IDENTITY CASCADE;` để reset id và tránh metadata cũ còn sót
- Skip excluded files
- Filepath = relative paths

### 3.2 Canonical Artifacts + Scaler

#### [NEW] src/build_canonical.py

- `features/feature_db.npy` — ma trận (N, 310) raw
- `features/feature_scaler.npz` — z-score mean/std
- `features/faiss.index` — Faiss IndexFlatIP trên scaled+L2-normed vectors
- `features/file_index.json` — đầy đủ metadata: file_id, filepath, filename, species, duration_sec, sample_rate, source, quality

#### [NEW] src/build_metadata.py

- `data/metadata.csv` — 7 cột (file_id, filename, species, filepath, duration_sec, sample_rate, source)
- Source lấy từ `data/balanced8_metadata.csv`, filepath = relative

### 3.3 Search Engine

#### [NEW] src/search_engine.py

- `AnimalSoundSearchEngine`: z-score → L2-normalize → Faiss IndexFlatIP
- Pure cosine default, output schema: rank, filepath, species, similarity_score, distance
- `similarity_score = clip((cosine+1)/2, 0, 1)`

#### [MODIFY] src/search.py

- Default `classifier_guided=False` → pure cosine
- Optional `classifier_guided=True` → SVM hybrid

### 3.4 Re-train + Re-evaluate

```bash
python src/build_metadata.py
python src/build_canonical.py
python src/train_classifier.py
python src/evaluate_balanced8.py
```

### 3.5 Tests

#### [NEW] tests/test_preprocessing.py
- trim, zero-pad, shape, mono, exceptions

#### [NEW] tests/test_feature_extraction.py
- vector shape 310, feature names count, sub-feature dims

#### [NEW] tests/test_search_engine.py
- self_match_rank1 (score ≥ 0.999 AND filepath match)
- external_query_returns_5
- output_schema (exact keys + 0≤score≤1)
- invalid_format → AudioFormatError
- sorted_descending

#### [NEW] tests/test_database.py 🆕
- `test_connection()` — connect thành công
- `test_insert_and_query()` — insert → select → verify
- `test_species_stats()` — update → count đúng
- Test DB phải dùng schema test hoặc dữ liệu tạm, không phá database chính đã index.

---

## Phase 4: Intermediate Results

*(Giữ nguyên v4)*

### [NEW] src/visualization.py
- `save_waveform`, `save_mel_spectrogram`, `save_mfcc_heatmap`, `save_comparison`, `save_similarity_bar`

### [NEW] src/generate_search_examples.py
- 2 kịch bản → `features/intermediate/search_scenario_{1,2}/`
- Mỗi kịch bản: query_info.json, query_vector.npy, ranking.json, ranking_full.csv, ảnh

---

## Phase 5: Demo UI

*(Giữ nguyên v4)*

- Waveform + Spectrogram query
- Audio Player **từng file** Top-5
- Comparison chart query vs **5** results
- Default: pure cosine. Toggle: SVM hybrid
- README cập nhật lệnh chạy
- Stub `demo/app.py` redirect → `app/demo.py`

---

## Phase 6: Documentation

*(Giữ nguyên v4)*

- `docs/feature_justification.md` — R-06.1
- `docs/system_diagram.md` — R-06.2 (Mermaid, cập nhật PostgreSQL trong diagram)
- `docs/search_results_report.md` — R-06.3
- `docs/final_report.md` — layout example.pdf

---

## Verification Checklist

| # | Issue | Phase | Verify |
|---|---|---|---|
| R1#1 | 310D exact | P2 | `assert extract_all(y).shape == (310,)` |
| R1#2 | Pure cosine default | P3 | `search_engine.search()` no hybrid |
| R1#3 | Faiss IndexFlatIP | P3 | `features/faiss.index` exists |
| R1#4 | Cấu trúc R-02 | P1 | `ls data/raw data/processed features/ docs/ tests/ app/` |
| R1#5 | Metadata 7 cột | P3 | `head -1 data/metadata.csv` |
| R1#6 | External query test | P3 | `pytest tests/test_search_engine.py::test_external_query` |
| R1#7 | `features/intermediate/` | P4 | Images + .npy + JSON exist |
| R1#8 | Zero-pad no tile | P1 | `grep -r "np.tile" src/preprocess.py` → nothing |
| R1#9 | Requirements pin | P1 | `pip install -r requirements.txt` succeeds |
| R2#1 | Exclude mechanism | P0 | `data/excluded_files.csv` + filter |
| R2#2 | Phase order | All | Rebuild only P3 |
| R2#3 | z-score before Faiss | P3 | `features/feature_scaler.npz` exists |
| R2#4 | Pin == versions | P1 | `grep "==" requirements.txt` |
| R2#5 | Custom exceptions | P1 | `src/exceptions.py` exists |
| R2#6 | Audit spectrogram | P0 | `docs/data_audit.md` |
| R2#7 | Rich file_index.json | P3 | file_id, filepath, filename, species, duration_sec, sample_rate, source, quality |
| R2#8 | Self-match filepath | P3 | Test asserts filepath |
| R2#9 | README + paths | P5 | `python app/demo.py` works |
| **PG#1** | **Docker PostgreSQL** | **P1.5** | **`docker compose up -d` + `\dt` shows 3 tables** |
| **PG#2** | **database.py refactor** | **P1.5** | **`psycopg2` import, `%s` params, context manager, `check_connection()`** |
| **PG#3** | **DB tests** | **P3** | **`pytest tests/test_database.py` passes** |
| **PG#4** | **.env not committed** | **P1** | **`.gitignore` has `.env`** |
| **PG#5** | **.env.example committed** | **P1.5** | **README has `cp .env.example .env`** |
| **PG#6** | **DB schema matches metadata** | **P1.5/P3** | **audio_files has `file_id`, `duration_sec`, `source`, `quality`** |
| **PG#7** | **Docker healthcheck valid** | **P1.5** | **No `CMD-ONLY`; uses `CMD-SHELL` or `CMD`** |
| **PG#8** | **Clean rebuild reset** | **P3** | **TRUNCATE uses `RESTART IDENTITY CASCADE`** |

---

## Open Questions

> [!IMPORTANT]
> **Q1: Thông tin trang bìa?** Tên + MSSV thành viên nhóm.
>
> **Q2: SoundDino audit?** Mặc định: nghe/xem spectrogram toàn bộ SoundDino, sample 5/nguồn khác.
