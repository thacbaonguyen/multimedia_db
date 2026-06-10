# CSDLDPT - Walkthrough Chi Tiết Toàn Bộ Dự Án

Tài liệu này dành cho người mới học dự án. Mục tiêu là giúp bạn đọc được toàn bộ code theo đúng luồng chạy thật: file nào làm gì, hàm nào nhận đầu vào nào, sinh đầu ra nào, artifact nào phụ thuộc vào artifact nào, và khi sửa một phần thì cần kiểm tra những gì.

Bản refactor này dùng tiếng Việt có dấu, giữ lại nội dung kỹ thuật từ bản scan trước và sắp xếp lại để dễ học hơn.

Thông tin đã đối chiếu từ project hiện tại:

| Hạng mục | Giá trị |
|---|---:|
| Metadata | `data/metadata.csv`, 1042 dòng, 7 cột |
| Số loài | 8 |
| Feature matrix | `features/feature_db.npy`, shape `(1042, 310)` |
| Scaler | `features/feature_scaler.npz`, gồm `mean`, `std`, `weights` |
| File index | `features/file_index.json`, 1042 entries |
| Unit tests | 42 passed, 5 warnings |

Các loài trong CSDL: `cat`, `cow`, `dog`, `frog`, `hen`, `monkey`, `rooster`, `sheep`.

---

## 1. Dự Án Này Là Gì?

Dự án xây dựng hệ thống truy vấn âm thanh tiếng động vật theo nội dung. Người dùng đưa vào một file audio, hệ thống không tìm theo tên file mà tự trích xuất đặc trưng âm học rồi trả về 5 file trong CSDL giống nhất.

Luồng ý tưởng:

```text
Audio đầu vào
  -> tiền xử lý tín hiệu
  -> trích xuất vector đặc trưng 310 chiều
  -> chuẩn hóa vector giống CSDL
  -> tìm kiếm cosine similarity bằng Faiss
  -> trả Top-5 file giống nhất
```

Điểm cần nhớ:

- Đây là hệ thống retrieval, không phải classifier.
- `species` được dùng để hiển thị và đánh giá, không phải đầu ra được mô hình dự đoán trực tiếp.
- Query và database phải đi qua cùng pipeline preprocessing và cùng phép biến đổi vector, nếu không điểm similarity sẽ không còn đáng tin.
- `feature_db.npy` lưu vector thô, còn `faiss.index` lưu vector đã z-score, nhân weight và L2-normalize.

---

## 2. Cấu Trúc Thư Mục

```text
int1418/
├── CONVENTIONS.md
├── context/
│   ├── problem.md
│   ├── rules.md
│   ├── skills.md
│   └── data_collection_guide.md
├── docs/
│   ├── CSDLDPT-Nhom3.md
│   ├── CSDLDPT-Nhom3.pdf
│   └── images/
└── CSDLDPT/
    ├── README.md
    ├── requirements.txt
    ├── docker-compose.yml
    ├── scripts/init.sql
    ├── app/demo.py
    ├── src/
    │   ├── exceptions.py
    │   ├── preprocess.py
    │   ├── feature.py
    │   ├── database.py
    │   ├── index_balanced8.py
    │   ├── build_metadata.py
    │   ├── build_canonical.py
    │   ├── search_engine.py
    │   ├── visualization.py
    │   ├── generate_search_examples.py
    │   └── evaluate_feature_space.py
    ├── data/
    │   ├── balanced8_raw/
    │   ├── balanced8_processed/
    │   ├── raw -> balanced8_raw
    │   ├── processed -> balanced8_processed
    │   ├── metadata.csv
    │   ├── balanced8_metadata.csv
    │   └── excluded_files.csv
    ├── features/
    │   ├── feature_db.npy
    │   ├── feature_scaler.npz
    │   ├── faiss.index
    │   ├── file_index.json
    │   ├── <filename>.npy
    │   └── intermediate/
    ├── tests/
    └── docs/
```

Ý nghĩa từng nhóm:

| Nhóm | Vai trò |
|---|---|
| `context/` | Đề bài, luật và kỹ năng bắt buộc của đồ án |
| `CSDLDPT/src/` | Code xử lý dữ liệu, đặc trưng, database, indexing, search, đánh giá |
| `CSDLDPT/app/` | Demo UI bằng Gradio |
| `CSDLDPT/data/` | CSV metadata và audio raw/processed |
| `CSDLDPT/features/` | Vector `.npy`, scaler, Faiss index, kết quả trung gian |
| `CSDLDPT/tests/` | Test đảm bảo hành vi cốt lõi |
| `docs/` | Báo cáo nộp môn học |
| `CSDLDPT/docs/` | Tài liệu kỹ thuật nội bộ của project |

Lưu ý về symlink:

- `CSDLDPT/data/raw` trỏ tới `balanced8_raw`.
- `CSDLDPT/data/processed` trỏ tới `balanced8_processed`.
- Code build thường đọc `balanced8_processed`.
- Metadata và `file_index.json` lưu path dạng `data/processed/<filename>`.

---

## 3. Hai Luồng Chạy Chính

### 3.1 Luồng Offline: Build CSDL Và Faiss Index

Chạy từ thư mục `CSDLDPT/`:

```bash
python src/index_balanced8.py
python src/build_canonical.py
```

`index_balanced8.py` làm nhiệm vụ từ audio thô đến PostgreSQL và vector `.npy`:

```text
data/balanced8_raw/
  -> preprocess_all()
  -> data/balanced8_processed/
  -> extract_from_file()
  -> insert_record()
  -> PostgreSQL audio_files
  -> features/<filename>.npy
  -> data/metadata.csv
```

`build_canonical.py` làm nhiệm vụ từ vector `.npy` đến artifact dùng cho search:

```text
PostgreSQL audio_files
  -> load_all_vectors()
  -> feature_db.npy
  -> feature_scaler.npz
  -> z-score + feature weights + L2 normalize
  -> faiss.index
  -> file_index.json
```

### 3.2 Luồng Online: Search Khi Người Dùng Upload Audio

Chạy demo:

```bash
python app/demo.py
```

Khi người dùng bấm Search:

```text
audio_path từ Gradio
  -> preprocess_audio_for_features()
  -> extract_all()
  -> engine.search(query_vec, top_k=5)
  -> lookup file_index.json
  -> render HTML, waveform, spectrogram, chart
```

Bên trong `engine.search()`:

```text
query vector thô 310D
  -> z-score bằng mean/std đã fit trên database
  -> nhân feature weights
  -> L2 normalize
  -> Faiss IndexFlatIP search
  -> similarity_score và distance
  -> Top-5 results
```

---

## 4. Dataset Và Artifact Hiện Có

### 4.1 Phân Bố Theo Loài

| Loài | Số file |
|---|---:|
| cat | 159 |
| cow | 123 |
| dog | 160 |
| frog | 120 |
| hen | 100 |
| monkey | 150 |
| rooster | 100 |
| sheep | 130 |
| Tổng | 1042 |

### 4.2 Phân Bố Theo Nguồn

| Nguồn | Số file |
|---|---:|
| local | 442 |
| esc50 | 269 |
| unknown | 125 |
| dynamicsuperb | 90 |
| animalqa | 80 |
| sounddino | 36 |

### 4.3 Artifact Quan Trọng

| Artifact | Vai trò |
|---|---|
| `data/metadata.csv` | Metadata 1042 file, gồm `file_id`, `filename`, `species`, `filepath`, `duration_sec`, `sample_rate`, `source` |
| `features/<filename>.npy` | Vector 310D thô cho từng file |
| `features/feature_db.npy` | Ma trận vector thô shape `(1042, 310)` |
| `features/feature_scaler.npz` | Mean, std và weights dùng để transform vector |
| `features/faiss.index` | Index Faiss dùng cho truy vấn nhanh |
| `features/file_index.json` | Map index Faiss sang metadata file |
| `features/intermediate/` | Ảnh, JSON, CSV phục vụ báo cáo kết quả trung gian |

Điểm dễ nhầm:

- Thư mục processed hiện có thể chứa 1051 file `.wav`, nhưng metadata/index chỉ giữ 1042 file sau audit.
- 9 file bị loại nằm trong `data/excluded_files.csv`.
- Vector `.npy` theo từng file có 1042 file, khớp với metadata và index.

---

## 5. Vector Đặc Trưng 310D

File định nghĩa chính: `src/feature.py`.

Layout vector:

| Vị trí | Nhóm đặc trưng | Số chiều | Cách tạo |
|---|---|---:|---|
| `0:26` | MFCC | 26 | 13 mean + 13 std |
| `26:282` | Mel Spectrogram | 256 | 128 mean + 128 std |
| `282:306` | Chroma | 24 | 12 mean + 12 std |
| `306:308` | Spectral Centroid | 2 | mean + std |
| `308:310` | ZCR | 2 | mean + std |

Tổng:

```text
26 + 256 + 24 + 2 + 2 = 310
```

Trước khi đưa vào Faiss hoặc search, vector thô được transform như sau:

```python
safe_std = np.where(std < 1e-8, 1.0, std)
scaled = (raw_vector - mean) / safe_std
weighted = scaled * weights
normalized = weighted / ||weighted||
```

Weights hiện tại:

| Nhóm | Weight |
|---|---:|
| MFCC | 3.0 |
| Mel Spectrogram | 1.0 |
| Chroma | 2.0 |
| Spectral Centroid | 2.0 |
| ZCR | 2.0 |

Lý do cần weight:

- Mel Spectrogram chiếm 256/310 chiều, tức hơn 82% vector.
- Nếu tính cosine trực tiếp, Mel dễ áp đảo các nhóm nhỏ hơn.
- MFCC chỉ có 26 chiều nhưng rất quan trọng cho âm sắc, nên được nhân 3.
- Chroma, Centroid và ZCR ít chiều nhưng bổ sung thông tin pitch, độ sáng và voiced/unvoiced, nên được nhân 2.

---

## 6. Giải Thích Từng File Source

### 6.1 `src/exceptions.py`

File này định nghĩa lỗi riêng cho audio pipeline.

| Class | Kế thừa | Khi nào dùng |
|---|---|---|
| `AudioFileNotFoundError` | `FileNotFoundError` | Đường dẫn audio không tồn tại |
| `AudioFormatError` | `ValueError` | Extension không hợp lệ |
| `AudioProcessingError` | `RuntimeError` | `librosa` không đọc được audio hoặc file bị lỗi |

Vì sao cần custom exception:

- Test có thể kiểm tra đúng loại lỗi.
- UI hoặc caller có thể hiển thị thông báo rõ hơn.
- Code tránh bắt `Exception` chung chung ở mọi nơi.

---

### 6.2 `src/preprocess.py`

Mục đích: chuẩn hóa mọi audio về cùng format trước khi trích xuất đặc trưng.

Pipeline chính:

```text
load audio
  -> mono
  -> resample 22050 Hz
  -> trim silence
  -> truncate hoặc zero-pad về 2 giây
  -> normalize RMS về -20 dB
```

Constants quan trọng:

| Constant | Giá trị | Ý nghĩa |
|---|---:|---|
| `SAMPLE_RATE` | 22050 | Sample rate chuẩn |
| `DURATION` | 2.0 | Mỗi clip dài 2 giây |
| `N_SAMPLES` | 44100 | 22050 x 2 |
| `VALID_AUDIO_EXTS` | `.wav`, `.mp3`, `.flac`, `.ogg` | Extension được phép load |

Các hàm chính:

#### `load_excluded_filenames(csv_path)`

Đọc `data/excluded_files.csv` và trả về set tên file bị loại.

Logic:

- Nếu CSV không tồn tại, trả về set rỗng.
- Chỉ lấy dòng có `decision='excluded'`.
- Thêm cả tên raw và tên processed.

Ví dụ:

```text
sounddino_x.wav
cow_sounddino_x.wav
```

Lý do cần cả hai tên: raw file nằm trong folder loài, còn processed file được thêm prefix loài.

#### `load_audio(path, target_sr=22050)`

Đọc file audio bằng `librosa.load()`.

Kiểm tra trước khi đọc:

- File có tồn tại không.
- Extension có thuộc danh sách hợp lệ không.

Khi đọc:

```python
y, sr = librosa.load(path, sr=target_sr, mono=True)
```

Ý nghĩa:

- `sr=target_sr`: tự resample về 22050 Hz.
- `mono=True`: chuyển stereo về mono.

#### `normalize_length(y, n_samples=44100)`

Đưa tín hiệu về đúng 44100 samples.

Ba trường hợp:

- Nếu toàn silence sau trim: trả về vector zeros.
- Nếu dài hơn 2 giây: lấy đoạn giữa.
- Nếu ngắn hơn 2 giây: zero-pad ở cuối.

Vì sao zero-pad thay vì repeat/tile:

- Repeat tạo pattern giả.
- Pattern giả làm sai MFCC/Mel.
- Zero-pad giữ nguyên nội dung thật, phần thiếu được xem là im lặng.

#### `normalize_amplitude(y, target_db=-20.0)`

Chuẩn hóa biên độ theo RMS.

Công thức:

```python
rms = sqrt(mean(y ** 2))
target_rms = 10 ** (target_db / 20.0)
y_out = y * (target_rms / rms)
```

Nếu `rms` quá nhỏ thì giữ nguyên để tránh chia cho 0.

#### `preprocess_audio_for_features(path)`

Đây là hàm quan trọng nhất của preprocessing.

```text
load_audio()
  -> normalize_length()
  -> normalize_amplitude()
  -> return y, sr
```

Hàm này được dùng cho cả:

- Indexing database.
- Query audio trong demo.
- Test feature/search.

Nếu bạn tự viết code query mới, nên gọi hàm này thay vì gọi trực tiếp `librosa.load()`.

#### `preprocess_file(src_path, dst_path)`

Xử lý một file và ghi ra `.wav`.

```text
source audio
  -> preprocess_audio_for_features()
  -> soundfile.write()
  -> destination wav
```

#### `preprocess_all(raw_dir, processed_dir, excluded_csv, verbose)`

Batch process toàn bộ thư mục raw.

Luồng:

1. Tạo `processed_dir` nếu chưa có.
2. Load danh sách excluded.
3. Duyệt `raw_dir` bằng `os.walk`.
4. Chỉ lấy file kết thúc bằng `.wav`.
5. Lấy species từ tên folder.
6. Đặt tên processed bằng `{species_lower}_{filename}`.
7. Skip file excluded.
8. Gọi `preprocess_file`.
9. Trả danh sách record `{filename, species}`.

Điểm cần chú ý: batch preprocess hiện chỉ quét `.wav`, dù `load_audio()` có thể đọc thêm `.mp3`, `.flac`, `.ogg`.

---

### 6.3 `src/feature.py`

Mục đích: biến tín hiệu audio thành vector 310 chiều.

Constants:

| Constant | Giá trị |
|---|---:|
| `SAMPLE_RATE` | 22050 |
| `N_MFCC` | 13 |
| `N_MELS` | 128 |
| `N_CHROMA` | 12 |
| `HOP_LENGTH` | 512 |
| `N_FFT` | 2048 |
| `FEATURE_DIM` | 310 |

Ý nghĩa frame:

- `N_FFT=2048`: mỗi frame FFT nhìn khoảng 92.9 ms ở 22050 Hz.
- `HOP_LENGTH=512`: frame tiếp theo cách frame trước khoảng 23.2 ms.

#### `extract_mfcc(y, sr, n_mfcc=13)`

Tạo MFCC matrix bằng `librosa.feature.mfcc()`.

Nếu matrix có shape `(13, T)`:

- Mean theo thời gian tạo 13 số.
- Std theo thời gian tạo 13 số.
- Tổng output là 26 chiều.

MFCC giúp phân biệt âm sắc, tức "màu âm" của tiếng kêu.

#### `extract_mel_spectrogram(y, sr, n_mels=128)`

Tạo Mel Spectrogram.

Các bước:

1. Tính Mel spectrogram với 128 bands.
2. Chuyển power sang dB bằng `librosa.power_to_db`.
3. Lấy mean và std theo trục thời gian.

Output:

```text
128 mean + 128 std = 256 chiều
```

Mel Spectrogram giúp mô tả phân bố năng lượng theo tần số và thời gian.

#### `extract_chroma(y, sr)`

Tính Chroma STFT.

Output:

```text
12 mean + 12 std = 24 chiều
```

Chroma mô tả cấu trúc cao độ, hữu ích khi âm thanh có tính melodic hoặc harmonic.

#### `extract_spectral_centroid(y, sr)`

Tính trung tâm khối phổ tần.

Output:

```text
[centroid_mean, centroid_std] = 2 chiều
```

Centroid cao thường tương ứng âm thanh sáng, sắc, nhiều năng lượng tần số cao.

#### `extract_zcr(y)`

Tính Zero-Crossing Rate.

Output:

```text
[zcr_mean, zcr_std] = 2 chiều
```

ZCR cao thường liên quan đến tín hiệu nhiều nhiễu hoặc biến thiên nhanh.

#### Các đặc trưng phụ

Các hàm sau có trong code nhưng không nằm trong vector 310D chính:

| Hàm | Ý nghĩa |
|---|---|
| `extract_spectral_bandwidth()` | Độ rộng phổ |
| `extract_spectral_rolloff()` | Tần số rolloff |
| `extract_rms()` | Năng lượng RMS |

Chúng được giữ lại để phân tích hoặc mở rộng sau này.

#### `extract_all(y, sr)`

Hàm tổng hợp quan trọng nhất.

Thứ tự concat:

```python
[
    extract_mfcc(y, sr),               # 26
    extract_mel_spectrogram(y, sr),    # 256
    extract_chroma(y, sr),             # 24
    extract_spectral_centroid(y, sr),  # 2
    extract_zcr(y),                    # 2
]
```

Sau khi concat:

- Assert shape phải là `(310,)`.
- Convert về `np.float32`.

#### `extract_from_file(path, preprocess=True)`

Nếu `preprocess=True`:

```text
preprocess_audio_for_features()
  -> extract_all()
```

Nếu `preprocess=False`:

```text
librosa.load()
  -> extract_all()
```

Trong search thực tế nên dùng `preprocess=True`.

#### `FEATURE_NAMES`

Danh sách 310 tên feature.

Tác dụng:

- Hữu ích nếu muốn tạo DataFrame.
- Hữu ích khi debug từng chiều.
- Assert giúp đảm bảo số tên luôn khớp `FEATURE_DIM`.

---

### 6.4 `src/database.py`

Mục đích: quản lý storage dạng hybrid.

Storage gồm:

| Thành phần | Công nghệ | Lưu gì |
|---|---|---|
| Metadata | PostgreSQL | `file_id`, `filename`, `species`, `filepath`, `duration`, `source` |
| Vector từng file | NumPy `.npy` | Vector 310D |
| Scaler | NumPy `.npz` | `mean`, `std`, `weights` |

#### Config database

`DB_CONFIG` đọc từ `.env`, fallback mặc định:

```text
DB_HOST=localhost
DB_PORT=5432
DB_NAME=animal_sounds
DB_USER=csdldpt
DB_PASSWORD=csdldpt123
```

#### `get_connection()`

Context manager cho PostgreSQL connection.

Mẫu dùng:

```python
with get_connection() as conn:
    with conn.cursor() as cur:
        ...
```

Ưu điểm:

- Tự đóng connection trong `finally`.
- Code gọi gọn hơn.

#### `check_connection()`

Chạy:

```sql
SELECT 1
```

Nếu PostgreSQL chưa chạy hoặc sai credentials, hàm sẽ raise exception.

#### `init_db()`

Tạo 3 bảng nếu chưa có:

- `audio_files`
- `species_stats`
- `search_log`

Và tạo index:

- `idx_audio_species`
- `idx_audio_filename`
- `idx_audio_file_id`

Hàm này là backup cho `scripts/init.sql`.

#### `insert_record(...)`

Ghi một file vào hệ thống.

Hai việc xảy ra cùng lúc:

1. Save vector ra `features/<filename>.npy`.
2. Insert/update metadata vào PostgreSQL.

SQL dùng:

```sql
ON CONFLICT (filename) DO UPDATE
```

Nghĩa là chạy lại pipeline không tạo duplicate, mà update record cũ.

#### `get_all_records()`

Lấy metadata từ bảng `audio_files`, chỉ lấy:

```sql
WHERE quality = 'kept'
```

Thứ tự:

```sql
ORDER BY id
```

Thứ tự này rất quan trọng vì nó quyết định thứ tự vector trong `feature_db.npy` và index trong Faiss.

#### `truncate_all()`

Xóa dữ liệu trong 3 bảng:

```sql
TRUNCATE audio_files, species_stats, search_log
RESTART IDENTITY CASCADE
```

Dùng khi rebuild từ đầu.

#### `save_feature_scaler(matrix, weights)`

Fit mean/std trên toàn bộ database:

```python
mean = matrix.mean(axis=0)
std = matrix.std(axis=0)
std = np.where(std < 1e-8, 1.0, std)
```

Sau đó save vào `.npz`.

Nếu có `weights`, lưu thêm `weights`.

#### `load_feature_scaler()`

Load `mean`, `std`, `weights` từ `feature_scaler.npz`.

Nếu file không tồn tại, trả:

```python
(None, None, None)
```

#### `apply_feature_scaler(matrix, mean, std, weights)`

Áp dụng:

```text
z-score
  -> optional feature weights
```

Hàm này chưa L2-normalize. L2 normalize được làm trong `build_canonical.py` hoặc `search_engine.py`.

#### `load_all_vectors(scaled=False)`

Đọc toàn bộ vector `.npy` theo records trong DB.

Trả về:

```python
ids, filenames, species_list, matrix
```

Nếu `scaled=True`, hàm sẽ apply scaler/weights trước khi trả matrix.

#### `update_species_stats()`

Cập nhật bảng thống kê theo loài:

```sql
SELECT species, COUNT(*), AVG(duration_sec)
FROM audio_files
WHERE quality = 'kept'
GROUP BY species
```

#### `get_db_stats()`

Trả về dict:

```python
{
    "total_files": ...,
    "n_species": ...,
    "species_list": ...
}
```

Demo UI dùng hàm này để hiển thị panel database.

#### `log_search(query_file, top1_result, top1_score)`

Ghi lịch sử tìm kiếm vào bảng `search_log`.

---

### 6.5 `scripts/init.sql`

File SQL này được Docker chạy khi PostgreSQL volume được tạo lần đầu.

Bảng `audio_files`:

| Cột | Ý nghĩa |
|---|---|
| `id` | Primary key |
| `file_id` | ID logic như `cat_0001` |
| `filename` | Tên file |
| `species` | Loài |
| `filepath` | Path tương đối |
| `duration_sec` | Thời lượng |
| `sample_rate` | Sample rate |
| `source` | Nguồn dữ liệu |
| `quality` | `kept` hoặc trạng thái khác |
| `indexed_at` | Thời điểm index |
| `feat_path` | Path tới vector `.npy` |

Bảng `species_stats`:

- `species`
- `file_count`
- `avg_duration_sec`
- `updated_at`

Bảng `search_log`:

- `query_file`
- `top1_result`
- `top1_score`
- `searched_at`

---

### 6.6 `src/index_balanced8.py`

Mục đích: orchestrator build database từ raw audio.

#### `detect_source(filename)`

Suy đoán source dựa trên tên file.

Luật:

| Điều kiện | Source |
|---|---|
| Có `esc50` hoặc bắt đầu bằng `1-`, `2-`, ..., `5-` | `esc50` |
| Có `sounddino` | `sounddino` |
| Có `animalqa` | `animalqa` |
| Có `dynamicsuperb` hoặc `ds_` | `dynamicsuperb` |
| Có `local` | `local` |
| Không khớp | `unknown` |

#### `run(verbose=True)`

Đây là pipeline build chính.

Các bước:

1. In banner.
2. Check PostgreSQL bằng `check_connection()`.
3. Preprocess raw audio bằng `preprocess_all()`.
4. Tạo schema bằng `init_db()`.
5. Xóa dữ liệu cũ bằng `truncate_all()`.
6. Với từng file processed:
   - Gọi `extract_from_file(fpath, preprocess=True)`.
   - Assert vector shape `(310,)`.
   - Đọc duration bằng `soundfile.info()`.
   - Sinh `file_id`.
   - Detect source.
   - Gọi `insert_record()`.
7. Cập nhật `species_stats`.
8. Build `metadata.csv`.
9. In tổng kết.

Điểm cần hiểu: file này chưa build Faiss index. Sau nó phải chạy tiếp `build_canonical.py`.

---

### 6.7 `src/build_metadata.py`

Mục đích: tạo `data/metadata.csv`.

#### `load_source_map()`

Đọc `data/balanced8_metadata.csv` nếu có, tạo map:

```python
{filename: source}
```

Nếu file không tồn tại hoặc đọc lỗi, trả map rỗng.

#### `detect_source_from_filename(filename)`

Fallback detect source khi metadata gốc không có.

Logic gần giống `index_balanced8.detect_source()`.

#### `build_metadata(processed_dir, output_path, excluded_csv, verbose)`

Luồng:

1. Load excluded filenames.
2. Load source map.
3. Duyệt các file `.wav` trong processed dir.
4. Skip file excluded.
5. Lấy species từ prefix trước dấu `_` đầu tiên.
6. Sinh `file_id` theo thứ tự từng loài.
7. Đọc duration và sample rate bằng `soundfile`.
8. Lấy source từ source map hoặc fallback detect.
9. Ghi CSV 7 cột.

CSV output:

```text
file_id, filename, species, filepath, duration_sec, sample_rate, source
```

---

### 6.8 `src/build_canonical.py`

Mục đích: tạo artifact chuẩn dùng cho search.

Output:

| File | Vai trò |
|---|---|
| `feature_db.npy` | Ma trận vector thô `(N, 310)` |
| `feature_scaler.npz` | Mean, std, weights |
| `faiss.index` | Index vector đã transform |
| `file_index.json` | Metadata map cho từng vector index |

#### `FEATURE_WEIGHTS`

Code tạo weights:

```python
FEATURE_WEIGHTS = np.concatenate([
    np.full(N_MFCC * 2, 3.0),
    np.full(N_MELS * 2, 1.0),
    np.full(N_CHROMA * 2, 2.0),
    np.full(2, 2.0),
    np.full(2, 2.0),
]).astype(np.float32)
```

Tổng chiều phải bằng 310.

#### `load_metadata_map(csv_path)`

Đọc `metadata.csv` thành dict:

```python
{
    filename: {
        file_id,
        filepath,
        species,
        duration_sec,
        sample_rate,
        source
    }
}
```

#### `build_canonical(verbose=True)`

Các bước:

1. Load vector từ database bằng `load_all_vectors(scaled=False)`.
2. Assert dimension là 310.
3. Save matrix thô thành `feature_db.npy`.
4. Fit z-score scaler và lưu weights.
5. Transform matrix:

```python
safe_std = np.where(std < 1e-8, 1.0, std)
scaled = ((matrix - mean) / safe_std).astype(np.float32)
scaled *= FEATURE_WEIGHTS
faiss.normalize_L2(scaled)
```

6. Tạo `faiss.IndexFlatIP(310)`.
7. Add vector đã normalize vào index.
8. Ghi `faiss.index`.
9. Tạo `file_index.json`.

Vì sao `IndexFlatIP` tương đương cosine similarity:

- Inner Product giữa hai vector đã L2-normalize bằng cosine similarity.
- Do đó Faiss score chính là cosine score.

---

### 6.9 `src/search_engine.py`

Mục đích: core search online.

#### `AnimalSoundSearchEngine.__init__()`

Khởi tạo:

- `dimension = 310`
- `index = None`
- `file_index = {}`
- `scaler_mean = None`
- `scaler_std = None`
- `feature_weights = None`
- `_loaded = False`

#### `load(index_path, file_index_path, scaler_path)`

Load 3 artifact:

1. `faiss.index`
2. `file_index.json`
3. `feature_scaler.npz`

Khi load JSON, key từ string được convert về int:

```python
self.file_index = {int(k): v for k, v in raw.items()}
```

#### `is_loaded()`

Trả True nếu engine đã load và `index` không None.

#### `_prepare_query(query_vector)`

Transform query giống hệt lúc build index:

```text
raw 310D
  -> z-score
  -> feature weights
  -> L2 normalize
```

Nếu transform query khác transform database, search sẽ sai.

#### `search(query_vector, top_k=5)`

Các bước:

1. Assert engine đã load.
2. Assert query shape là `(310,)`.
3. Chuẩn bị query bằng `_prepare_query()`.
4. Gọi:

```python
scores, indices = self.index.search(q, top_k)
```

5. Với mỗi index trả về:
   - Lookup metadata trong `file_index`.
   - Clip score về `[0, 1]`.
   - Round similarity 4 chữ số.
   - Tính `distance = 1 - similarity`.

Output schema:

```python
{
    "rank": 1,
    "filepath": "data/processed/...",
    "species": "cat",
    "similarity_score": 0.9876,
    "distance": 0.0124,
}
```

#### `create_engine(features_dir)`

Factory function:

```python
engine = AnimalSoundSearchEngine()
engine.load(...)
return engine
```

Demo UI dùng hàm này khi app start.

---

### 6.10 `src/visualization.py`

Mục đích: tạo ảnh phục vụ demo và báo cáo.

Điểm quan trọng:

```python
matplotlib.use("Agg")
```

`Agg` giúp vẽ hình trong môi trường server/headless, không cần cửa sổ GUI.

Các hàm:

| Hàm | Output |
|---|---|
| `save_waveform()` | Ảnh waveform |
| `save_mel_spectrogram()` | Ảnh Mel spectrogram |
| `save_mfcc_heatmap()` | Ảnh MFCC heatmap |
| `save_comparison()` | So sánh waveform query và results |
| `save_similarity_bar()` | Bar chart điểm similarity Top-5 |

Mẫu chung của các hàm:

1. Tạo folder output nếu cần.
2. Tạo figure bằng matplotlib.
3. Vẽ bằng `librosa.display`.
4. Save PNG.
5. `plt.close(fig)` để tránh leak bộ nhớ.

---

### 6.11 `src/generate_search_examples.py`

Mục đích: tạo kết quả trung gian cho báo cáo.

Output chính:

```text
features/intermediate/
  ├── feature_examples/
  ├── search_scenario_1_in_db/
  ├── search_scenario_2_external/
  └── query_external_dog.wav
```

#### `generate_feature_examples(output_dir)`

Chọn 4 loài:

```python
["cat", "dog", "frog", "cow"]
```

Với mỗi loài:

- Chọn file đầu tiên.
- Preprocess.
- Save waveform.
- Save spectrogram.
- Save MFCC heatmap.

#### `generate_scenario(scenario_name, query_path, output_dir, engine)`

Tạo đầy đủ artifact cho một kịch bản search:

1. Preprocess query.
2. Extract vector 310D.
3. Search Top-5.
4. Save `query_vector.npy`.
5. Save query waveform và spectrogram.
6. Save `query_info.json`.
7. Save `ranking.json`.
8. Tính full ranking với toàn bộ 1042 file.
9. Save `ranking_full.csv`.
10. Save spectrogram từng result.
11. Save comparison chart.
12. Save similarity bar.

#### `main()`

Tạo 2 kịch bản:

- Scenario 1: query là file có trong CSDL, kỳ vọng rank 1 chính nó.
- Scenario 2: query là dog file có thêm Gaussian noise, mô phỏng file ngoài CSDL.

---

### 6.12 `src/evaluate_feature_space.py`

Mục đích: đánh giá chất lượng không gian đặc trưng.

Quan trọng: đây không phải classifier. Label `species` chỉ dùng để đo xem các file cùng loài có gần nhau hơn không.

Các khái niệm:

| Hàm | Vai trò |
|---|---|
| `get_feature_slices()` | Xác định slice của MFCC, Mel, Chroma, Centroid, ZCR |
| `load_artifacts()` | Load `feature_db.npy`, scaler, weights, file index |
| `normalize_rows()` | L2-normalize từng row |
| `transform_features()` | z-score, weight, L2-normalize |
| `precision_at_k()` | Tính Top-1 cùng loài và Precision@K |
| `overlap_pairs()` | Tìm cặp loài khác nhau có cosine cao |
| `evaluate_variant()` | Đánh giá một bộ weights |
| `build_weight_variants()` | Tạo các cấu hình ablation |
| `print_report()` | In report Markdown |

Các variant ablation:

- `baseline_current`
- `no_chroma`
- `low_chroma_0_5`
- `low_chroma_1_0`
- `mel_downweight_0_75`
- `mel_downweight_0_5`
- `mfcc_focus`

Kết quả hiện có trong `docs/feature_space_evaluation.md`:

| Metric | Baseline |
|---|---:|
| Top-1 cùng loài | 90.31% |
| Precision@5 | 73.01% |

---

### 6.13 `app/demo.py`

Mục đích: Gradio UI cho người dùng upload hoặc thu âm và xem Top-5.

Điểm quan trọng:

```python
engine = create_engine(FEATURES_DIR)
```

Engine được load ngay khi app start. Nếu thiếu `faiss.index`, `file_index.json` hoặc `feature_scaler.npz`, app có thể lỗi trước khi bấm Search.

#### `do_search(audio_path)`

Callback chính của nút Search.

Luồng:

```text
audio_path
  -> preprocess_audio_for_features()
  -> extract_all()
  -> engine.search(top_k=5)
  -> log_search()
  -> vẽ query waveform/spectrogram
  -> build result table HTML
  -> vẽ spectrogram từng result
  -> vẽ similarity chart
  -> return toàn bộ output cho Gradio
```

Điểm cần chú ý:

- Branch `audio_path is None` hiện trả số output ít hơn list outputs của UI. Nếu muốn UI bền hơn, nên sửa branch này để trả đủ 14 output.
- `_plot_comparison()` hiện vẽ bar chart similarity, chưa vẽ mini waveform dù comment có nhắc tới mini waveforms.

#### `_plot_spectrogram(y, sr, title)`

Tạo Mel spectrogram và save vào file tạm `.png`.

#### `_plot_waveform(y, sr, title)`

Tạo waveform và save vào file tạm `.png`.

#### `_plot_comparison(y_query, results, sr)`

Tạo horizontal bar chart cho Top-5 similarity.

#### `_build_results_html(results)`

Tạo bảng HTML gồm:

- Rank
- File
- Species
- Similarity bar
- Similarity score

#### `get_stats_html()`

Hiển thị:

- Tổng số file.
- Số loài.
- Feature dimension.
- Chip từng loài và số file.

#### UI layout

Gồm:

- Cột trái: input audio, button Search, database info.
- Cột phải: bảng kết quả, waveform query, spectrogram query.
- Bên dưới: 5 audio player và spectrogram result.
- Cuối: similarity chart.

---

### 6.14 `CSDLDPT/clean_md.py`

File này không thuộc pipeline runtime. Nó là script tiện ích để làm sạch Markdown báo cáo.

Nó làm các việc:

- Đọc `../docs/CSDLDPT-Nhom3.md`.
- Xóa form feed.
- Xóa page number đứng riêng một dòng.
- Chuyển một số heading thành Markdown heading.
- Ghi đè lại file báo cáo.

Cẩn thận: vì script ghi đè report chính, không nên chạy nếu chưa chắc chắn.

---

## 7. Config Và Dependency

### 7.1 `docker-compose.yml`

Service chính:

```yaml
db:
  image: postgres:16-alpine
  container_name: csdldpt_postgres
```

Các phần quan trọng:

- Environment lấy từ `.env`.
- Port mặc định `5432`.
- Volume `pgdata` giữ dữ liệu PostgreSQL.
- Mount `scripts/init.sql` vào `/docker-entrypoint-initdb.d/init.sql`.
- Healthcheck dùng `pg_isready`.

### 7.2 `.env.example`

```text
DB_HOST=localhost
DB_PORT=5432
DB_NAME=animal_sounds
DB_USER=csdldpt
DB_PASSWORD=csdldpt123
```

### 7.3 `.gitignore`

Các nhóm bị ignore:

- Raw/processed audio.
- Feature binary.
- Faiss index.
- Intermediate output.
- `.env`.
- `.venv`.
- Cache Python/pytest.

Lý do: những file này là dữ liệu lớn, file sinh tự động hoặc thông tin local.

### 7.4 `requirements.txt`

Thư viện chính:

| Thư viện | Vai trò |
|---|---|
| `librosa` | Load audio, feature extraction |
| `soundfile` | Đọc/ghi WAV, lấy metadata audio |
| `numpy` | Vector và ma trận |
| `pandas` | CSV metadata |
| `matplotlib` | Visualization |
| `gradio` | Demo UI |
| `faiss-cpu` | Vector search |
| `psycopg2-binary` | PostgreSQL driver |
| `python-dotenv` | Load `.env` |
| `pytest` | Unit tests |

---

## 8. Tests Đang Kiểm Tra Gì?

### 8.1 `tests/test_preprocessing.py`

Kiểm tra:

- File không tồn tại raise `AudioFileNotFoundError`.
- File extension sai raise `AudioFormatError`.
- `normalize_length()` luôn trả 44100 samples.
- File ngắn được zero-pad.
- File dài được truncate.
- Audio toàn silence trả zeros.
- RMS sau normalize khớp target -20 dB.
- Excluded CSV load được.
- Missing excluded CSV trả set rỗng.

### 8.2 `tests/test_feature_extraction.py`

Kiểm tra:

- `extract_all()` trả shape `(310,)`.
- `FEATURE_NAMES` có đúng 310 tên.
- Từng nhóm feature có đúng số chiều:
  - MFCC 26D.
  - Mel 256D.
  - Chroma 24D.
  - Centroid 2D.
  - ZCR 2D.
- Vector không có NaN.
- Vector không có Inf.
- Vector dtype là `float32`.

### 8.3 `tests/test_search_engine.py`

Kiểm tra:

- Query chính vector trong DB thì rank 1 là chính nó.
- Self-match score >= 0.999.
- Feature weights được load và không bị rơi về all-ones.
- `_prepare_query()` khớp manual transform.
- Search trả đúng 5 kết quả.
- Schema có đủ key bắt buộc.
- Similarity nằm trong `[0, 1]`.
- Distance bằng `1 - similarity`.
- Kết quả sort giảm dần.
- Rank là `[1, 2, 3, 4, 5]`.
- Query ngoài CSDL vẫn trả 5 kết quả.
- Vector sai dimension raise AssertionError.
- Engine chưa load thì không được search.

Kết quả gần nhất:

```text
42 passed, 5 warnings
```

Warnings đến từ dependency/Python 3.13/Faiss SWIG metadata, không làm test fail.

---

## 9. Thứ Tự Chạy Từ Đầu

Chạy từ thư mục `CSDLDPT/`:

```bash
cp .env.example .env
docker compose up -d
python src/index_balanced8.py
python src/build_canonical.py
python src/evaluate_feature_space.py
python src/generate_search_examples.py
python -m pytest tests/ -v
python app/demo.py
```

Vì sao thứ tự này quan trọng:

| Bước | Vì sao cần |
|---|---|
| `docker compose up -d` | Tạo PostgreSQL và schema |
| `index_balanced8.py` | Cần DB để insert metadata |
| `build_canonical.py` | Cần DB và vector `.npy` để build Faiss |
| `evaluate_feature_space.py` | Cần canonical artifacts |
| `generate_search_examples.py` | Cần engine load được Faiss index |
| `app/demo.py` | Load engine ngay khi start |

Reset toàn bộ DB:

```bash
docker compose down -v
docker compose up -d
python src/index_balanced8.py
python src/build_canonical.py
```

---

## 10. Các Điểm Dễ Nhầm

1. `data/processed` là symlink, không phải folder độc lập.
2. `preprocess_all()` chỉ quét `.wav`, còn `load_audio()` có thể đọc `.mp3`, `.flac`, `.ogg`.
3. `feature_db.npy` là vector thô, không phải vector đã sẵn sàng search.
4. `faiss.index` chứa vector đã z-score, nhân weights và L2-normalize.
5. `feature_scaler.npz` phải có weights. Nếu thiếu weights, test sẽ báo lỗi.
6. `file_index.json` lưu key dạng string trong JSON, nhưng search engine convert về int.
7. `distance` trong output là `1 - similarity_score`, không phải raw distance từ Faiss.
8. Query external noisy trong report có thể không trả đúng loài. Đây là giới hạn feature thủ công với noise, không phải pipeline crash.
9. `app/demo.py` load engine ngay lúc start, nên phải build canonical trước khi chạy demo.
10. `clean_md.py` ghi đè báo cáo chính, không chạy bừa.

---

## 11. Cách Học Project Này

Thứ tự đọc khuyến nghị:

1. Đọc `context/problem.md` để hiểu đề bài.
2. Đọc `src/preprocess.py`, tập trung vào `preprocess_audio_for_features()`.
3. Đọc `src/feature.py`, tự vẽ lại layout vector 310D.
4. Đọc `src/build_canonical.py`, hiểu z-score, weights và L2 normalize.
5. Đọc `src/search_engine.py`, hiểu cách Top-5 được lấy ra.
6. Đọc `app/demo.py`, hiểu UI gọi pipeline như thế nào.
7. Đọc `tests/`, xem hệ thống đang cam kết hành vi gì.
8. Đọc `database.py` sau cùng, vì file này dài nhưng chủ yếu là storage.

Bài tập nhỏ để tự kiểm tra:

```python
from feature import extract_from_file
from search_engine import create_engine

path = "data/processed/cat_local_B_ANI01_MC_FN_SIM01_101.wav"
vec = extract_from_file(path, preprocess=True)
print(vec.shape)
print(vec[:5])

engine = create_engine("features")
results = engine.search(vec, top_k=5)
print(results[0])
```

Kỳ vọng:

- `vec.shape` là `(310,)`.
- Rank 1 là chính file query nếu file đó có trong DB.

---

## 12. Tóm Tắt Từng File Một Dòng

| File | Cần nhớ |
|---|---|
| `src/exceptions.py` | Định nghĩa 3 loại lỗi audio riêng |
| `src/preprocess.py` | Chuẩn hóa audio về mono, 22050 Hz, 2 giây, RMS -20 dB |
| `src/feature.py` | Tạo vector 310D từ MFCC, Mel, Chroma, Centroid, ZCR |
| `src/database.py` | Lưu metadata vào PostgreSQL và vector vào `.npy` |
| `src/index_balanced8.py` | Build DB từ raw audio |
| `src/build_metadata.py` | Tạo `data/metadata.csv` |
| `src/build_canonical.py` | Tạo `feature_db.npy`, scaler, weights, Faiss index, file index |
| `src/search_engine.py` | Search Top-5 bằng cosine/Faiss |
| `src/visualization.py` | Tạo waveform, spectrogram, MFCC, chart |
| `src/generate_search_examples.py` | Tạo kết quả trung gian cho báo cáo |
| `src/evaluate_feature_space.py` | Đánh giá retrieval space và ablation weights |
| `app/demo.py` | Gradio UI cho upload/record và Top-5 |
| `scripts/init.sql` | Schema PostgreSQL |
| `tests/*` | Kiểm tra preprocessing, feature, search |
| `clean_md.py` | Tiện ích làm sạch Markdown, không thuộc runtime |

---

## 13. Checklist Khi Sửa Code

### Nếu sửa preprocessing

- Chạy `tests/test_preprocessing.py`.
- Rebuild vector vì vector cũ được tạo từ preprocessing cũ.
- Chạy lại `index_balanced8.py`.
- Chạy lại `build_canonical.py`.

### Nếu sửa feature layout

- Cập nhật `FEATURE_DIM`.
- Cập nhật `FEATURE_NAMES`.
- Cập nhật `FEATURE_WEIGHTS`.
- Cập nhật `evaluate_feature_space.get_feature_slices()`.
- Cập nhật tài liệu vector.
- Rebuild toàn bộ artifacts.

### Nếu sửa search transform

- Sửa đồng bộ `build_canonical.py`.
- Sửa đồng bộ `search_engine.py`.
- Sửa `generate_search_examples.py` nếu còn tính ranking full thủ công.
- Chạy `tests/test_search_engine.py`.

### Nếu sửa database schema

- Cập nhật `scripts/init.sql`.
- Cập nhật `database.init_db()`.
- Reset Docker volume nếu cần schema mới.

### Nếu sửa UI

- Kiểm tra số output của `do_search()` khớp list `outputs`.
- Chạy demo.
- Thử một file có trong DB.
- Thử một file ngoài DB.

---

## 14. Kết Luận Nhanh

Nếu chỉ nhớ một luồng, hãy nhớ luồng này:

```text
Raw audio
  -> preprocess.py
  -> feature.py
  -> database.py
  -> build_canonical.py
  -> search_engine.py
  -> app/demo.py
```

Nếu chỉ nhớ một công thức, hãy nhớ công thức search:

```text
raw vector 310D
  -> z-score
  -> feature weights
  -> L2 normalize
  -> Faiss IndexFlatIP
  -> cosine similarity Top-5
```

Nếu hệ thống trả kết quả lạ, hãy kiểm tra theo thứ tự:

1. Query có đi qua `preprocess_audio_for_features()` không?
2. Vector có đúng shape `(310,)` không?
3. `feature_scaler.npz` có đúng `mean`, `std`, `weights` không?
4. `build_canonical.py` và `search_engine.py` có dùng cùng transform không?
5. `file_index.json` có khớp với `feature_db.npy` không?
6. Demo có đang load đúng thư mục `features/` không?

