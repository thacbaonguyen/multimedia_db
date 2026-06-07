# CSDLDPT - Walkthrough Chi Tiet Toan Bo Du An

Tai lieu nay duoc viet cho nguoi moi hoc du an. Muc tieu la giup ban doc code theo dung luong chay that, hieu file nao lam gi, ham nao nhan dau vao nao, tao dau ra nao, va artifact nao phu thuoc vao artifact nao.

Ban scan nay dua tren project hien tai tai `/home/thacbao/Documents/project/int1418`, ngay 2026-06-07. Da doi chieu source, tests, metadata va artifact hien co:

- `data/metadata.csv`: 1042 dong, 7 cot.
- 8 loai: `cat`, `cow`, `dog`, `frog`, `hen`, `monkey`, `rooster`, `sheep`.
- `features/feature_db.npy`: shape `(1042, 310)`.
- `features/feature_scaler.npz`: co `mean`, `std`, `weights`.
- `features/file_index.json`: 1042 entries.
- Test hien tai: `42 passed, 5 warnings`.

Ghi chu ve duong dan: trong repo co ca `CSDLDPT/data/balanced8_processed/` va symlink `CSDLDPT/data/processed -> balanced8_processed`. Code build thuong doc `balanced8_processed`, con metadata/file_index luu path tuong doi dang `data/processed/<filename>`.

---

## 1. Ban Do Tong Quan

Du an giai bai toan Content-Based Audio Retrieval cho tieng dong vat:

1. Nhan audio goc cua nhieu loai dong vat.
2. Chuan hoa moi audio ve cung format: mono, 22050 Hz, 2 giay, RMS -20 dB.
3. Trich xuat moi file thanh vector 310 chieu.
4. Luu metadata vao PostgreSQL, vector rieng vao `.npy`.
5. Build ma tran vector, scaler, feature weights va Faiss index.
6. Khi user upload query, xu ly query bang dung pipeline do roi tim Top-5 bang cosine similarity.

Ba y quan trong nhat:

- He thong khong phai classifier. No khong "du doan loai" truc tiep; no tra ve 5 file giong nhat. Cot `species` dung de hien thi va danh gia.
- `feature_db.npy` la vector raw, chua z-score/chua weight/chua L2 normalize. `faiss.index` moi la ban da transform de search.
- Query bat buoc di qua cung preprocessing va feature transform voi database, neu khong similarity se vo nghia.

---

## 2. Cau Truc Thu Muc

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

Vai tro tung nhom:

- `context/`: de bai va quy tac bat buoc, khong phai code runtime.
- `CSDLDPT/src/`: code xu ly du lieu, feature, DB, index, search, evaluate.
- `CSDLDPT/app/`: demo UI bang Gradio.
- `CSDLDPT/data/`: CSV metadata va audio raw/processed.
- `CSDLDPT/features/`: vector `.npy`, scaler, Faiss index, output trung gian.
- `CSDLDPT/tests/`: pytest kiem tra hanh vi cot loi.
- `docs/`: bao cao nop mon hoc.
- `CSDLDPT/docs/`: tai lieu ky thuat cua project.

---

## 3. Luong Chay Lon

### 3.1 Luong Offline: Build Database Va Index

Lenh chay tu thu muc `CSDLDPT/`:

```bash
python src/index_balanced8.py
python src/build_canonical.py
```

`index_balanced8.py` lam cac viec sau:

1. Kiem tra PostgreSQL.
2. Goi `preprocess_all()` de doc `data/balanced8_raw/`, tao audio chuan trong `data/balanced8_processed/`.
3. Reset database PostgreSQL.
4. Voi moi file processed:
   - trich xuat vector 310D bang `feature.extract_from_file()`;
   - tinh duration bang `soundfile`;
   - sinh `file_id`;
   - suy doan source;
   - insert metadata vao PostgreSQL;
   - save vector thanh `features/<filename>.npy`.
5. Cap nhat bang `species_stats`.
6. Tao `data/metadata.csv`.

`build_canonical.py` lam cac viec sau:

1. Doc tat ca `.npy` vector theo record trong PostgreSQL.
2. Luu ma tran raw thanh `features/feature_db.npy`.
3. Fit z-score scaler tren toan bo database.
4. Tao vector weights: MFCC x3, Mel x1, Chroma x2, Centroid x2, ZCR x2.
5. Transform ma tran: z-score -> weights -> L2 normalize.
6. Add vao `faiss.IndexFlatIP`.
7. Ghi `features/faiss.index`.
8. Tao `features/file_index.json` de map row index cua Faiss ve metadata.

### 3.2 Luong Online: Search Khi User Upload Audio

Lenh chay demo:

```bash
python app/demo.py
```

Khi bam Search trong UI:

```text
audio upload
  -> preprocess_audio_for_features()
  -> extract_all()
  -> engine.search(query_vec, top_k=5)
  -> _build_results_html()
  -> plot waveform/spectrogram/result chart
  -> tra ket qua ve Gradio outputs
```

Ben trong `engine.search()`:

```text
query vector raw 310D
  -> z-score bang mean/std trong feature_scaler.npz
  -> nhan feature_weights
  -> L2 normalize
  -> faiss.index.search(q, 5)
  -> tra indices va scores
  -> lookup file_index.json
  -> output list dict: rank, filepath, species, similarity_score, distance
```

---

## 4. Dataset Va Artifact Hien Co

Thong ke doc tu project hien tai:

| Loai | So file |
|---|---:|
| cat | 159 |
| cow | 123 |
| dog | 160 |
| frog | 120 |
| hen | 100 |
| monkey | 150 |
| rooster | 100 |
| sheep | 130 |
| Tong | 1042 |

Thong ke theo source:

| Source | So file |
|---|---:|
| local | 442 |
| esc50 | 269 |
| unknown | 125 |
| dynamicsuperb | 90 |
| animalqa | 80 |
| sounddino | 36 |

Tat ca 1042 file trong `metadata.csv` co:

- `duration_sec = 2.0`.
- `sample_rate = 22050`.
- Vector raw co 310 chieu.

Luu y nho nhung quan trong:

- `processed_wav_count` trong thu muc processed hien la 1051, lon hon 1042. Ly do co 9 file excluded van co the con nam tren disk sau lan preprocess truoc, nhung metadata/index chi lay 1042 file kept.
- `feature_npy_count` la 1042, khop voi metadata/index.
- `features/intermediate/` co 39 file trung gian cho report/demo.

---

## 5. Vector 310D

File dinh nghia: `CSDLDPT/src/feature.py`.

Layout chinh xac:

| Slice | Nhom | So chieu | Y nghia |
|---|---|---:|---|
| `0:26` | MFCC | 26 | 13 mean + 13 std |
| `26:282` | Mel Spectrogram | 256 | 128 mean + 128 std |
| `282:306` | Chroma | 24 | 12 mean + 12 std |
| `306:308` | Spectral Centroid | 2 | mean + std |
| `308:310` | ZCR | 2 | mean + std |

Cong thuc transform truoc khi search:

```python
safe_std = np.where(std < 1e-8, 1.0, std)
scaled = (raw_vector - mean) / safe_std
weighted = scaled * weights
normalized = weighted / ||weighted||
```

Weights hien tai:

| Nhom | Weight |
|---|---:|
| MFCC | 3.0 |
| Mel Spectrogram | 1.0 |
| Chroma | 2.0 |
| Spectral Centroid | 2.0 |
| ZCR | 2.0 |

Ly do co weights: Mel chiem 256/310 chieu, neu khong weight thi cosine space bi Mel dominate. MFCC quan trong cho am sac nhung chi co 26 chieu, nen duoc nhan 3.

---

## 6. File-by-file Walkthrough

Phan nay doc theo source file, ghi line range de ban mo code va doi chieu truc tiep.

### 6.1 `src/exceptions.py`

File ngan nhung quan trong vi no lam error handling ro rang hon.

- Lines 1-8: docstring noi file tao custom exception cho 3 nhom loi: thieu file, sai dinh dang, loi xu ly audio.
- Lines 11-12: `AudioFileNotFoundError` ke thua `FileNotFoundError`. Khi `load_audio()` gap path khong ton tai, no raise class nay de test bat dung loi.
- Lines 15-16: `AudioFormatError` ke thua `ValueError`. Dung khi extension khong nam trong `.wav/.mp3/.flac/.ogg`.
- Lines 19-20: `AudioProcessingError` ke thua `RuntimeError`. Dung khi `librosa.load()` fail vi file hong hoac noi dung khong doc duoc.

Cho nguoi moi: custom exception giup caller biet loi thuoc loai nao, thay vi bat mot `Exception` chung chung.

### 6.2 `src/preprocess.py`

Muc dich: bien moi audio thanh cung dang tin hieu de feature extraction so sanh cong bang.

#### Dau file va constants

- Lines 1-6: docstring tom tat pipeline: load -> trim silence -> truncate/zero-pad -> normalize amplitude -> save.
- Line 8: `from __future__ import annotations` giup type hints duoc xu ly linh hoat hon.
- Lines 10-13: import thu vien standard: `csv`, `logging`, `os`, `Optional`.
- Lines 15-17: import thu vien audio/numeric: `librosa`, `numpy`, `soundfile`.
- Line 19: import 3 custom exceptions.
- Line 21: tao logger theo ten module.
- Lines 23-28: constants:
  - `SAMPLE_RATE = 22050`: moi file se resample ve 22050 Hz.
  - `DURATION = 2.0`: moi clip dung 2 giay.
  - `N_SAMPLES = 44100`: so sample cua 2 giay tai 22050 Hz.
  - `VALID_AUDIO_EXTS`: query/load cho phep `.wav`, `.mp3`, `.flac`, `.ogg`.
- Lines 30-32: tao path mac dinh tu vi tri file hien tai:
  - raw la `data/balanced8_raw`;
  - processed la `data/balanced8_processed`;
  - excluded CSV la `data/excluded_files.csv`.

#### `load_excluded_filenames()` - lines 35-54

Ham nay doc danh sach file bi loai.

- Line 35: tham so mac dinh la `EXCLUDED_CSV`.
- Lines 36-40: docstring noi ham tra ve set filename co `decision='excluded'`.
- Line 41: khoi tao set rong.
- Lines 42-43: neu CSV khong ton tai thi return set rong, giup pipeline khong crash khi chua co audit file.
- Lines 44-45: mo CSV bang UTF-8 va doc bang `csv.DictReader`, tuc moi row la dict theo ten cot.
- Line 47: chi xu ly row co `decision` sau khi strip/lower bang `excluded`.
- Lines 48-49: lay raw filename va them vao set.
- Lines 51-53: lay `species`, lower, roi them ca dang processed: `{species}_{raw_name}`. Ly do: raw file ten `sounddino_x.wav`, nhung sau preprocess file co the thanh `cow_sounddino_x.wav`.
- Line 54: tra ve set de cac ham khac check membership nhanh O(1).

#### `load_audio()` - lines 57-71

Ham nay doc 1 file audio va chuan hoa ve mono + sample rate.

- Line 57: signature tra ve tuple `(np.ndarray, int)`, tuc `(y, sr)`.
- Lines 58-60: docstring noi ham co the raise 3 custom exceptions.
- Lines 62-63: neu file khong ton tai, raise `AudioFileNotFoundError`.
- Lines 64-66: tach extension, lower, neu khong thuoc set hop le thi raise `AudioFormatError`.
- Lines 67-70: goi `librosa.load(path, sr=target_sr, mono=True)`. `sr=target_sr` dong thoi resample; `mono=True` tron stereo ve mono.
- Lines 69-70: moi exception tu librosa duoc boc thanh `AudioProcessingError`, giu traceback goc bang `from e`.
- Line 71: tra ve tin hieu `y` va sample rate `sr`.

#### `normalize_length()` - lines 74-94

Ham nay dam bao moi tin hieu dung 44100 sample.

- Line 74: nhan `y` va so sample muc tieu.
- Lines 75-80: docstring mo ta 3 truong hop: dai thi cat giua, ngan thi zero-pad cuoi, silence thi zeros.
- Line 82: `librosa.effects.trim(y, top_db=20)` cat im lang dau/cuoi. `top_db=20` nghia la phan qua nho so voi peak se bi xem la silence.
- Lines 84-86: neu sau trim khong con sample nao, log warning va tra ve vector zeros dung length.
- Lines 88-91: neu tin hieu dai hon muc tieu, lay doan giua. `start = (len - n_samples)//2` giup khong thien ve dau file.
- Lines 93-94: neu ngan hon muc tieu, zero-pad cuoi bang `np.pad`. Code co ghi ro KHONG tile/repeat de tranh tao mau lap gia.

#### `normalize_amplitude()` - lines 97-103

Ham nay chuan hoa loudness theo RMS.

- Line 97: tham so `target_db=-20.0`.
- Line 99: tinh RMS bang can bac hai cua trung binh binh phuong sample.
- Lines 100-101: neu RMS gan 0, tra lai `y` de tranh chia cho 0.
- Line 102: doi dB sang RMS tuyen tinh: `10 ** (target_db / 20)`. Voi -20 dB, target RMS la 0.1.
- Line 103: nhan tin hieu voi ty le `target_rms / rms`.

#### `preprocess_audio_for_features()` - lines 106-116

Day la ham quan trong nhat cua preprocessing.

- Lines 106-112: docstring noi ham dung cho ca indexing va query.
- Line 113: doc file bang `load_audio`.
- Line 114: chuan hoa length.
- Line 115: chuan hoa amplitude.
- Line 116: tra ve `(y, sr)`.

Neu ban can xu ly mot query audio, dung ham nay, dung dung truc tiep `librosa.load()` vi se lech pipeline voi DB.

#### `preprocess_file()` - lines 119-123

- Line 119: nhan source path va destination path.
- Line 121: goi pipeline day du.
- Line 122: ghi file `.wav` bang `soundfile.write`.
- Line 123: tra ve `dst_path` de caller biet file output.

#### `preprocess_all()` - lines 126-181

Batch process toan bo thu muc raw.

- Lines 126-131: signature cho phep custom raw dir, processed dir, excluded csv, verbose.
- Line 133: tao processed dir neu chua co.
- Lines 135-137: load excluded set, in so luong neu co.
- Line 139: tao list `files_to_process`, moi phan tu gom source path, destination path, destination filename, species.
- Lines 140-143: `os.walk(raw_dir)` va chi lay file ket thuc bang `.wav`. Luu y: batch preprocess hien chi xu ly `.wav`, mac du `load_audio()` ho tro them `.mp3/.flac/.ogg`.
- Lines 145-148: skip neu raw filename nam trong excluded.
- Line 150: `species = os.path.basename(root)`, lay ten folder loai. Raw folders hien la `Cat`, `Cow`, ...
- Line 152: filename processed co prefix loai lower-case, vi du `Cat/x.wav` thanh `cat_x.wav`.
- Lines 155-158: skip neu processed filename nam trong excluded.
- Lines 160-162: tao path source/destination va append vao list.
- Line 164: sort list de ket qua on dinh giua cac lan chay.
- Lines 166-170: lap qua tung file, preprocess va append result dict gom `filename`, `species`.
- Lines 171-175: neu gap loi audio da dinh nghia thi log/print nhung tiep tuc file khac.
- Lines 176-177: moi 100 file thi in progress.
- Lines 179-181: in tong ket va tra ve list records.
- Lines 184-188: neu chay file truc tiep, goi `preprocess_all()` va in so loai/tong file.

### 6.3 `src/feature.py`

Muc dich: bien tin hieu audio da preprocess thanh vector 310D.

#### Dau file va constants

- Lines 1-13: docstring ghi ro vector 310D gom MFCC 26, Mel 256, Chroma 24, Centroid 2, ZCR 2.
- Lines 15-21: import `numpy`, `librosa`, pipeline preprocess va exceptions.
- Lines 23-29: constants:
  - `SAMPLE_RATE = 22050`.
  - `N_MFCC = 13`.
  - `N_MELS = 128`.
  - `N_CHROMA = 12`.
  - `HOP_LENGTH = 512`.
  - `N_FFT = 2048`.
  - `FEATURE_DIM = 310`.

`N_FFT=2048` nghia moi frame FFT nhin khoang 2048 sample, gan 92.9 ms o 22050 Hz. `HOP_LENGTH=512` nghia frame tiep theo bat dau sau 512 sample, gan 23.2 ms.

#### `extract_mfcc()` - lines 32-44

- Line 32: nhan `y`, `sr`, `n_mfcc`.
- Lines 33-39: docstring giai thich MFCC la dac trung am sac.
- Lines 40-43: goi `librosa.feature.mfcc` voi `n_fft` va `hop_length` co dinh.
- Line 44: concat mean theo thoi gian va std theo thoi gian. Neu matrix MFCC shape `(13, T)`, output la `(26,)`.

#### `extract_mel_spectrogram()` - lines 47-61

- Lines 47-55: docstring noi Mel bat phan bo nang luong va pattern thoi gian.
- Lines 56-59: tinh Mel spectrogram voi 128 Mel bands.
- Line 60: doi power sang dB bang `librosa.power_to_db(mel, ref=np.max)`.
- Line 61: concat 128 mean va 128 std thanh vector 256D.

#### `extract_chroma()` - lines 64-75

- Lines 64-70: docstring noi Chroma bat pitch/harmonic structure.
- Lines 72-74: tinh `chroma_stft`.
- Line 75: concat 12 mean va 12 std thanh 24D.

#### `extract_spectral_centroid()` - lines 78-89

- Lines 78-84: docstring noi centroid do do sang cua am thanh.
- Lines 86-88: tinh centroid tren tung frame.
- Line 89: tra `[mean, std]`, output 2D.

#### `extract_zcr()` - lines 92-101

- Lines 92-98: docstring noi ZCR do ti le doi dau cua waveform.
- Line 100: `librosa.feature.zero_crossing_rate(y, hop_length=HOP_LENGTH)`.
- Line 101: tra `[mean, std]`, output 2D.

#### Dac trung phu - lines 104-124

- Lines 104-107: comment noi cac feature nay khong nam trong vector 310D.
- Lines 109-112: `extract_spectral_bandwidth()`, tra mean/std bandwidth.
- Lines 115-118: `extract_spectral_rolloff()`, tra mean/std rolloff.
- Lines 121-124: `extract_rms()`, tra mean/std RMS.

Nhung ham nay co ich cho bao cao/phan tich, nhung khong duoc concat trong `extract_all()`.

#### `extract_all()` - lines 131-153

Day la contract trung tam cua project.

- Lines 131-143: docstring liet ke thanh phan vector.
- Lines 144-150: goi 5 ham theo thu tu co dinh:
  1. MFCC 26D.
  2. Mel 256D.
  3. Chroma 24D.
  4. Centroid 2D.
  5. ZCR 2D.
- Lines 151-152: assert shape phai dung `(310,)`. Neu sau nay sua feature ma quen update `FEATURE_DIM`, assert se bat loi.
- Line 153: convert sang `np.float32`, can thiet vi Faiss lam viec tot voi float32.

#### `extract_from_file()` - lines 156-170

- Lines 156-165: signature cho phep `preprocess=True`.
- Lines 166-168: neu preprocess, goi `preprocess_audio_for_features()`.
- Lines 168-169: neu khong preprocess, load truc tiep bang librosa. Che do nay co the dung cho thu nghiem, nhung search query nen dung `preprocess=True`.
- Line 170: goi `extract_all()`.

#### `FEATURE_NAMES` - lines 173-194

- Lines 177-191: tao list ten cho 310 dimension theo dung layout.
- Lines 193-194: assert so ten bang 310.

#### Main block - lines 197-219

Neu chay `python src/feature.py`, file se:

- tao random signal 2 giay;
- extract vector;
- neu co data processed thi extract file dau tien;
- in shape va ten feature dau/cuoi.

### 6.4 `src/database.py`

Muc dich: quan ly PostgreSQL metadata va file vector `.npy`.

#### Dau file va config

- Lines 1-4: docstring: PostgreSQL luu metadata, numpy binary luu vector, Faiss cho search.
- Lines 6-16: import standard, numpy, psycopg2, dotenv.
- Lines 18-20: reconfigure stdout/stderr UTF-8 neu Python ho tro. Huu ich vi output co tieng Viet.
- Line 23: load `.env` tu `CSDLDPT/.env`.
- Lines 25-31: `DB_CONFIG` lay bien moi truong, fallback:
  - host localhost
  - port 5432
  - database `animal_sounds`
  - user `csdldpt`
  - password `csdldpt123`
- Lines 33-36: constants duong dan features, scaler, sample rate, feature dim.

#### Connection management - lines 43-58

- Lines 43-50: `get_connection()` la context manager. Khi vao `with`, no connect PostgreSQL; khi thoat, no dong connection trong `finally`.
- Lines 53-58: `check_connection()` chay `SELECT 1`, neu DB khong san sang se raise exception.

#### `init_db()` - lines 65-111

Ham backup tao schema neu Docker init SQL chua chay.

- Lines 67-68: mo connection va cursor.
- Lines 70-84: tao bang `audio_files`.
- Lines 86-93: tao bang `species_stats`.
- Lines 95-103: tao bang `search_log`.
- Lines 105-108: tao index cho `species`, `filename`, `file_id`.
- Line 110: commit transaction.
- Line 111: in thong bao.

#### `insert_record()` - lines 118-156

Day la ham ghi 1 audio vao storage hybrid.

- Lines 118-128: tham so gom metadata, vector, quality, features dir.
- Line 130: dam bao features dir ton tai.
- Lines 132-135: lay ten `.npy` tu filename `.wav`, save vector float32 ra disk.
- Lines 137-138: mo connection/cursor.
- Lines 139-155: SQL insert vao `audio_files`.
- Lines 144-153: `ON CONFLICT (filename) DO UPDATE`, nghia la chay lai pipeline khong tao duplicate; record cu duoc update.
- Line 156: commit.

#### `get_all_records()` - lines 159-170

- Lines 159-160: docstring noi chi lay `quality='kept'`.
- Lines 163-169: select 8 cot can cho build artifacts, order by `id` de thu tu on dinh.
- Line 170: tra list tuple.

#### `truncate_all()` - lines 173-182

- Lines 177-180: truncate 3 bang, restart identity, cascade.
- Dung khi rebuild tu dau de id quay lai 1.

#### Scaler va vector operations - lines 189-265

`save_feature_scaler()`:

- Lines 189-193: nhan ma tran `(N, 310)`, path, optional weights.
- Lines 196-198: neu matrix rong, mean zeros va std ones.
- Lines 200-202: neu co data, mean/std theo axis 0, sau do std qua nho duoc doi thanh 1.
- Lines 203-206: save `mean`, `std`, va neu co thi `weights` vao `.npz`.
- Line 207: tra mean/std.

`load_feature_scaler()`:

- Lines 212-213: neu file scaler khong ton tai, tra `(None, None, None)`.
- Lines 214-218: load arrays va convert float32.

`apply_feature_scaler()`:

- Lines 228-229: neu thieu mean/std thi tra matrix goc.
- Lines 230-231: z-score.
- Lines 232-233: neu co weights thi nhan weights.
- Line 234: tra scaled.

`load_all_vectors()`:

- Lines 237-245: docstring noi return `(ids, filenames, species_list, matrix NxD)`.
- Line 246: lay records tu DB.
- Lines 249-256: voi moi record, neu `feat_path` ton tai thi `np.load()` vector va append metadata.
- Line 258: stack list vector thanh ma tran, neu rong thi zeros `(0, 310)`.
- Lines 259-264: neu `scaled=True`, load hoac tao scaler roi apply scaler/weights.
- Line 265: tra metadata lists va matrix.

#### Statistics va logging - lines 272-316

- Lines 272-287: `update_species_stats()` insert/update count va duration trung binh theo loai.
- Lines 290-305: `get_db_stats()` tra tong file, so loai, list `(species, file_count)`.
- Lines 308-316: `log_search()` ghi query, top1, score vao `search_log`.

#### Main block - lines 319-328

Chay file truc tiep se check DB, init DB va in stats. Neu loi, in goi y chay `.env` va Docker.

### 6.5 `scripts/init.sql`

Schema PostgreSQL duoc Docker chay lan dau khi volume moi duoc tao.

- Lines 1-3: comment huong dan reset bang `docker compose down -v && docker compose up -d`.
- Lines 5-17: bang `audio_files`:
  - `id`: primary key auto increment.
  - `file_id`: ID logic nhu `cat_0001`.
  - `filename`: ten file unique.
  - `species`: loai.
  - `filepath`: path tuong doi dung cho demo/search.
  - `duration_sec`, `sample_rate`, `source`, `quality`, `indexed_at`, `feat_path`.
- Lines 19-24: bang `species_stats`, moi loai 1 dong.
- Lines 26-32: bang `search_log`, luu lich su query.
- Lines 34-36: indexes de query theo species/filename/file_id nhanh hon.

### 6.6 `src/index_balanced8.py`

Muc dich: orchestrator build database tu raw audio.

#### Dau file

- Lines 1-5: docstring mo ta pipeline.
- Lines 7-14: import.
- Lines 16-18: reconfigure output UTF-8.
- Lines 20-21: add `src` vao `sys.path` de import local modules bang ten file.
- Lines 23-28: import DB functions, feature extraction va preprocessing.
- Lines 30-33: constants path raw, processed, excluded.

#### `detect_source()` - lines 36-49

Ham suy doan source tu filename:

- Lines 38-40: neu ten co `esc50` hoac bat dau bang `1-` den `5-`, source `esc50`.
- Lines 41-48: check `sounddino`, `animalqa`, `dynamicsuperb/ds_`, `local`.
- Line 49: fallback `unknown`.

#### `run()` - lines 52-156

- Lines 52-56: bat timer va in banner.
- Lines 58-66: step 0, check PostgreSQL. Neu fail thi in cach start Docker va return, khong crash.
- Lines 68-71: step 1, preprocess raw -> processed, skip excluded.
- Lines 73-77: step 2, init DB roi truncate old data.
- Lines 79-83: step 3, chuan bi counters va bien dem ok/fail.
- Lines 85-87: lap qua records tu preprocessing.
- Lines 89-91: double-check excluded.
- Line 93: path file processed.
- Lines 95-99: extract vector 310D va assert shape.
- Lines 100-102: doc duration bang `soundfile.info`.
- Lines 104-106: tang counter moi species, tao `file_id`.
- Lines 108-109: detect source.
- Lines 111-122: `insert_record()` ghi metadata vao PostgreSQL va vector vao `.npy`.
- Lines 123-126: cap nhat ok/fail.
- Lines 128-130: in progress moi 100 file.
- Lines 132-138: in ket qua, update stats, lay stats.
- Lines 139-141: in count tung species.
- Lines 143-146: build `data/metadata.csv`.
- Lines 148-157: in tong ket va goi y lenh tiep theo.
- Lines 159-160: neu chay file truc tiep thi goi `run()`.

### 6.7 `src/build_metadata.py`

Muc dich: tao CSV metadata tu processed audio.

- Lines 1-4: docstring noi schema CSV 7 cot.
- Lines 6-13: import.
- Lines 15-18: add `src` vao path va import excluded loader.
- Lines 20-24: path processed, output CSV, metadata goc, excluded CSV.

`load_source_map()` - lines 27-41:

- Tao map `{filename: source}` tu `balanced8_metadata.csv`.
- Neu file khong ton tai thi return map rong.
- Bat exception chung va `pass`, nghia la loi doc metadata goc khong lam pipeline fail.

`detect_source_from_filename()` - lines 44-57:

- Fallback logic giong `index_balanced8.detect_source()`, dua tren substring trong filename.

`build_metadata()` - lines 60-120:

- Lines 67-68: load excluded set va source map.
- Lines 70-72: tao rows, counters, lay danh sach `.wav`.
- Lines 74-77: skip file nam trong excluded.
- Lines 79-81: lay absolute path va species bang prefix truoc dau `_` dau tien.
- Lines 83-85: tao `file_id`.
- Lines 87-94: doc duration/sample_rate, fallback 2.0/22050 neu loi.
- Line 97: source uu tien `source_map`, fallback detect tu filename.
- Lines 99-107: append row 7 cot.
- Lines 109-111: tao DataFrame, tao folder output, ghi CSV.
- Lines 113-119: in stats neu verbose.
- Line 120: tra DataFrame.

### 6.8 `src/build_canonical.py`

Muc dich: tao artifact search tu database da index.

#### Constants va feature weights

- Lines 1-9: docstring liet ke output.
- Lines 13-20: import json/os/sys/time, faiss, numpy, pandas.
- Lines 22-26: add `src` vao path va import helpers.
- Lines 28-32: comment giai thich vi sao can feature weighting.
- Lines 33-39: tao `FEATURE_WEIGHTS`:
  - 26 phan tu dau bang 3.0 cho MFCC.
  - 256 phan tu tiep bang 1.0 cho Mel.
  - 24 phan tu tiep bang 2.0 cho Chroma.
  - 2 cho Centroid bang 2.0.
  - 2 cho ZCR bang 2.0.
- Line 40: assert len weights bang 310.
- Lines 42-44: path project root, features dir, metadata CSV.

#### `load_metadata_map()` - lines 47-54

- Doc `data/metadata.csv` bang pandas.
- Tao dict `{filename: row.to_dict()}` de sau do build `file_index.json`.

#### `build_canonical()` - lines 57-143

- Lines 57-60: bat timer, tao features dir.
- Lines 62-65: in banner.
- Lines 67-75: load vectors tu DB bang `load_all_vectors(scaled=False)`, assert dim 310.
- Lines 77-83: save raw matrix thanh `feature_db.npy`.
- Lines 85-92: fit scaler va save weights vao `feature_scaler.npz`.
- Lines 94-109: build Faiss index:
  - Line 98: safe std.
  - Line 99: z-score matrix.
  - Line 100: nhan `FEATURE_WEIGHTS`.
  - Line 102: L2 normalize row vectors.
  - Line 103: tao `faiss.IndexFlatIP(310)`.
  - Line 104: add vectors.
  - Lines 105-106: save index.
- Lines 111-134: build `file_index.json`:
  - Line 114: load metadata map.
  - Lines 116-127: voi moi vector row, tao metadata dict.
  - Lines 129-131: write JSON UTF-8.
- Lines 136-143: in tong ket.
- Lines 146-147: main block.

Neu sau nay sua vector layout, phai sua ca `FEATURE_WEIGHTS` va `evaluate_feature_space.get_feature_slices()`.

### 6.9 `src/search_engine.py`

Muc dich: core search online bang Faiss.

#### Class state

- Lines 1-10: docstring mo ta pipeline query va output schema.
- Lines 14-19: import json, os, Optional, faiss, numpy.
- Lines 22-26: class docstring noi IndexFlatIP sau L2 normalize = cosine similarity.
- Lines 28-35: `__init__` tao:
  - `dimension = 310`.
  - `index = None`.
  - `file_index = {}`.
  - `scaler_mean`, `scaler_std`, `feature_weights`.
  - `_loaded = False`.

#### `load()` - lines 37-52

- Line 43: docstring.
- Line 44: load Faiss index tu disk.
- Lines 45-47: load JSON va convert keys tu string sang int.
- Lines 48-51: load scaler arrays.
- Line 52: danh dau engine da loaded.

#### `is_loaded()` - lines 54-55

Tra True neu `_loaded` va `index` khong None.

#### `_prepare_query()` - lines 57-68

Day la phan phai khop voi `build_canonical.py`.

- Lines 57-61: docstring noi z-score -> weights -> L2.
- Line 62: assert scaler da load.
- Line 63: std qua nho thanh 1.
- Line 64: z-score query, reshape thanh `(1, 310)`, float32.
- Lines 65-66: neu co weights thi nhan weights.
- Line 67: `faiss.normalize_L2(q)`.
- Line 68: return query da san sang search.

#### `search()` - lines 70-109

- Lines 70-85: docstring mo ta args va output.
- Line 86: assert engine loaded.
- Lines 87-88: assert query shape dung `(310,)`.
- Line 90: prepare query.
- Line 91: `self.index.search(q, top_k)` tra `scores` va `indices`.
- Lines 93-97: lap qua score/index, skip `idx=-1`.
- Lines 98-100: score la inner product cua 2 vector L2-normalized, nen tuong duong cosine; clip ve `[0,1]` va round 4.
- Lines 101-107: tao dict output:
  - `rank`
  - `filepath`
  - `species`
  - `similarity_score`
  - `distance = 1 - sim`
- Line 109: tra list results.

#### Utility methods - lines 111-137

- Lines 111-113: `get_total_files()` tra `index.ntotal`.
- Lines 115-120: `get_species_list()` lay unique species tu file_index.
- Lines 123-137: `create_engine(features_dir)` tao engine va load 3 artifact:
  - `faiss.index`
  - `file_index.json`
  - `feature_scaler.npz`

#### Main block - lines 140-159

Dung de test nhanh self-match voi file dau tien trong index.

### 6.10 `src/visualization.py`

Muc dich: tao anh waveform/spectrogram/MFCC/comparison cho report va demo.

- Lines 1-5: docstring.
- Lines 7-17: import. Line 15 `matplotlib.use('Agg')` rat quan trong vi server khong can GUI.
- Lines 19-26: set style matplotlib chung.
- Lines 28-32: constants ve audio/FFT/Mel/MFCC.

`save_waveform()` - lines 35-52:

- Tao folder output neu can.
- Ve waveform bang `librosa.display.waveshow`.
- Set title/xlabel/ylabel.
- Save PNG dpi 150 va close figure de khong leak memory.

`save_mel_spectrogram()` - lines 55-78:

- Tinh mel spectrogram.
- Doi sang dB.
- Ve bang `librosa.display.specshow` voi cmap `magma`.
- Them colorbar dB.
- Save va close.

`save_mfcc_heatmap()` - lines 81-104:

- Tinh MFCC 13 coefficient.
- Ve heatmap bang cmap `coolwarm`.
- Save va close.

`save_comparison()` - lines 107-138:

- Tao subplot gom query + cac result.
- Query mau do, result dung list mau co dinh.
- Dung chung truc x thoi gian.
- Save chart comparison.

`save_similarity_bar()` - lines 141-165:

- Lay label `#rank species`.
- Lay scores.
- Ve horizontal bar chart.
- Ghi score o cuoi bar.
- Set xlim 0..1.05.

### 6.11 `src/generate_search_examples.py`

Muc dich: tao ket qua trung gian cho report.

- Lines 1-8: docstring liet ke 3 output group.
- Lines 12-19: import csv/json/os/sys/librosa/numpy.
- Lines 20-25: add `src` vao path va import feature/preprocess/search engine.
- Lines 26-29: import visualization helpers.
- Lines 31-34: path constants.

`generate_feature_examples()` - lines 37-62:

- Lines 39-40: tao folder `feature_examples`.
- Line 43: chon 4 species minh hoa: cat, dog, frog, cow.
- Line 44: list files trong processed.
- Lines 46-50: voi moi species, chon file dau tien.
- Line 52: preprocess file.
- Lines 54-59: save waveform, spectrogram, MFCC.
- Lines 61-62: print progress.

`generate_scenario()` - lines 65-172:

- Lines 65-72: signature cho 1 scenario search.
- Lines 73-74: tao folder scenario.
- Lines 76-80: preprocess query, extract vector, lay filename/species tu ten file.
- Lines 82-83: search top 5.
- Lines 85-90: save query vector, waveform, spectrogram.
- Lines 92-102: ghi `query_info.json` gom sample vector, mean/std.
- Lines 104-106: ghi `ranking.json` top-5.
- Lines 108-124: tinh ranking full voi tat ca 1042 files:
  - load `feature_db.npy`, scaler, weights;
  - transform query va DB giong search engine;
  - cosine = matrix dot query.
- Lines 126-144: ghi `ranking_full.csv`.
- Lines 146-159: voi moi result, preprocess result audio va save spectrogram.
- Lines 161-163: save comparison chart neu co result waveform.
- Line 166: save similarity bar.
- Lines 168-172: print scenario summary.

`main()` - lines 175-217:

- Tao `INTER_DIR`, load engine.
- Tao feature examples.
- Scenario 1: chon cat file dau tien trong DB, expect self-match.
- Scenario 2: chon dog file cuoi, them Gaussian noise `sigma=0.02`, save thanh `query_external_dog.wav`, roi search nhu file ngoai DB.

### 6.12 `src/evaluate_feature_space.py`

Muc dich: danh gia feature space, khong phai train classifier.

- Lines 1-7: docstring noi species labels chi dung lam metadata danh gia.
- Lines 11-20: import argparse/json/os/dataclass/combinations/numpy va feature constants.
- Lines 23-25: path constants.

`FeatureSlices` - lines 28-34:

Dataclass chua 5 slices cua vector 310D.

`get_feature_slices()` - lines 37-51:

- Tinh boundaries tu `N_MFCC`, `N_MELS`, `N_CHROMA`.
- Assert tong bang `FEATURE_DIM`.
- Return slices:
  - MFCC 0:26
  - Mel 26:282
  - Chroma 282:306
  - Centroid 306:308
  - ZCR 308:310

`load_artifacts()` - lines 54-75:

- Load feature matrix, scaler, weights, file_index.
- Lay species label theo thu tu index.
- Assert length matrix khop file_index va dim bang 310.

`normalize_rows()` - lines 78-82:

L2 normalize tung row, neu norm qua nho thi dung 1 de tranh chia 0.

`transform_features()` - lines 85-95:

Ap dung dung cong thuc search: z-score -> weights -> L2 normalize.

`precision_at_k()` - lines 98-116:

- `ranking = argsort(-similarity)[:, 1:top_k+1]`: bo cot dau vi do la self-match.
- Voi moi query, check hang xom co cung species khong.
- Tinh top1 same-species accuracy va Precision@K.
- Tra them score theo species.

`overlap_pairs()` - lines 119-130:

Tinh cap species khac nhau co mean cosine cao nhat, giup phat hien loai de nham lan.

`evaluate_variant()` - lines 133-160:

- Transform matrix theo weights cua variant.
- Tinh similarity matrix bang dot product.
- Tao mask same-class va different-class.
- Tinh intra cosine, inter cosine, gap, top1, precision@K, overlap pairs.

`build_weight_variants()` - lines 163-195:

Tao cac bien the ablation:

- `baseline_current`
- `no_chroma`
- `low_chroma_0_5`
- `low_chroma_1_0`
- `mel_downweight_0_75`
- `mel_downweight_0_5`
- `mfcc_focus`

`print_report()` - lines 202-235:

In markdown table cho ket qua evaluation.

`main()` - lines 238-257:

Parse CLI args, load artifacts, build variants, evaluate, print report, optional ghi JSON.

Ket qua hien co trong `docs/feature_space_evaluation.md`: baseline Top-1 cung loai 90.31%, Precision@5 73.01%.

### 6.13 `app/demo.py`

Muc dich: Gradio UI de user upload/record audio va xem Top-5.

#### Dau file va global setup

- Lines 1-5: docstring.
- Lines 9-19: import os/sys/tempfile, gradio, librosa, matplotlib, numpy.
- Line 17: `matplotlib.use('Agg')` cho server/headless.
- Lines 21-25: tinh `APP_DIR`, `PROJECT_ROOT`, `SRC_DIR`, add src vao path.
- Lines 27-31: import feature/preprocess/search/db/visualization.
- Lines 33-34: path constants.
- Line 37: `engine = create_engine(FEATURES_DIR)` load index ngay khi app start.

Luu y: neu `faiss.index`, `file_index.json`, `feature_scaler.npz` thieu, app co the loi ngay luc start, truoc khi user bam Search.

#### `do_search()` - lines 43-102

Day la ham callback cua nut Search.

- Lines 43-44: nhan `audio_path` tu Gradio.
- Lines 45-47: neu chua co audio thi return message va cac output rong. Luu y hien tai branch nay tra 13 gia tri, trong khi UI khai bao 14 outputs; day la diem can sua neu muon empty-input branch ben hon.
- Lines 49-51: preprocess query va extract vector 310D.
- Lines 53-54: search top 5.
- Lines 56-62: ghi log search vao PostgreSQL neu co result.
- Lines 64-66: ve spectrogram va waveform query.
- Line 69: tao HTML bang ket qua.
- Lines 71-85: voi moi result, resolve filepath, tao audio player va spectrogram.
- Line 88: tao chart similarity.
- Lines 90-94: pad list audio/spec len 5 neu thieu result.
- Lines 96-102: return 14 output theo dung thu tu UI can.

#### Plot helpers - lines 109-153

`_plot_spectrogram()`:

- Tinh mel spectrogram, convert dB, ve specshow, save vao temp PNG.

`_plot_waveform()`:

- Ve waveform va save temp PNG.

`_plot_comparison()`:

- Hien tai la bar chart Top-5 similarity, khong ve mini waveform du comment noi "mini waveforms".

#### `_build_results_html()` - lines 156-191

- Lines 157-172: build tung row HTML. Bar width = similarity x 200 px, clamp 0..200.
- Lines 174-191: return table HTML gom rank, file, species, similarity va chu thich method/vector.

Can than: HTML build bang f-string tu metadata. Metadata file path/species do project tao, nen rui ro thap; neu nhan input tu nguon khong tin cay thi nen escape HTML.

#### `get_stats_html()` - lines 198-215

- Goi `get_db_stats()`.
- Tao chips cho moi species.
- Neu DB loi, tra HTML mau do bao loi.

#### UI layout - lines 222-297

- Lines 222-225: CSS gioi han max width va center h1.
- Lines 227-234: tao `gr.Blocks`, Markdown tieu de.
- Lines 236-245: cot trai co audio input, button, DB info.
- Lines 247-251: cot phai co result HTML, query waveform/spectrogram.
- Lines 253-264: tao 5 row audio player + spectrogram result.
- Lines 265-266: image cho similarity chart.
- Lines 269-277: list outputs co 14 component.
- Line 279: bind `btn.click(do_search, inputs=[audio_in], outputs=outputs)`.
- Lines 281-286: footer Markdown feature/search/database.
- Lines 289-297: khi chay truc tiep, in stats va `demo.launch` tai `0.0.0.0:7860`.

### 6.14 `src/build_metadata.py` va `src/index_balanced8.py` co diem trung lap

Ca hai deu co logic detect source tu filename. Khac nhau nho:

- `index_balanced8.detect_source()` dung trong insert DB.
- `build_metadata.detect_source_from_filename()` dung khi source map tu `balanced8_metadata.csv` khong co.

Neu sau nay doi quy tac source, nen doi ca hai noi de metadata DB va CSV khop nhau.

### 6.15 `CSDLDPT/clean_md.py`

File nay khong tham gia pipeline search. No la script tien ich lam sach markdown bao cao.

- Lines 1-4: import regex va doc `../docs/CSDLDPT-Nhom3.md`.
- Lines 6-10: xoa form feed va page number dong rieng.
- Lines 12-19: split thanh lines, strip moi line, giu dong rong.
- Lines 21-30: neu line matching heading pattern thi them `#`, `##`, `###`.
- Lines 34-36: gom text va rut gon nhieu dong rong.
- Lines 38-39: ghi nguoc vao `../docs/CSDLDPT-Nhom3.md`.

Diem can de y: line 27 co chuoi `DANH MỤC TÙ VIẾT TẮT`, co ve la typo cua `TỪ`. Neu dung script nay lai, can can than vi no ghi de bao cao.

### 6.16 Config files

#### `docker-compose.yml`

- Lines 1-4: service `db` dung image `postgres:16-alpine`, container `csdldpt_postgres`.
- Lines 5-8: DB env lay tu `.env`, co default.
- Lines 9-10: map port `${DB_PORT:-5432}:5432`.
- Lines 11-13: mount volume `pgdata` va mount `scripts/init.sql` vao docker init dir.
- Lines 14-18: healthcheck bang `pg_isready`.
- Lines 20-21: define named volume `pgdata`.

#### `.env.example`

5 bien:

```text
DB_HOST=localhost
DB_PORT=5432
DB_NAME=animal_sounds
DB_USER=csdldpt
DB_PASSWORD=csdldpt123
```

#### `.gitignore` trong `CSDLDPT/`

Bo qua data raw/processed, features binary, audio files, cache, venv, `.env`. Dieu nay dung vi audio/vector rat lon va co the la generated artifacts.

#### `requirements.txt`

Thu vien chinh:

- `librosa`, `soundfile`: audio I/O va feature extraction.
- `numpy`, `scipy`, `pandas`: tinh toan va CSV.
- `matplotlib`: visualization.
- `gradio`: demo UI.
- `faiss-cpu`: vector search.
- `pytest`: tests.
- `psycopg2-binary`, `python-dotenv`: PostgreSQL config/connection.

---

## 7. Tests: Doc Nhu Mot Ban Dac Ta Hanh Vi

### 7.1 `tests/test_preprocessing.py`

- Lines 21-30: test `load_audio()` raise dung exception khi file khong ton tai hoac extension sai.
- Lines 33-57: test `normalize_length()`:
  - output dung 44100 samples;
  - file ngan duoc zero-pad, khong tile;
  - file dai bi truncate;
  - all silence tra zeros.
- Lines 60-71: test `normalize_amplitude()`:
  - silence duoc giu nguyen;
  - RMS sau normalize gan target -20 dB.
- Lines 74-86: test preprocess voi file real neu data ton tai.
- Lines 89-101: test excluded CSV load duoc va missing CSV tra set rong.

### 7.2 `tests/test_feature_extraction.py`

- Lines 17-19: fixture random audio 2 giay.
- Lines 22-31: test vector tong shape 310, feature names 310, constant 310.
- Lines 34-57: test tung sub-feature shape:
  - MFCC 26D.
  - Mel 256D.
  - Chroma 24D.
  - Centroid 2D.
  - ZCR 2D.
- Lines 60-71: vector khong co NaN/Inf va dtype float32.
- Lines 74-93: count feature names theo prefix.

### 7.3 `tests/test_search_engine.py`

- Lines 11-18: path artifact can co.
- Lines 21-30: fixture engine skip neu chua build artifact.
- Lines 33-45: fixture load feature_db va file_index.
- Lines 48-59: self-match rank 1:
  - query vector dau tien trong DB;
  - score >= 0.999;
  - filepath rank 1 khop file_index[0].
- Lines 62-83: test query transform:
  - weights phai load va khong phai all ones;
  - `_prepare_query()` phai khop manual transform z-score -> weights -> L2.
- Lines 86-122: output schema:
  - tra 5 results;
  - co keys bat buoc;
  - score trong 0..1;
  - distance = 1 - similarity;
  - sort giam dan;
  - rank 1..5.
- Lines 125-146: external query:
  - vector random 310D van tra 5;
  - file audio ngoai DB neu co thi tra 5 va khong self-match.
- Lines 149-158: invalid input:
  - vector sai dimension raise AssertionError;
  - engine chua load raise AssertionError.

Ket qua hien tai:

```text
42 passed, 5 warnings
```

Warnings den tu dependency/Python 3.13 va Faiss SWIG type metadata, khong lam test fail.

---

## 8. Thu Tu Chay Dung Tu Dau

Tu thu muc `CSDLDPT/`:

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

Giai thich vi sao thu tu nay quan trong:

- `docker compose up -d`: tao PostgreSQL va schema.
- `index_balanced8.py`: can DB dang chay de insert metadata.
- `build_canonical.py`: can DB da co records va vector `.npy`.
- `evaluate_feature_space.py`: can `feature_db.npy`, `feature_scaler.npz`, `file_index.json`.
- `generate_search_examples.py`: can engine load duoc `faiss.index`.
- `app/demo.py`: load engine ngay luc start, nen phai build canonical truoc.

Reset toan bo DB:

```bash
docker compose down -v
docker compose up -d
python src/index_balanced8.py
python src/build_canonical.py
```

---

## 9. Cac Diem De Nham

1. `data/processed` khong phai folder rieng, no la symlink toi `balanced8_processed`.
2. `preprocess_all()` chi quet `.wav`; nhung `load_audio()` co the load `.mp3/.flac/.ogg` khi query hoac xu ly 1 file rieng.
3. `feature_db.npy` la raw vectors. Search khong dung raw truc tiep.
4. `feature_scaler.npz` luu ca mean/std va weights. Neu weights thieu, search engine van chay nhung tests se bat loi vi weights khong duoc degrade thanh all-ones.
5. `file_index.json` keys trong JSON la string, nhung search engine convert ve int khi load.
6. `distance` trong output la `1 - clipped_similarity`, khong phai distance raw cua Faiss.
7. Query external noisy trong report co top-1 khong phai dog. Day khong co nghia code crash; no cho thay hand-crafted feature nhay voi noise.
8. UI empty input branch trong `do_search()` co kha nang tra thieu output so voi 14 Gradio outputs.
9. `clean_md.py` ghi de report root docs. Khong chay neu chua backup/noi dung report dang quan trong.

---

## 10. Cach Hoc Project Nay

Neu ban moi hoc, dung doc tat ca file mot luc. Thu tu nen hoc:

1. Doc `context/problem.md` de hieu de bai.
2. Doc `src/preprocess.py`, tap trung `preprocess_audio_for_features()`.
3. Doc `src/feature.py`, ve lai layout 310D.
4. Doc `src/build_canonical.py`, hieu z-score, weights, L2 normalize.
5. Doc `src/search_engine.py`, hieu search top-5.
6. Doc `app/demo.py`, hieu UI goi pipeline.
7. Doc tests de biet hanh vi nao duoc dam bao.
8. Doc `database.py` sau cung, vi no dai nhung logic chu yeu la storage.

Mot bai tap nho de tu kiem tra:

- Lay mot file trong `data/processed/`.
- Goi `extract_from_file(path, preprocess=True)`.
- In `vec.shape`, `vec[:5]`.
- Goi `engine.search(vec, top_k=5)`.
- Kiem tra rank 1 co phai chinh file do khong.

---

## 11. Tom Tat Tung File Mot Dong

| File | Mot dong can nho |
|---|---|
| `src/exceptions.py` | Dinh nghia 3 loai loi audio rieng. |
| `src/preprocess.py` | Chuan hoa audio ve mono 22050 Hz, 2 giay, RMS -20 dB. |
| `src/feature.py` | Tao vector 310D tu MFCC, Mel, Chroma, Centroid, ZCR. |
| `src/database.py` | Luu metadata vao PostgreSQL, vector vao `.npy`, scaler vao `.npz`. |
| `src/index_balanced8.py` | Orchestrator raw audio -> processed -> features -> DB -> metadata CSV. |
| `src/build_metadata.py` | Tao `data/metadata.csv` tu processed files. |
| `src/build_canonical.py` | Tao `feature_db.npy`, scaler, weights, `faiss.index`, `file_index.json`. |
| `src/search_engine.py` | Load artifacts va search Top-5 bang cosine/Faiss. |
| `src/visualization.py` | Luu waveform, spectrogram, MFCC, comparison chart. |
| `src/generate_search_examples.py` | Tao ket qua trung gian cho 2 kich ban search. |
| `src/evaluate_feature_space.py` | Danh gia retrieval space bang labels, ablation weights. |
| `app/demo.py` | Gradio UI cho upload/record va hien Top-5. |
| `scripts/init.sql` | PostgreSQL schema: `audio_files`, `species_stats`, `search_log`. |
| `tests/*` | Kiem tra preprocessing, feature shape, search schema/self-match. |
| `clean_md.py` | Script lam sach markdown bao cao, khong thuoc pipeline runtime. |

---

## 12. Checklist Khi Sua Code

Neu sua preprocessing:

- Chay `tests/test_preprocessing.py`.
- Rebuild vectors vi vector cu duoc tao tu preprocessing cu.
- Chay lai `index_balanced8.py` va `build_canonical.py`.

Neu sua feature layout:

- Cap nhat `FEATURE_DIM`.
- Cap nhat `FEATURE_NAMES`.
- Cap nhat `FEATURE_WEIGHTS`.
- Cap nhat `evaluate_feature_space.get_feature_slices()`.
- Cap nhat docs ve vector 310D neu dimension thay doi.
- Rebuild full artifacts.

Neu sua search transform:

- Doi ca `build_canonical.py` va `search_engine.py`.
- Doi `generate_search_examples.py` neu no tinh ranking full manual.
- Chay `tests/test_search_engine.py`.

Neu sua DB schema:

- Cap nhat `scripts/init.sql`.
- Cap nhat `database.init_db()`.
- Reset Docker volume neu can schema moi.

Neu sua UI:

- Kiem tra so output cua `do_search()` phai khop list `outputs`.
- Start demo va thu upload 1 file co trong DB, 1 file ngoai DB.

