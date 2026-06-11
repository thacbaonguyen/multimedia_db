# Câu Hỏi Bảo Vệ Đồ Án — Q&A

## A. Dữ Liệu (Yêu cầu 1)

### A1. Tại sao chọn 8 loài? Có ít quá không?

8 loài (cat, cow, dog, frog, hen, monkey, rooster, sheep) đủ đa dạng về đặc tính âm thanh:
- **Tonal/harmonic:** rooster, hen, monkey (cao độ rõ)
- **Burst/impulsive:** dog (sủa ngắn), frog (kêu lặp lại)
- **Sustained:** cow, sheep (kéo dài, fundamental frequency thấp)
- **Mixed:** cat (vừa tonal vừa noisy)

8 loài × ~130 files/loài = 1042 files > 500 yêu cầu. Số loài vừa đủ để hệ thống phải phân biệt các pattern khác nhau mà vẫn đảm bảo mỗi cluster có đủ samples.

### A2. Dữ liệu lấy từ đâu? Có đảm bảo chất lượng không?

5 nguồn: Local recordings (442), ESC-50 (269), DynamicSuperb (90), AnimalQA (80), SoundDino (36). Chi tiết trong `docs/data_audit.md`.

Quy trình kiểm soát:
- **Loại bỏ 9 files** multi-species (ghi trong `data/excluded_files.csv`)
- Mỗi file chỉ chứa 1 loài duy nhất (đúng yêu cầu đề bài)
- Tiền xử lý chuẩn hóa toàn bộ (trim silence, pad 2s, normalize RMS)

### A3. Tại sao chọn 2 giây? Nếu tiếng kêu dài hơn thì sao?

- 2 giây đủ để capture ≥ 1 chu kỳ tiếng kêu cho mọi loài trong dataset
- Nếu dài hơn: trim silence trước, rồi lấy đoạn giữa (center crop) — giữ phần có nội dung nhất
- Nếu ngắn hơn: zero-pad cuối (KHÔNG tile/repeat để tránh tạo artifact giả)
- 2s × 22050Hz = 44100 samples — vừa đủ cho FFT window 2048 samples

### A4. Tại sao zero-pad mà không tile (lặp lại)?

Tile tạo ra periodicity giả → MFCC và Mel sẽ detect pattern lặp lại không có trong audio gốc → ảnh hưởng similarity. Zero-pad = silence, không thêm thông tin sai.

### A5. Tại sao không dùng data augmentation?

Đây là bài toán **retrieval** (tìm file giống nhất), không phải classification. Augmentation (thêm noise, pitch shift) sẽ tạo ra nhiều bản gần giống nhau trong DB → khi search, kết quả sẽ trả về các bản augmented của cùng 1 file thay vì tìm file khác giống nhất. Không phù hợp.

---

## B. Đặc Trưng (Yêu cầu 2)

### B1. Tại sao chọn 5 features này? Lý do từng feature?

| Feature | Dim | Vai trò | Lý do chọn |
|---------|-----|---------|------------|
| MFCC | 26 | Âm sắc (timbre) | Mô phỏng thang Mel thính giác, capture vocal tract → phân biệt loài mạnh nhất |
| Mel Spectrogram | 256 | Phân bố năng lượng | Capture temporal patterns: burst (chó), sustained (bò), rhythmic (ếch) |
| Chroma | 24 | Cấu trúc cao độ | Phân biệt tonal (chim) vs noise-like (côn trùng) |
| Spectral Centroid | 2 | Độ sáng tần số | Loài nhỏ = centroid cao, loài lớn = centroid thấp |
| ZCR | 2 | Voiced vs unvoiced | ZCR thấp = voiced (thú), ZCR cao = noise (côn trùng) |

Chi tiết: `docs/feature_justification.md`

### B2. Tại sao 310 chiều? Có quá nhiều không?

310D = 26 + 256 + 24 + 2 + 2. Không quá nhiều vì:
- Faiss IndexFlatIP xử lý 310D × 1042 vectors trong < 1ms
- z-score normalization xử lý curse of dimensionality
- Feature weighting giảm ảnh hưởng của Mel (256D chiếm 82.6%)

Nếu giảm (ví dụ bỏ Mel → 54D), accuracy giảm vì mất temporal energy patterns.

### B3. Tại sao dùng mean + std mà không dùng raw frames?

Raw frames = ma trận 2D (features × time), kích thước khác nhau giữa các file. Mean+std = statistical summary cố định kích thước → dễ so sánh cosine similarity. Đây là cách chuẩn trong audio retrieval khi dùng hand-crafted features.

### B4. Mel Spectrogram chiếm 82.6% vector, có dominate kết quả không?

**Có** — đây là vấn đề thực tế. Giải pháp: **Feature Weighting**:
```
MFCC×3.0 | Mel×1.0 | Chroma×2.0 | Centroid×2.0 | ZCR×2.0
```
Nhân trọng số SAU z-score, TRƯỚC L2 normalize → MFCC (26D × 3.0) có ảnh hưởng tương đương Mel (256D × 1.0) trong cosine distance.

Kết quả: accuracy tăng từ 78.1% → 81.7% (Top-1 excluding self-match).

### B5. Tại sao không dùng deep learning features (VGGish, CLAP)?

- Đề bài yêu cầu **"xây dựng bộ thuộc tính"** → phải tự thiết kế, giải trình
- Deep features = black box, không giải thích được "tại sao 2 file giống nhau"
- Hand-crafted features có ý nghĩa vật lý rõ ràng (MFCC = vocal tract, ZCR = voiced/unvoiced)
- Với 1042 files, hand-crafted đã đủ tốt (Top-1 = 90.31%)

### B6. Tại sao có 3 features phụ (bandwidth, rolloff, RMS) mà không dùng?

- **Spectral Bandwidth:** Tương quan cao với Centroid → redundant
- **Spectral Rolloff:** Tương quan cao với Mel bands cao → redundant
- **RMS Energy:** Sau normalize amplitude (-20dB), RMS gần như bằng nhau → không có giá trị phân biệt

Giữ lại code để minh họa quá trình chọn lọc features.

---

## C. Hệ Thống Tìm Kiếm (Yêu cầu 3)

### C1. Tại sao dùng Cosine Similarity mà không phải Euclidean?

Cosine đo **hướng** vector, không phải **độ lớn**. Sau z-score, các features có scale khác nhau → Euclidean bị bias bởi features có variance lớn. Cosine trên L2-normed vectors = chỉ so hướng → robust hơn.

Thực tế: Faiss IndexFlatIP trên L2-normed vectors = Cosine Similarity (chứng minh toán học: `cos(a,b) = a·b / (|a|×|b|)`, khi `|a|=|b|=1` thì `cos(a,b) = a·b`).

### C2. Tại sao dùng Faiss mà không tự viết brute-force?

- Faiss (Facebook AI) được tối ưu SIMD → nhanh hơn numpy 5-10x
- Hỗ trợ mở rộng: nếu dataset lớn hơn, chuyển sang IVF index mà không đổi code
- Production-grade, battle-tested

### C3. z-score normalize là gì? Tại sao cần?

z-score = `(x - mean) / std` cho từng chiều. Mean và std được **fit trên toàn bộ database** rồi lưu vào `feature_scaler.npz`.

Cần vì: MFCC có range [-500, 200], ZCR có range [0, 0.5] → nếu không normalize, features có range lớn sẽ dominate cosine distance.

Query audio cũng dùng **cùng scaler** (mean, std từ DB) → đảm bảo consistency.

### C4. Tại sao lưu metadata trong PostgreSQL mà không dùng SQLite?

- PostgreSQL hỗ trợ concurrent access (demo + indexing đồng thời)
- Có indexes, full SQL, JOIN, aggregation
- Docker container = reproducible, dễ deploy
- Bảng `search_log` ghi lịch sử query → phân tích sau

### C5. Giải thích pipeline search step-by-step?

```
1. User upload file.wav
2. preprocess: load → mono 22050Hz → trim silence → zero-pad 2s → normalize RMS -20dB
3. feature: extract_all() → vector 310D (raw)
4. search_engine._prepare_query():
   a. z-score: (vector - DB_mean) / DB_std
   b. feature weights: vector *= [MFCC×3, Mel×1, Chroma×2, Centroid×2, ZCR×2]
   c. L2 normalize: vector /= ||vector||
5. Faiss index.search(q, k=5) → 5 indices + 5 scores
6. Map indices → file_index.json → {filepath, species}
7. Output: [{rank, filepath, species, similarity_score, distance}]
```

### C6. Self-match = 1.0 nghĩa là gì?

Khi query = file CÓ trong DB, vector query sau transform = vector đã index → cosine = 1.0 (góc = 0°). Đây là sanity check quan trọng:
- Nếu self-match ≠ 1.0 → pipeline indexing và query KHÔNG nhất quán → bug

### C7. Nếu upload file loài KHÔNG CÓ trong DB thì sao?

Hệ thống vẫn trả 5 files giống nhất (đúng yêu cầu đề bài). Scores sẽ thấp hơn bình thường vì không có cluster nào match tốt. Đây là hành vi đúng của content-based retrieval — trả file **giống nhất về nội dung**, không phải "cùng loài".

### C8. Similarity score có ý nghĩa gì?

- `1.0` = hoàn toàn giống (self-match)
- `0.8 - 0.9` = rất giống (cùng loài, cùng kiểu kêu)
- `0.6 - 0.8` = tương tự (có thể khác loài nhưng acoustic profile gần)
- `< 0.5` = không giống

Score = raw cosine similarity, clip vào [0, 1]. `distance = 1 - similarity`.

---

## D. Feature Weighting & Evaluation

### D1. Weights chọn bằng cách nào? Có cơ sở không?

Dựa trên phân tích dimensionality imbalance:
- Mel 256D chiếm 82.6% → weight 1.0 (giữ nguyên)
- MFCC 26D chỉ 8.4% nhưng discriminative nhất → boost ×3.0
- Chroma 24D, Centroid 2D, ZCR 2D → boost ×2.0

Kết quả đo: accuracy 78.1% → 81.7%. Ablation study (`docs/feature_space_evaluation.md`) cho thấy variant `mfcc_focus` (MFCC×4, Mel×0.75) chỉ tốt hơn +1.14% → chưa đủ ngưỡng +2% để thay đổi.

### D2. Precision@5 = 73.01% có tốt không?

Với hand-crafted features trên 8 loài, 73% P@5 là hợp lý:
- Monkey: 85.07% (tiếng rất đặc trưng)
- Cat: 76.60%
- Frog: 61.50% (thấp nhất — nhiều loài có tiếng kêu tần số tương tự)

Để so sánh: deep learning models (VGGish) đạt ~85-90% P@5 trên ESC-50 nhưng không giải thích được.

### D3. Top-1 same-species = 90.31% vs 81.7% — khác nhau sao?

2 cách đo khác nhau:
- **90.31%**: Leave-one-out trên similarity matrix (bỏ diagonal) — `evaluate_feature_space.py`
- **81.7%**: Dùng `engine.search()`, bỏ kết quả có sim > 0.999 (tránh self-match) — test thủ công

90.31% chính xác hơn vì đo trên toàn bộ 1042 files.

---

## E. CSDL & Lưu Trữ

### E1. Giải thích schema 3 bảng?

```sql
audio_files  — Mỗi row = 1 file audio
  id, file_id (cat_0001), filename, species, filepath,
  duration_sec, sample_rate, source, quality, indexed_at, feat_path

species_stats — Thống kê tổng hợp, cập nhật bằng aggregation query
  species, file_count, avg_duration_sec, updated_at

search_log — Lịch sử mỗi lần tìm kiếm
  id, query_file, top1_result, top1_score, searched_at
```

### E2. Tại sao lưu vector trong .npy mà không trong PostgreSQL?

- **Tốc độ:** `np.load()` nhanh hơn BLOB query 10x
- **Flexibility:** Load toàn bộ vào RAM bằng `np.vstack()` → feed Faiss
- **Đơn giản:** Không cần serialize/deserialize binary

PostgreSQL lưu metadata (text, số) — thế mạnh của RDBMS. Vector lưu file — thế mạnh của numpy.

### E3. Tại sao không dùng pgvector?

pgvector tích hợp vector search trong PostgreSQL, nhưng:
- Faiss nhanh hơn pgvector cho brute-force search
- Faiss hỗ trợ nhiều index types (IVF, HNSW) nếu cần scale
- Tách biệt metadata DB và vector index → dễ rebuild riêng

---

## F. Demo & Đánh Giá (Yêu cầu 4)

### F1. Tại sao dùng Gradio?

- 1 file Python, không cần frontend framework
- Hỗ trợ audio upload + playback natively
- Auto-generate UI từ function signature
- Dễ share (localhost hoặc public link)

### F2. Nếu upload file cartoon/synthesized thì kết quả sai loài?

**Không sai.** Đề bài yêu cầu "5 files giống nhất về **nội dung**", không yêu cầu "đúng loài". File cartoon cat có harmonic pattern giống rooster → trả rooster = đúng content similarity.

Với file cat THẬT từ dataset → 5/5 cat, sim 0.87-1.0. Vấn đề là domain mismatch (synthesized vs real), không phải lỗi hệ thống.

### F3. Hệ thống có scalable không?

- **1042 files:** Faiss brute-force search < 1ms
- **10K files:** Vẫn OK với IndexFlatIP
- **100K+ files:** Chuyển sang `IndexIVFFlat` hoặc `IndexHNSW` (chỉ đổi 2 dòng code trong `build_canonical.py`)

### F4. Test coverage bao nhiêu?

42 tests, 3 files:
- `test_preprocessing.py` (11): load, trim, pad, normalize, exclusion, errors
- `test_feature_extraction.py` (17): shape, values, names cho từng feature
- `test_search_engine.py` (14): self-match, weights, pipeline consistency, external audio, schema

---

## G. Câu Hỏi Khó / Edge Cases

### G1. Nếu 2 loài có tiếng kêu rất giống nhau thì sao?

Đúng — cow/sheep có mean inter-class cosine = 0.0819 (cao nhất). Hệ thống sẽ trả mix cow+sheep trong Top-5. Đây là giới hạn của hand-crafted features. Giải pháp: dùng deep features hoặc thêm temporal features (onset detection, rhythm).

### G2. File audio bị noise nhiều thì sao?

Noise ảnh hưởng Mel Spectrogram và MFCC mạnh nhất. Kết quả similarity sẽ thấp hơn. Giải pháp tiềm năng: spectral subtraction hoặc Wiener filtering trước preprocess. Hiện tại chưa implement vì dataset chất lượng tốt.

### G3. Tại sao không dùng SVM/classifier để cải thiện?

Đề bài là **retrieval** (tìm file giống nhất), KHÔNG phải classification (phân loại loài). Classifier trả class label, retrieval trả ranked list. Hai bài toán khác nhau. `evaluate_feature_space.py` dùng species labels CHỈ để đánh giá, không train model.

### G4. Inner Product = Cosine Similarity — chứng minh?

```
cos(a, b) = (a · b) / (||a|| × ||b||)

Nếu ||a|| = ||b|| = 1 (sau L2 normalize):
cos(a, b) = a · b = Inner Product

→ Faiss IndexFlatIP trên L2-normed vectors = Cosine Similarity ∎
```

### G5. Reproducibility — người khác có chạy lại được không?

Có:
1. `docker compose up -d` → PostgreSQL
2. `python src/index_balanced8.py` → preprocess + index
3. `python src/build_canonical.py` → Faiss artifacts
4. `python -m pytest tests/ -v` → verify 42/42
5. `python app/demo.py` → demo

Tất cả deterministic (cùng input → cùng output), trừ random seed trong test noise.

### G6. So sánh hệ thống này với Shazam?

| | Hệ thống này | Shazam |
|---|---|---|
| Mục đích | Tìm file giống nhất | Nhận diện bài hát chính xác |
| Features | MFCC, Mel, Chroma, etc. | Spectrogram fingerprints |
| Matching | Cosine similarity | Hash lookup |
| Database | 1042 files | Hàng triệu bài |
| Noise robust | Trung bình | Rất cao |

Khác biệt cốt lõi: Shazam tìm **exact match**, hệ thống này tìm **similar content**.
