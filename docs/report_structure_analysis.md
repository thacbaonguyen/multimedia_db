# 📋 Phân tích Cấu trúc Báo cáo Mẫu (example.pdf)

> **Chủ đề mẫu:** Hệ CSDL phân loại ảnh hoa (15 loại hoa, SVM + HOG)
> **Chủ đề của bạn:** Hệ CSDL tìm kiếm tiếng động vật (8 loài, Cosine + SVM hybrid)
> **Mục đích:** Rút ra cấu trúc trình bày, áp dụng cho chủ đề âm thanh

---

## Cấu trúc báo cáo mẫu (đã rút gọn)

```
Trang bìa:
  - HỌC VIỆN CÔNG NGHỆ BƯU CHÍNH VIỄN THÔNG
  - BÁO CÁO BÀI TẬP LỚN — HỆ CƠ SỞ DỮ LIỆU ĐA PHƯƠNG TIỆN
  - Giảng viên hướng dẫn: TS. Nguyễn Đình Hóa

Mục lục

Câu 1: Đặc điểm của kho dữ liệu
  - Số lượng, số lớp, đặc điểm từng mẫu, kích thước chuẩn hóa

Câu 2: Kỹ thuật xử lý và phân loại (lý thuyết nền tảng)
  I.  Kỹ thuật xử lý (tiền xử lý)
      1. Quá trình xử lý: thu nhận, tiền xử lý (resize, lọc trung bình)
      2. Đặc trưng dữ liệu (liệt kê các loại đặc trưng)
      3. Biểu diễn dữ liệu (vector, raster / waveform, spectrogram)
  II. Các phương pháp xử lý hiện hành (survey)
  III. Kỹ thuật phân loại
      1. SVM (giới thiệu, định nghĩa, ý tưởng, cơ sở lý thuyết, bài toán 2 lớp, đa lớp)
      2. KNN (giới thiệu, so sánh)
  IV. Phương pháp rút trích đặc trưng (HOG)
      - Giải thích chi tiết từng bước tính toán vector đặc trưng

Câu 3: Xây dựng hệ thống (phần thực hành)
  V.  Xây dựng hệ thống
      1. Sơ đồ khối (block diagram)
      2. Quá trình thực hiện
         Bước 1: Tiền xử lý (resize, lọc, chuẩn hóa)
         Bước 2: Trích rút đặc trưng (tính vector, minh họa)
         Bước 3: Huấn luyện mô hình (SVM train/test)
         Bước 4: Nhận dạng / Tìm kiếm
           4.1 Kết quả train/test (accuracy, confusion matrix)
           4.2 Kết quả tìm kiếm từ dữ liệu mới
           4.3 Code (đính kèm source code)
```

---

## Ánh xạ sang chủ đề âm thanh động vật

| Mẫu (Ảnh hoa) | → Bài của bạn (Âm thanh động vật) |
|---|---|
| Câu 1: Đặc điểm kho ảnh (200 ảnh, 15 loại, 64×64) | **Câu 1:** Đặc điểm kho âm thanh (1051 files, 8 loài, 2s mono 22050Hz) |
| Câu 2.I: Tiền xử lý (resize, lọc trung bình) | **Câu 2.I:** Tiền xử lý âm thanh (mono, resample, normalize amplitude) |
| Câu 2.II: Các PP xử lý hiện hành | **Câu 2.II:** Các PP xử lý âm thanh hiện hành (MFCC, Mel Spec, thang Mel...) |
| Câu 2.III: Kỹ thuật phân loại (SVM, KNN) | **Câu 2.III:** Kỹ thuật tìm kiếm tương tự (Cosine Similarity, Faiss, SVM reranking) |
| Câu 2.IV: Đặc trưng HOG (chi tiết tính toán) | **Câu 2.IV:** Đặc trưng âm thanh (MFCC, Mel Spec, ZCR, Spectral, Chroma — chi tiết) |
| Câu 3: Sơ đồ khối + 4 bước thực hiện | **Câu 3:** Sơ đồ khối + 4 bước thực hiện |
| Bước 1: Tiền xử lý (resize) | **Bước 1:** Tiền xử lý (mono + resample + trim + pad) |
| Bước 2: Trích rút HOG | **Bước 2:** Trích rút MFCC + Mel + ZCR + Centroid + Chroma → vector 58D (hoặc 310D) |
| Bước 3: Train SVM | **Bước 3:** Xây dựng CSDL vector + Train SVM classifier |
| Bước 4: Nhận dạng + Kết quả + Code | **Bước 4:** Tìm kiếm Top-5 + Kết quả (Hit@1, P@5, confusion matrix) + Demo + Code |

---

## Dàn ý báo cáo đề xuất cho bài của bạn

```
TRANG BÌA
  HỌC VIỆN CÔNG NGHỆ BƯU CHÍNH VIỄN THÔNG
  BÁO CÁO BÀI TẬP LỚN
  HỆ CƠ SỞ DỮ LIỆU ĐA PHƯƠNG TIỆN
  Đề tài: Hệ CSDL lưu trữ và tìm kiếm tiếng động vật
  Giảng viên hướng dẫn: TS. Nguyễn Đình Hóa
  Thành viên nhóm: ...

MỤC LỤC

═══════════════════════════════════════════

CÂU 1: ĐẶC ĐIỂM KHO DỮ LIỆU ÂM THANH
  - Tổng số file: 1,051
  - Số loài: 8 (Dog, Cat, Cow, Frog, Sheep, Monkey, Hen, Rooster)
  - Bảng phân bố số file theo loài
  - Định dạng: .wav, mono, 22050Hz, 2 giây
  - Nguồn: ESC-50, AnimalQA, DynamicSuperb, SoundDino, tự thu thập
  - Minh họa: waveform/spectrogram mẫu cho 2-3 loài

CÂU 2: KỸ THUẬT XỬ LÝ VÀ TÌM KIẾM ÂM THANH

  I. Kỹ thuật xử lý âm thanh số
     1. Tín hiệu âm thanh số (sample, sample rate, Nyquist)
     2. Tiền xử lý: chuyển mono, resample 22050Hz, trim silence, pad/truncate
     3. Normalize amplitude (RMS normalization)

  II. Các phương pháp xử lý âm thanh hiện hành
     - Thang tần số Mel và cảm nhận thính giác
     - Các kỹ thuật trích xuất đặc trưng phổ biến
     - Content-Based Audio Retrieval (CBAR)

  III. Kỹ thuật tìm kiếm tương tự
     1. Cosine Similarity (định nghĩa, công thức, ưu điểm)
     2. Euclidean Distance (so sánh)
     3. Vector Database & Faiss
     4. SVM Classifier (giới thiệu, ý tưởng, kernel, multi-class)
     5. Hybrid Reranking (75% cosine + 25% SVM)

  IV. Phương pháp trích rút đặc trưng âm thanh (chi tiết)
     1. MFCC (bản chất, công thức, tại sao phù hợp tiếng động vật, 26D)
     2. Mel Spectrogram (biểu diễn 2D, năng lượng phổ, 256D)
     3. Spectral Centroid (trung tâm phổ, độ sáng/tối, 2D)
     4. Zero Crossing Rate (tốc độ đổi dấu, 2D)
     5. Chroma Features (cao độ, melody pattern, 24D)
     6. Tổng hợp: Bảng giá trị thông tin (đặc trưng → chiều → loài phân biệt tốt)

CÂU 3: XÂY DỰNG HỆ THỐNG

  V. Xây dựng hệ thống tìm kiếm tiếng động vật

     1. Sơ đồ khối
        a) Indexing Pipeline (Audio → Preprocess → Feature → Vector DB)
        b) Query Pipeline (Query → Preprocess → Feature → Search → Top-5)

     2. Quá trình thực hiện

        Bước 1: Tiền xử lý dữ liệu
        - Input/Output minh họa
        - Waveform trước/sau xử lý

        Bước 2: Trích rút đặc trưng
        - Minh họa spectrogram, MFCC heatmap cho 2-3 loài
        - Cách tổng hợp vector (mean + std → concat)
        - Bảng vector mẫu

        Bước 3: Xây dựng CSDL + Huấn luyện mô hình
        - Cấu trúc SQLite (bảng audio_files, species_stats)
        - Z-score normalization
        - SVM training (train/val/test split, hyperparameter search)
        - Kết quả: Accuracy, F1, Confusion Matrix

        Bước 4: Tìm kiếm và đánh giá
        4.1 Kết quả Retrieval (Hit@1, P@5, bảng chi tiết theo loài)
        4.2 Kết quả tìm kiếm từ file mới (2 kịch bản: có/không trong CSDL)
            - Minh họa: spectrogram query vs top-5
        4.3 Demo UI (screenshot Gradio)
        4.4 Source Code (đính kèm hoặc tham chiếu)

CÂU 4: DEMO VÀ ĐÁNH GIÁ KẾT QUẢ
  - Screenshot demo UI
  - Nhận xét ưu/nhược điểm
  - Hướng phát triển

═══════════════════════════════════════════
```

---

## Những điểm rút ra từ báo cáo mẫu

1. **Cấu trúc theo câu hỏi đề bài** — Mỗi yêu cầu = 1 section lớn
2. **Lý thuyết trước, thực hành sau** — Câu 2 trình bày lý thuyết, Câu 3 mới là implement
3. **Giải thích chi tiết đặc trưng** — HOG được giải thích từng bước (gradient → histogram → block → normalize → vector). Tương tự, MFCC/Mel cần giải thích chi tiết
4. **Sơ đồ khối bắt buộc** — Có block diagram rõ ràng cho pipeline
5. **Kết quả train/test riêng biệt** — Có accuracy, confusion matrix
6. **Code đính kèm cuối cùng** — Source code được paste trực tiếp vào báo cáo
7. **Minh họa trực quan** — Screenshots kết quả tìm kiếm, biểu đồ
