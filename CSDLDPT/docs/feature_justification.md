# Giải Trình Lựa Chọn Đặc Trưng Âm Thanh

## 1. Tổng quan vector 310D

Hệ thống sử dụng vector đặc trưng **310 chiều** được tổng hợp từ 5 nhóm đặc trưng bổ sung lẫn nhau:

| STT | Đặc trưng | Số chiều | Thống kê | Tổng |
|---|---|---|---|---|
| 1 | MFCC | 13 | mean + std | **26** |
| 2 | Mel Spectrogram | 128 | mean + std | **256** |
| 3 | Chroma STFT | 12 | mean + std | **24** |
| 4 | Spectral Centroid | 1 | mean + std | **2** |
| 5 | Zero-Crossing Rate | 1 | mean + std | **2** |
| | | | **Tổng** | **310** |

## 2. Giải trình từng đặc trưng

### 2.1 MFCC — Mel-Frequency Cepstral Coefficients (26D)

**Cơ sở lý thuyết:**
- Mô phỏng thang tần số Mel của thính giác người
- Biến đổi DCT trên log Mel spectrogram → compact representation của phổ tần số
- Bất biến (invariant) với mức âm lượng tuyệt đối

**Vai trò trong hệ thống:**
- Capture **timbre** (âm sắc) — đặc trưng chính để phân biệt tiếng kêu giữa các loài
- Vocal tract characteristics: kích thước thanh quản ảnh hưởng trực tiếp đến MFCC
- Đặc trưng mạnh nhất trong audio classification (Davis & Mermelstein, 1980)

**Ví dụ phân biệt:**
- Mèo kêu: MFCC thấp tập trung, formant rõ ràng
- Ếch kêu: MFCC có pattern tuần hoàn
- Côn trùng: MFCC phẳng, thiếu formant

### 2.2 Mel Spectrogram (256D)

**Cơ sở lý thuyết:**
- Phân tích phân bố năng lượng trên 128 Mel bands theo thời gian
- Chuyển sang dB scale (`librosa.power_to_db`) để phù hợp cảm nhận thính giác logarithmic
- Giữ nguyên thông tin temporal mà MFCC đã nén qua DCT

**Vai trò trong hệ thống:**
- Capture **temporal patterns**: burst (chó sủa), continuous (mèo kêu), rhythmic (ếch kêu), harmonic (chim hót)
- Phân biệt cấu trúc thời gian: tiếng sủa ngắn vs. tiếng rên dài
- Bổ sung cho MFCC: MFCC capture shape phổ, Mel Spectrogram capture distribution năng lượng

**Lý do chọn 128 bands:**
- Cân bằng giữa chi tiết và kích thước vector
- 128 bands = resolution ~170 Hz/band ở dải thấp, đủ phân giải cho animal vocalizations (100 Hz – 8 kHz)

### 2.3 Chroma STFT (24D)

**Cơ sở lý thuyết:**
- Ánh xạ phổ tần số về 12 pitch classes (C, C#, D, ..., B)
- Bất biến với octave: C4 và C5 cùng chroma
- Capture cấu trúc hòa âm (harmonic structure)

**Vai trò trong hệ thống:**
- Phân biệt loài có **âm điệu rõ** (chim hót, ếch kêu) với loài **noise-like** (côn trùng)
- Tiếng chim: chroma có peak rõ ở vài pitch classes → melodic
- Tiếng côn trùng: chroma phẳng → broadband noise

### 2.4 Spectral Centroid (2D)

**Cơ sở lý thuyết:**
- Trung tâm khối (center of mass) của phổ tần số
- Đo **brightness** — tần số trung bình có trọng số năng lượng
- Centroid = Σ(f × M(f)) / Σ(M(f))

**Vai trò trong hệ thống:**
- Loài nhỏ (chim, côn trùng): centroid **cao** (nhiều năng lượng ở tần số cao)
- Loài lớn (bò, chó): centroid **thấp** (năng lượng tập trung tần số thấp)
- Compact discriminator: chỉ 2D nhưng phân biệt nhanh giữa nhóm loài

### 2.5 Zero-Crossing Rate (2D)

**Cơ sở lý thuyết:**
- Tỷ lệ tín hiệu đổi dấu qua 0 trong mỗi frame
- Đo đặc tính **voiced vs. unvoiced**
- ZCR thấp → tín hiệu tuần hoàn (voiced), ZCR cao → noise-like

**Vai trò trong hệ thống:**
- Côn trùng: ZCR **rất cao** (stridulation = noise-like)
- Thú lớn (bò, chó): ZCR **thấp** (voiced harmonics)
- Bổ sung cho Centroid: cùng phân nhóm nhưng từ góc nhìn temporal thay vì spectral

## 3. Lý do KHÔNG đưa vào vector chính

| Đặc trưng | Lý do loại bỏ |
|---|---|
| Spectral Bandwidth | Tương quan cao với Centroid, thêm noise vào cosine distance |
| Spectral Rolloff | Redundant với Mel Spectrogram bands cao |
| Spectral Flatness | Vai trò trùng với ZCR (cùng đo voiced/unvoiced) |
| RMS Energy | Bất biến sau normalize amplitude, không mang thông tin phân biệt |

Các đặc trưng trên được giữ lại trong code (`feature.py`) dưới dạng utility functions cho phân tích, nhưng **không nằm trong vector 310D** để tránh tăng chiều không cần thiết và giảm hiệu quả cosine similarity.

## 4. Pipeline chuẩn hóa

```
Audio file → load (mono, 22050 Hz)
           → trim silence (librosa.effects.trim, top_db=20)
           → truncate hoặc zero-pad (2 giây = 44100 samples)
           → normalize amplitude (RMS, target -20 dB)
           → extract 310D feature vector
           → z-score normalize (per-dimension, fit trên toàn DB)
           → L2 normalize → Faiss IndexFlatIP (= Cosine Similarity)
```

## 5. Tham khảo

- Davis, S., & Mermelstein, P. (1980). Comparison of parametric representations for monosyllabic word recognition in continuously spoken sentences. *IEEE Transactions on ASSP*.
- Logan, B. (2000). Mel Frequency Cepstral Coefficients for Music Modeling. *ISMIR*.
- librosa documentation: https://librosa.org/doc/latest/
