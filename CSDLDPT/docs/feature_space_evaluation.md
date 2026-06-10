# Đánh Giá Feature Space

Mục tiêu của phần này là kiểm tra chất lượng không gian đặc trưng cho bài toán
**content-based similarity retrieval**. Nhãn loài chỉ được dùng để đánh giá
gián tiếp, không dùng để biến hệ thống thành classifier.

## Baseline và Ablation

| Variant | Intra cosine | Inter cosine | Gap | Top-1 cùng loài | Precision@5 |
|---|---:|---:|---:|---:|---:|
| baseline_current | 0.1754 | -0.0254 | 0.2009 | 90.31% | 73.01% |
| no_chroma | 0.1937 | -0.0286 | 0.2223 | 90.12% | 72.92% |
| low_chroma_0_5 | 0.1923 | -0.0284 | 0.2207 | 89.92% | 72.92% |
| low_chroma_1_0 | 0.1883 | -0.0277 | 0.2160 | 89.92% | 73.01% |
| mel_downweight_0_75 | 0.1705 | -0.0248 | 0.1953 | 90.88% | 73.26% |
| mel_downweight_0_5 | 0.1652 | -0.0240 | 0.1892 | 91.36% | 73.13% |
| mfcc_focus | 0.1866 | -0.0274 | 0.2140 | 91.17% | 74.15% |

## Precision@5 Theo Loài

| Loài | Precision@5 |
|---|---:|
| cat | 76.60% |
| cow | 66.18% |
| dog | 71.12% |
| frog | 61.50% |
| hen | 70.80% |
| monkey | 85.07% |
| rooster | 81.00% |
| sheep | 69.69% |

## Cặp Loài Overlap Cao Nhất

| Cặp loài | Mean cosine |
|---|---:|
| cow / sheep | 0.0819 |
| frog / sheep | 0.0806 |
| cow / hen | 0.0648 |
| cat / dog | 0.0578 |
| hen / sheep | 0.0557 |
| cow / frog | 0.0447 |
| frog / hen | 0.0320 |
| hen / rooster | 0.0193 |
| dog / hen | 0.0103 |
| cat / monkey | 0.0085 |

## Quyết Định

Giữ variant `baseline_current` cho hệ thống chính:

- Variant tốt nhất theo Precision@5 là `mfcc_focus`, tăng từ 73.01% lên 74.15%.
- Mức tăng chỉ là +1.14 điểm phần trăm, chưa đạt ngưỡng +2 điểm phần trăm để đổi production weights.
- Top-1 và Precision@5 hiện tại đã ổn cho scope đồ án, còn các lỗi với query ngoài CSDL chủ yếu đến từ domain mismatch/noise hơn là do thiếu classifier.

Lệnh tái lập:

```bash
python src/evaluate_feature_space.py
```
