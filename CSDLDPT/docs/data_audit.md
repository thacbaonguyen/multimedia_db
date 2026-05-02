# Data Audit Log

## Phương pháp
- Quét toàn bộ 45 SoundDino files theo tên file để phát hiện multi-species hoặc non-vocalization
- Tiêu chí exclude: tên file đề cập ≥ 2 loài, hoặc tiếng không phải tiếng kêu động vật (mechanical, narration)
- Kết quả ghi vào `data/excluded_files.csv`

## Kết quả
- **Tổng SoundDino:** 45 files (Cow 11, Sheep 10, Dog 11 không dùng trong audit, Cat 14)
- **Excluded:** 9 files
- **Kept:** 36 SoundDino files
- **Tổng dataset sau audit:** 1042 files (vẫn ≥ 500)

## Chi tiết files excluded

| Filename | Species | Reason |
|---|---|---|
| `sounddino_village-birds-cicadas-lowing-cows.wav` | Cow | Multi-species: birds + cicadas + cows |
| `sounddino_nature-birds-lowing-cows-in-the-pasture.wav` | Cow | Multi-species: birds + cows |
| `sounddino_village-birds-cows-goats-chickens-in-the-distance.wav` | Cow | Multi-species: birds + cows + goats + chickens |
| `sounddino_on-the-farm-roosters-cows-birds-in-the-background.wav` | Cow | Multi-species: roosters + cows + birds |
| `sounddino_the-sound-of-a-cow-being-milked.wav` | Cow | Non-vocalization: milking mechanical sound |
| `sounddino_milking-a-cow-against-the-background-of-mooing.wav` | Cow | Mixed: milking noise + mooing background |
| `sounddino_learn-to-name-a-cow.wav` | Cow | Possibly contains human narration |
| `sounddino_cows-on-the-farm.wav` | Cow | Ambiguous farm ambience |
| `sounddino_a-large-flock-of-sheep-is-led-to-pasture.wav` | Sheep | Herding sounds mixed with bleating |

## Nguồn khác (ESC-50, AnimalQA, DynamicSuperb)
- Các nguồn này đã được curate bởi tổ chức nghiên cứu, mỗi file đã được gán nhãn đơn loài
- Không phát hiện vấn đề multi-species từ metadata
