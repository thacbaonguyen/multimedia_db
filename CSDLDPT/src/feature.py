"""
CSDLDPT - Module Trích Xuất Đặc Trưng Âm Thanh
Trích xuất vector đặc trưng 310 chiều từ mỗi file âm thanh.

Vector 310D:
  MFCC              : 26  chiều  (13 mean + 13 std)
  Mel Spectrogram   : 256 chiều  (128 mean + 128 std)
  Chroma            : 24  chiều  (12 mean + 12 std)
  Spectral Centroid : 2   chiều  (mean + std)
  ZCR               : 2   chiều  (mean + std)
  ─────────────────────────
  Tổng              : 310 chiều
"""

from __future__ import annotations

import numpy as np
import librosa

from preprocess import preprocess_audio_for_features
from exceptions import AudioFormatError, AudioFileNotFoundError, AudioProcessingError

SAMPLE_RATE = 22050
N_MFCC      = 13       # số hệ số MFCC (dùng mean + std → 26 chiều)
N_MELS      = 128      # số Mel bands (dùng mean + std → 256 chiều)
N_CHROMA    = 12       # hệ số chroma (mean + std → 24 chiều)
HOP_LENGTH  = 512
N_FFT       = 2048
FEATURE_DIM = 310


def extract_mfcc(y: np.ndarray, sr: int = SAMPLE_RATE, n_mfcc: int = N_MFCC) -> np.ndarray:
    """
    MFCC - Mel-Frequency Cepstral Coefficients (26 chiều)

    Cơ sở: Mô phỏng thang Mel của thính giác người.
    Ứng dụng: Đặc trưng mạnh nhất để phân biệt loài động vật
    nhờ capture timbre và vocal tract characteristics.
    """
    mfcc = librosa.feature.mfcc(
        y=y, sr=sr, n_mfcc=n_mfcc,
        n_fft=N_FFT, hop_length=HOP_LENGTH,
    )
    return np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1)])


def extract_mel_spectrogram(y: np.ndarray, sr: int = SAMPLE_RATE, n_mels: int = N_MELS) -> np.ndarray:
    """
    Mel Spectrogram (256 chiều: 128 mean + 128 std)

    Cơ sở: Phân tích phân bố năng lượng trên thang Mel theo thời gian.
    Ứng dụng: Capture temporal patterns — burst (chó sủa), continuous (mèo kêu),
    rhythmic (ếch kêu), harmonic (chim hót).
    Chuyển sang dB scale (librosa.power_to_db) để phù hợp cảm nhận thính giác.
    """
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=n_mels,
        n_fft=N_FFT, hop_length=HOP_LENGTH,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return np.concatenate([mel_db.mean(axis=1), mel_db.std(axis=1)])


def extract_chroma(y: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """
    Chroma STFT (24 chiều: 12 mean + 12 std)

    Cơ sở: Bắt cấu trúc cao độ (pitch) và hòa âm.
    Ứng dụng: Phân biệt loài có âm điệu rõ (chim hót, ếch kêu)
    với loài phát tiếng noise-like (côn trùng).
    """
    chroma = librosa.feature.chroma_stft(
        y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH,
    )
    return np.concatenate([chroma.mean(axis=1), chroma.std(axis=1)])


def extract_spectral_centroid(y: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """
    Spectral Centroid (2 chiều: mean + std)

    Cơ sở: Trung tâm khối phổ tần số — đo độ "sáng" (brightness) của âm thanh.
    Ứng dụng: Loài nhỏ (chim, côn trùng) có centroid cao,
    loài lớn (bò, chó) có centroid thấp.
    """
    centroid = librosa.feature.spectral_centroid(
        y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH,
    )
    return np.array([centroid.mean(), centroid.std()])


def extract_zcr(y: np.ndarray) -> np.ndarray:
    """
    Zero-Crossing Rate (2 chiều: mean + std)

    Cơ sở: Tỷ lệ tín hiệu đổi dấu — đo đặc tính voiced vs unvoiced.
    Ứng dụng: Côn trùng có ZCR rất cao (noise-like),
    thú lớn có ZCR thấp (voiced harmonics).
    """
    zcr = librosa.feature.zero_crossing_rate(y, hop_length=HOP_LENGTH)
    return np.array([zcr.mean(), zcr.std()])


# ─────────────────────────────────────────────
# Các đặc trưng phụ (KHÔNG nằm trong vector 310D chính)
# Giữ lại để báo cáo/phân tích nếu cần
# ─────────────────────────────────────────────

def extract_spectral_bandwidth(y: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Spectral Bandwidth — đặc trưng phụ, không dùng trong vector 310D."""
    bw = librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)
    return np.array([bw.mean(), bw.std()])


def extract_spectral_rolloff(y: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Spectral Rolloff — đặc trưng phụ, không dùng trong vector 310D."""
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)
    return np.array([rolloff.mean(), rolloff.std()])


def extract_rms(y: np.ndarray) -> np.ndarray:
    """RMS Energy — đặc trưng phụ, không dùng trong vector 310D."""
    rms = librosa.feature.rms(y=y, hop_length=HOP_LENGTH)
    return np.array([rms.mean(), rms.std()])


# ─────────────────────────────────────────────
# Vector tổng hợp
# ─────────────────────────────────────────────

def extract_all(y: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """
    Trích xuất TOÀN BỘ đặc trưng → vector 310D.

    Thành phần:
      MFCC              : 26  chiều  (13 mean + 13 std)
      Mel Spectrogram   : 256 chiều  (128 mean + 128 std)
      Chroma            : 24  chiều  (12 mean + 12 std)
      Spectral Centroid : 2   chiều  (mean + std)
      ZCR               : 2   chiều  (mean + std)
      ─────────────────────────────
      Tổng              : 310 chiều
    """
    feats = np.concatenate([
        extract_mfcc(y, sr),               # 26
        extract_mel_spectrogram(y, sr),     # 256
        extract_chroma(y, sr),             # 24
        extract_spectral_centroid(y, sr),  # 2
        extract_zcr(y),                    # 2
    ])
    assert feats.shape == (FEATURE_DIM,), \
        f"Expected {FEATURE_DIM}D, got {feats.shape[0]}D"
    return feats.astype(np.float32)


def extract_from_file(path: str, sr: int = SAMPLE_RATE, preprocess: bool = True) -> np.ndarray:
    """
    Đọc audio và trích vector 310D.

    Args:
        path: Đường dẫn file audio
        sr: Sample rate
        preprocess: Nếu True, dùng cùng pipeline trim/pad/normalize
                    như indexing (bắt buộc cho search query).
    """
    if preprocess:
        y, sr = preprocess_audio_for_features(path, target_sr=sr)
    else:
        y, _ = librosa.load(path, sr=sr, mono=True)
    return extract_all(y, sr)


# ─────────────────────────────────────────────
# Feature names (310 entries, dùng cho DataFrame/docs)
# ─────────────────────────────────────────────

FEATURE_NAMES = (
    # MFCC: 26
    [f'mfcc_mean_{i}' for i in range(N_MFCC)] +
    [f'mfcc_std_{i}'  for i in range(N_MFCC)] +
    # Mel Spectrogram: 256
    [f'mel_mean_{i}' for i in range(N_MELS)] +
    [f'mel_std_{i}'  for i in range(N_MELS)] +
    # Chroma: 24
    [f'chroma_mean_{i}' for i in range(N_CHROMA)] +
    [f'chroma_std_{i}'  for i in range(N_CHROMA)] +
    # Spectral Centroid: 2
    ['spectral_centroid_mean', 'spectral_centroid_std'] +
    # ZCR: 2
    ['zcr_mean', 'zcr_std']
)

assert len(FEATURE_NAMES) == FEATURE_DIM, \
    f"FEATURE_NAMES mismatch: expected {FEATURE_DIM}, got {len(FEATURE_NAMES)}"


if __name__ == '__main__':
    import os

    print("=== Test Feature Extraction 310D ===")

    # Test 1: vector từ random signal
    y_test = np.random.randn(SAMPLE_RATE * 2).astype(np.float32)
    vec = extract_all(y_test)
    print(f"✓ Random signal → shape: {vec.shape}")

    # Test 2: vector từ file thật
    test_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
    if os.path.exists(test_dir):
        files = sorted(f for f in os.listdir(test_dir) if f.endswith('.wav'))
        if files:
            path = os.path.join(test_dir, files[0])
            vec = extract_from_file(path, preprocess=True)
            print(f"✓ File: {files[0]} → shape: {vec.shape}")
            print(f"  Values (5 đầu): {vec[:5].round(4)}")
            print(f"  Feature names: {FEATURE_NAMES[:3]} ... {FEATURE_NAMES[-3:]}")

    print(f"\n✓ FEATURE_DIM = {FEATURE_DIM}")
    print(f"✓ FEATURE_NAMES count = {len(FEATURE_NAMES)}")
