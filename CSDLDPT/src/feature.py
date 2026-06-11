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

# âm sắc
def extract_mfcc(y: np.ndarray, sr: int = SAMPLE_RATE, n_mfcc: int = N_MFCC) -> np.ndarray:
    mfcc = librosa.feature.mfcc(
        y=y, sr=sr, n_mfcc=n_mfcc,
        n_fft=N_FFT, hop_length=HOP_LENGTH,
    )
    return np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1)])

# tần số, dải âm theo time
def extract_mel_spectrogram(y: np.ndarray, sr: int = SAMPLE_RATE, n_mels: int = N_MELS) -> np.ndarray:
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=n_mels,
        n_fft=N_FFT, hop_length=HOP_LENGTH,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return np.concatenate([mel_db.mean(axis=1), mel_db.std(axis=1)])

# cao độ: chim,,,
def extract_chroma(y: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    chroma = librosa.feature.chroma_stft(
        y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH,
    )
    return np.concatenate([chroma.mean(axis=1), chroma.std(axis=1)])

# độ ság âm
def extract_spectral_centroid(y: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """(chim, côn trùng) có centroid cao,
    loài lớn (bò, chó) có centroid thấp.
    """
    centroid = librosa.feature.spectral_centroid(
        y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH,
    )
    return np.array([centroid.mean(), centroid.std()])

# đổi dấu, thay đổi âm
def extract_zcr(y: np.ndarray) -> np.ndarray:
    zcr = librosa.feature.zero_crossing_rate(y, hop_length=HOP_LENGTH)
    return np.array([zcr.mean(), zcr.std()])


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


def extract_all(y: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
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
    if preprocess:
        y, sr = preprocess_audio_for_features(path, target_sr=sr)
    else:
        y, _ = librosa.load(path, sr=sr, mono=True)
    return extract_all(y, sr)


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

    y_test = np.random.randn(SAMPLE_RATE * 2).astype(np.float32)
    vec = extract_all(y_test)
    print(f"✓ Random signal → shape: {vec.shape}")

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
