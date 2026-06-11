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


def _frame_signal_manual(
    y: np.ndarray,
    frame_length: int = N_FFT,
    hop_length: int = HOP_LENGTH,
    center: bool = True,
) -> np.ndarray:
    """Chia waveform thành các frame chồng lấn bằng NumPy."""
    y = np.asarray(y, dtype=np.float32)
    if center:
        pad = frame_length // 2
        y = np.pad(y, (pad, pad), mode='constant')

    if y.size < frame_length:
        y = np.pad(y, (0, frame_length - y.size), mode='constant')

    n_frames = 1 + (y.size - frame_length) // hop_length
    if n_frames <= 0:
        return np.zeros((1, frame_length), dtype=np.float32)

    shape = (n_frames, frame_length)
    strides = (y.strides[0] * hop_length, y.strides[0])
    frames = np.lib.stride_tricks.as_strided(y, shape=shape, strides=strides)
    return frames.copy().astype(np.float32)


def _stft_power_manual(
    y: np.ndarray,
    sr: int = SAMPLE_RATE,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Tự tính STFT power: frame -> Hann window -> rFFT -> |X|^2."""
    frames = _frame_signal_manual(y, frame_length=n_fft, hop_length=hop_length, center=True)
    window = np.hanning(n_fft).astype(np.float32)
    windowed = frames * window
    spectrum = np.fft.rfft(windowed, n=n_fft, axis=1)
    magnitude = np.abs(spectrum).astype(np.float32)
    power = (magnitude ** 2).astype(np.float32)
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr).astype(np.float32)
    return power, magnitude, freqs


def _hz_to_mel_manual(hz: np.ndarray | float) -> np.ndarray | float:
    """Chuyển Hz sang thang Mel theo công thức HTK."""
    return 2595.0 * np.log10(1.0 + np.asarray(hz) / 700.0)


def _mel_to_hz_manual(mel: np.ndarray | float) -> np.ndarray | float:
    """Chuyển Mel về Hz theo công thức HTK."""
    return 700.0 * (10.0 ** (np.asarray(mel) / 2595.0) - 1.0)


def _mel_filter_bank_manual(
    sr: int = SAMPLE_RATE,
    n_fft: int = N_FFT,
    n_mels: int = N_MELS,
) -> np.ndarray:
    """Tạo Mel filter bank tam giác shape (n_mels, n_fft//2 + 1)."""
    n_freqs = n_fft // 2 + 1
    min_mel = _hz_to_mel_manual(0.0)
    max_mel = _hz_to_mel_manual(sr / 2.0)
    mel_points = np.linspace(min_mel, max_mel, n_mels + 2)
    hz_points = _mel_to_hz_manual(mel_points)
    bin_points = np.floor((n_fft + 1) * hz_points / sr).astype(int)
    bin_points = np.clip(bin_points, 0, n_freqs - 1)

    filters = np.zeros((n_mels, n_freqs), dtype=np.float32)
    for i in range(n_mels):
        left, center, right = bin_points[i], bin_points[i + 1], bin_points[i + 2]

        if center > left:
            filters[i, left:center] = (
                np.arange(left, center, dtype=np.float32) - left
            ) / (center - left)
        if right > center:
            filters[i, center:right] = (
                right - np.arange(center, right, dtype=np.float32)
            ) / (right - center)

    return filters


def _power_to_db_manual(power: np.ndarray, amin: float = 1e-10) -> np.ndarray:
    """Đổi power spectrogram sang dB theo ref=max(power)."""
    power = np.maximum(power, amin)
    ref = max(float(np.max(power)), amin)
    return (10.0 * np.log10(power / ref)).astype(np.float32)


def extract_zcr_manual(y: np.ndarray) -> np.ndarray:
    """Tự tính ZCR: tỷ lệ số lần tín hiệu đổi dấu trong từng frame."""
    frames = _frame_signal_manual(y, frame_length=N_FFT, hop_length=HOP_LENGTH, center=True)
    signs = np.signbit(frames)
    crossings = np.count_nonzero(signs[:, 1:] != signs[:, :-1], axis=1)
    zcr = crossings.astype(np.float32) / max(N_FFT - 1, 1)
    return np.array([zcr.mean(), zcr.std()], dtype=np.float32)


def extract_spectral_centroid_manual(y: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Tự tính Spectral Centroid: trọng tâm phổ tần số của từng frame."""
    _, magnitude, freqs = _stft_power_manual(y, sr=sr)
    denom = magnitude.sum(axis=1)
    numer = (magnitude * freqs.reshape(1, -1)).sum(axis=1)
    centroid = np.divide(
        numer,
        denom,
        out=np.zeros_like(numer, dtype=np.float32),
        where=denom > 1e-10,
    )
    return np.array([centroid.mean(), centroid.std()], dtype=np.float32)


def extract_mel_spectrogram_manual(
    y: np.ndarray,
    sr: int = SAMPLE_RATE,
    n_mels: int = N_MELS,
) -> np.ndarray:
    """Tự tính Mel Spectrogram 256D: Mel power -> dB -> mean/std."""
    power, _, _ = _stft_power_manual(y, sr=sr)
    mel_filters = _mel_filter_bank_manual(sr=sr, n_fft=N_FFT, n_mels=n_mels)
    mel_power = mel_filters @ power.T
    mel_db = _power_to_db_manual(mel_power)
    feats = np.concatenate([mel_db.mean(axis=1), mel_db.std(axis=1)])
    return feats.astype(np.float32)


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
