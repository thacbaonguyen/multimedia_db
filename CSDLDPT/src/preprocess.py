"""
CSDLDPT - Module Tiền Xử Lý Âm Thanh
Chuẩn hóa tất cả file âm thanh: sample rate, độ dài, biên độ.

Pipeline: load → trim silence → truncate/zero-pad → normalize amplitude → save
"""

from __future__ import annotations

import csv
import logging
import os
from typing import Optional

import librosa
import numpy as np
import soundfile as sf

from exceptions import AudioFileNotFoundError, AudioFormatError, AudioProcessingError

logger = logging.getLogger(__name__)

# Tham số chuẩn hóa toàn cục
SAMPLE_RATE = 22050      # Hz - chuẩn cho audio ML
DURATION    = 2.0        # giây - độ dài chuẩn mỗi clip
N_SAMPLES   = int(SAMPLE_RATE * DURATION)

VALID_AUDIO_EXTS = {'.wav', '.mp3', '.flac', '.ogg'}

RAW_DIR       = os.path.join(os.path.dirname(__file__), '..', 'data', 'balanced8_raw')
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'balanced8_processed')
EXCLUDED_CSV  = os.path.join(os.path.dirname(__file__), '..', 'data', 'excluded_files.csv')


def load_excluded_filenames(csv_path: str = EXCLUDED_CSV) -> set[str]:
    """
    Đọc danh sách file bị exclude từ data/excluded_files.csv.
    Trả về set các filename có decision='excluded'.
    Match cả raw filename (sounddino_x.wav) và processed filename (cow_sounddino_x.wav).
    """
    excluded: set[str] = set()
    if not os.path.exists(csv_path):
        return excluded
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get('decision', '').strip().lower() == 'excluded':
                raw_name = row['filename'].strip()
                excluded.add(raw_name)
                # Cũng thêm dạng processed: {species}_{raw_name}
                species = row.get('species', '').strip().lower()
                if species:
                    excluded.add(f"{species}_{raw_name}")
    return excluded


def load_audio(path: str, target_sr: int = SAMPLE_RATE) -> tuple[np.ndarray, int]:
    """
    Đọc file audio, chuyển về mono + resample.
    Raises AudioFileNotFoundError, AudioFormatError, AudioProcessingError.
    """
    if not os.path.exists(path):
        raise AudioFileNotFoundError(f"Không tìm thấy: {path}")
    ext = os.path.splitext(path)[1].lower()
    if ext not in VALID_AUDIO_EXTS:
        raise AudioFormatError(f"Không phải audio hợp lệ ({ext}): {path}")
    try:
        y, sr = librosa.load(path, sr=target_sr, mono=True)
    except Exception as e:
        raise AudioProcessingError(f"Không thể đọc {path}: {e}") from e
    return y, sr


def normalize_length(y: np.ndarray, n_samples: int = N_SAMPLES) -> np.ndarray:
    """
    Trim silence → truncate hoặc ZERO-PAD (không tile).
    - Nếu dài hơn: lấy đoạn giữa
    - Nếu ngắn hơn: zero-pad cuối
    - Nếu toàn silence sau trim: trả về zeros + warning
    """
    # Trim silence trước (S-01)
    y_trimmed, _ = librosa.effects.trim(y, top_db=20)

    if len(y_trimmed) == 0:
        logger.warning("Audio toàn silence sau trim, trả về zeros")
        return np.zeros(n_samples, dtype=np.float32)

    if len(y_trimmed) >= n_samples:
        # Truncate: lấy đoạn giữa
        start = (len(y_trimmed) - n_samples) // 2
        return y_trimmed[start:start + n_samples]

    # Zero-pad cuối (KHÔNG tile/repeat)
    return np.pad(y_trimmed, (0, n_samples - len(y_trimmed)), mode='constant')


def normalize_amplitude(y: np.ndarray, target_db: float = -20.0) -> np.ndarray:
    """Chuẩn hóa biên độ về mức dB cố định (RMS normalization)."""
    rms = np.sqrt(np.mean(y ** 2))
    if rms < 1e-8:
        return y
    target_rms = 10 ** (target_db / 20.0)
    return y * (target_rms / rms)


def preprocess_audio_for_features(path: str, target_sr: int = SAMPLE_RATE) -> tuple[np.ndarray, int]:
    """
    Pipeline chuẩn hóa đầy đủ cho cả indexing và query.
    Đảm bảo query audio đi qua cùng pipeline như database audio.

    Returns: (y_processed, sr)
    """
    y, sr = load_audio(path, target_sr)
    y = normalize_length(y)
    y = normalize_amplitude(y)
    return y, sr


def preprocess_file(src_path: str, dst_path: str) -> str:
    """Xử lý 1 file: load → trim → pad → normalize → save."""
    y, sr = preprocess_audio_for_features(src_path)
    sf.write(dst_path, y, SAMPLE_RATE)
    return dst_path


def preprocess_all(
    raw_dir: str = RAW_DIR,
    processed_dir: str = PROCESSED_DIR,
    excluded_csv: str = EXCLUDED_CSV,
    verbose: bool = True,
) -> list[dict[str, str]]:
    """Tiền xử lý toàn bộ thư mục raw → processed, skip excluded files."""
    os.makedirs(processed_dir, exist_ok=True)

    excluded = load_excluded_filenames(excluded_csv)
    if excluded and verbose:
        print(f"  Excluded files loaded: {len(excluded)} entries")

    files_to_process: list[tuple[str, str, str, str]] = []
    for root, dirs, files in os.walk(raw_dir):
        for f in sorted(files):
            if not f.endswith('.wav'):
                continue
            # Skip excluded files (match raw filename)
            if f in excluded:
                if verbose:
                    print(f"  [SKIP] {f} (excluded)")
                continue

            species = os.path.basename(root)
            # Lưu file đã xử lý với tiền tố loài để tránh trùng lặp
            dst_fname = f"{species.lower()}_{f}"

            # Skip excluded files (match processed filename)
            if dst_fname in excluded:
                if verbose:
                    print(f"  [SKIP] {dst_fname} (excluded)")
                continue

            src_path = os.path.join(root, f)
            dst_path = os.path.join(processed_dir, dst_fname)
            files_to_process.append((src_path, dst_path, dst_fname, species.lower()))

    files_to_process.sort()

    results: list[dict[str, str]] = []
    for i, (src, dst, dst_fname, species) in enumerate(files_to_process):
        try:
            preprocess_file(src, dst)
            results.append({'filename': dst_fname, 'species': species})
        except (AudioFileNotFoundError, AudioFormatError, AudioProcessingError) as e:
            logger.error(f"  [LỖI] {dst_fname}: {e}")
            if verbose:
                print(f"  [LỖI] {dst_fname}: {e}")

        if verbose and (i + 1) % 100 == 0:
            print(f"  [{i+1}/{len(files_to_process)}] đã xử lý...")

    if verbose:
        print(f"Hoàn thành: {len(results)} files → {processed_dir}")
    return results


if __name__ == '__main__':
    print("=== Tiền xử lý dữ liệu âm thanh ===")
    records = preprocess_all(verbose=True)
    species = set(r['species'] for r in records)
    print(f"Số loài: {len(species)} | Tổng files: {len(records)}")
