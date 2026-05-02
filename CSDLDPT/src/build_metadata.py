"""
CSDLDPT - Tạo data/metadata.csv từ data/processed/
Schema: file_id, filename, species, filepath, duration_sec, sample_rate, source
"""

from __future__ import annotations

import csv
import os
import sys

import pandas as pd
import soundfile as sf

SRC_DIR = os.path.dirname(__file__)
sys.path.insert(0, SRC_DIR)

from preprocess import load_excluded_filenames

PROJECT_ROOT = os.path.join(SRC_DIR, '..')
PROCESSED_DIR = os.path.join(PROJECT_ROOT, 'data', 'balanced8_processed')
METADATA_CSV  = os.path.join(PROJECT_ROOT, 'data', 'metadata.csv')
BALANCED8_META = os.path.join(PROJECT_ROOT, 'data', 'balanced8_metadata.csv')
EXCLUDED_CSV   = os.path.join(PROJECT_ROOT, 'data', 'excluded_files.csv')


def load_source_map(meta_path: str = BALANCED8_META) -> dict[str, str]:
    """Load source info từ balanced8_metadata.csv nếu có."""
    source_map: dict[str, str] = {}
    if not os.path.exists(meta_path):
        return source_map
    try:
        df = pd.read_csv(meta_path)
        for _, row in df.iterrows():
            fname = str(row.get('filename', ''))
            src = str(row.get('source', 'unknown'))
            if fname:
                source_map[fname] = src
    except Exception:
        pass
    return source_map


def detect_source_from_filename(filename: str) -> str:
    """Suy đoán source từ tên file nếu metadata gốc không có."""
    name_lower = filename.lower()
    if 'esc50' in name_lower or name_lower.startswith(('1-', '2-', '3-', '4-', '5-')):
        return 'esc50'
    if 'sounddino' in name_lower:
        return 'sounddino'
    if 'animalqa' in name_lower:
        return 'animalqa'
    if 'dynamicsuperb' in name_lower or 'ds_' in name_lower:
        return 'dynamicsuperb'
    if 'local' in name_lower:
        return 'local'
    return 'unknown'


def build_metadata(
    processed_dir: str = PROCESSED_DIR,
    output_path: str = METADATA_CSV,
    excluded_csv: str = EXCLUDED_CSV,
    verbose: bool = True,
) -> pd.DataFrame:
    """Quét data/processed/, tạo metadata.csv với relative paths."""
    excluded = load_excluded_filenames(excluded_csv)
    source_map = load_source_map()

    rows: list[dict] = []
    species_counters: dict[str, int] = {}
    wav_files = sorted(f for f in os.listdir(processed_dir) if f.endswith('.wav'))

    for filename in wav_files:
        # Skip excluded
        if filename in excluded:
            continue

        filepath_abs = os.path.join(processed_dir, filename)
        # Species = prefix trước dấu _ đầu tiên
        species = filename.split('_')[0].lower()

        # file_id: species_NNNN
        species_counters[species] = species_counters.get(species, 0) + 1
        file_id = f"{species}_{species_counters[species]:04d}"

        # Duration từ soundfile
        try:
            info = sf.info(filepath_abs)
            duration_sec = round(info.frames / info.samplerate, 3)
            sample_rate = info.samplerate
        except Exception:
            duration_sec = 2.0
            sample_rate = 22050

        # Source: ưu tiên metadata gốc, fallback filename detection
        source = source_map.get(filename, detect_source_from_filename(filename))

        rows.append({
            'file_id': file_id,
            'filename': filename,
            'species': species,
            'filepath': f'data/processed/{filename}',  # RELATIVE path
            'duration_sec': duration_sec,
            'sample_rate': sample_rate,
            'source': source,
        })

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

    if verbose:
        print(f"✓ Metadata saved: {output_path}")
        print(f"  Total files: {len(df)}")
        print(f"  Species: {df['species'].nunique()}")
        print(f"  Excluded: {len(excluded)} files skipped")
        print(f"  Sources: {df['source'].value_counts().to_dict()}")

    return df


if __name__ == '__main__':
    build_metadata()
