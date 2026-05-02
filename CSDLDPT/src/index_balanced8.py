"""
CSDLDPT - Index the balanced 8-class dataset.

Pipeline: Preprocess → Feature 310D → PostgreSQL + .npy → Metadata CSV → Canonical Artifacts
"""

from __future__ import annotations

import os
import sys
import time
from datetime import datetime

import soundfile as sf

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

SRC_DIR = os.path.dirname(__file__)
sys.path.insert(0, SRC_DIR)

from database import (
    check_connection, init_db, insert_record, truncate_all,
    update_species_stats, get_db_stats, FEATURES_DIR,
)
from feature import extract_from_file, FEATURE_DIM
from preprocess import preprocess_all, load_excluded_filenames

PROJECT_ROOT = os.path.join(SRC_DIR, '..')
RAW_DIR       = os.path.join(PROJECT_ROOT, 'data', 'balanced8_raw')
PROCESSED_DIR = os.path.join(PROJECT_ROOT, 'data', 'balanced8_processed')
EXCLUDED_CSV  = os.path.join(PROJECT_ROOT, 'data', 'excluded_files.csv')


def detect_source(filename: str) -> str:
    """Suy đoán source từ tên file."""
    name = filename.lower()
    if 'esc50' in name or any(name.startswith(f'{d}-') for d in '12345'):
        return 'esc50'
    if 'sounddino' in name:
        return 'sounddino'
    if 'animalqa' in name:
        return 'animalqa'
    if 'dynamicsuperb' in name or 'ds_' in name:
        return 'dynamicsuperb'
    if 'local' in name:
        return 'local'
    return 'unknown'


def run(verbose: bool = True):
    t_start = time.time()
    print("=" * 60)
    print("  CSDLDPT — Balanced8 Index (PostgreSQL + 310D + Faiss)")
    print("=" * 60)

    # 0. Check PostgreSQL
    print("\n[0/5] Kiểm tra PostgreSQL...")
    try:
        check_connection()
        print("  ✓ PostgreSQL OK")
    except Exception as e:
        print(f"  ✗ PostgreSQL lỗi: {e}")
        print("  → Chạy: cp .env.example .env && docker compose up -d")
        return

    # 1. Preprocess
    print("\n[1/5] Preprocess (trim + zero-pad + normalize)...")
    records = preprocess_all(RAW_DIR, PROCESSED_DIR, EXCLUDED_CSV, verbose=False)
    print(f"  ✓ {len(records)} files processed (excluded files skipped)")

    # 2. Init DB + Truncate old data
    print("\n[2/5] Reset PostgreSQL database...")
    init_db()
    truncate_all()
    print("  ✓ Tables truncated, ready for fresh insert")

    # 3. Extract features + Insert
    print(f"\n[3/5] Extract 310D features and index {len(records)} files...")
    excluded = load_excluded_filenames(EXCLUDED_CSV)
    species_counters: dict[str, int] = {}
    ok, fail = 0, 0

    for i, rec in enumerate(records):
        fname = rec["filename"]
        species = rec["species"]

        # Double-check exclude
        if fname in excluded:
            continue

        fpath = os.path.join(PROCESSED_DIR, fname)

        try:
            # Feature extraction with preprocessing pipeline
            vec = extract_from_file(fpath, preprocess=True)
            assert vec.shape == (FEATURE_DIM,), f"Expected {FEATURE_DIM}D, got {vec.shape}"

            # Duration
            info = sf.info(fpath)
            duration_sec = round(info.frames / info.samplerate, 3)

            # file_id
            species_counters[species] = species_counters.get(species, 0) + 1
            file_id = f"{species}_{species_counters[species]:04d}"

            # Source
            source = detect_source(fname)

            # Insert → PostgreSQL + .npy
            insert_record(
                file_id=file_id,
                filename=fname,
                species=species,
                filepath=f"data/processed/{fname}",  # RELATIVE path
                duration_sec=duration_sec,
                source=source,
                feature_vec=vec,
                quality="kept",
                features_dir=FEATURES_DIR,
            )
            ok += 1
        except Exception as exc:
            print(f"  [ERROR] {fname}: {exc}")
            fail += 1

        if verbose and (i + 1) % 100 == 0:
            pct = (i + 1) / len(records) * 100
            print(f"  [{i + 1:>4}/{len(records)}] {pct:.0f}% — OK: {ok}, ERR: {fail}")

    print(f"  ✓ Indexed: {ok}, errors: {fail}")

    # 4. Update species stats
    print("\n[4/5] Update species statistics...")
    update_species_stats()
    stats = get_db_stats()
    print(f"  ✓ {stats['total_files']} files, {stats['n_species']} species")
    if stats['species_list']:
        for sp, cnt in stats['species_list']:
            print(f"    {sp:<12s}: {cnt} files")

    # 5. Build metadata CSV
    print("\n[5/5] Build data/metadata.csv...")
    from build_metadata import build_metadata
    build_metadata(verbose=verbose)

    elapsed = time.time() - t_start
    print(f"\n{'=' * 60}")
    print(f"  DONE in {elapsed:.1f}s")
    print(f"  Total indexed: {ok} files")
    print(f"  Feature dim  : {FEATURE_DIM}D")
    print(f"  Database     : PostgreSQL")
    print(f"{'=' * 60}")
    print(f"\n  → Tiếp theo: python src/build_canonical.py")
    print(f"  → Sau đó  : python src/train_classifier.py")


if __name__ == "__main__":
    run()
