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

    print("\n[0/5] check postgresql")
    try:
        check_connection()
        print("PostgreSQL OK")
    except Exception as e:
        print(f"PostgreSQL lỗi: {e}")
        return

    # Preprocess, trim + zeropad, normalize
    records = preprocess_all(RAW_DIR, PROCESSED_DIR, EXCLUDED_CSV, verbose=False)
    print(f"  ✓ {len(records)} files processed (excluded files skipped)")

    # init truncate db
    init_db()
    truncate_all()

    # extract feature
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
            # extract file
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

    print(f"  indexed: {ok}, errors: {fail}")

    # Update species stats
    update_species_stats()
    stats = get_db_stats()
    if stats['species_list']:
        for sp, cnt in stats['species_list']:
            print(f"    {sp:<12s}: {cnt} files")

    # build metadata CSV
    from build_metadata import build_metadata
    build_metadata(verbose=verbose)

    elapsed = time.time() - t_start
    print(f"okE in {elapsed:.1f}s")


if __name__ == "__main__":
    run()
