"""
CSDLDPT - Tạo canonical artifacts từ PostgreSQL database.

Output:
  features/feature_db.npy       — ma trận (N, 310) raw vectors
  features/feature_scaler.npz   — z-score scaler (mean, std)
  features/faiss.index          — Faiss IndexFlatIP (cosine similarity)
  features/file_index.json      — metadata cho từng vector {idx: {filepath, ...}}
"""

from __future__ import annotations

import json
import os
import sys
import time

import faiss
import numpy as np
import pandas as pd

SRC_DIR = os.path.dirname(__file__)
sys.path.insert(0, SRC_DIR)

from database import load_all_vectors, save_feature_scaler, FEATURE_DIM

PROJECT_ROOT = os.path.join(SRC_DIR, '..')
FEATURES_DIR = os.path.join(PROJECT_ROOT, 'features')
METADATA_CSV = os.path.join(PROJECT_ROOT, 'data', 'metadata.csv')


def load_metadata_map(csv_path: str = METADATA_CSV) -> dict[str, dict]:
    """Load metadata.csv thành dict {filename: {file_id, filepath, ...}}."""
    meta_map: dict[str, dict] = {}
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            meta_map[row['filename']] = row.to_dict()
    return meta_map


def build_canonical(verbose: bool = True) -> None:
    """Tạo feature_db.npy + scaler + faiss.index + file_index.json."""
    t_start = time.time()
    os.makedirs(FEATURES_DIR, exist_ok=True)

    if verbose:
        print("=" * 60)
        print("  Build Canonical Artifacts")
        print("=" * 60)

    # 1. Load all vectors từ PostgreSQL (quality='kept')
    if verbose:
        print("\n[1/5] Loading vectors from database...")
    ids, filenames, species_list, matrix = load_all_vectors(scaled=False)
    n_files = matrix.shape[0]
    dim = matrix.shape[1] if n_files > 0 else FEATURE_DIM
    assert dim == FEATURE_DIM, f"Expected {FEATURE_DIM}D, got {dim}D"
    if verbose:
        print(f"  Loaded {n_files} vectors, dim={dim}")

    # 2. Save raw feature_db.npy
    if verbose:
        print("\n[2/5] Saving features/feature_db.npy...")
    db_path = os.path.join(FEATURES_DIR, 'feature_db.npy')
    np.save(db_path, matrix)
    if verbose:
        print(f"  Shape: {matrix.shape}")

    # 3. Fit z-score scaler
    if verbose:
        print("\n[3/5] Fitting z-score scaler...")
    scaler_path = os.path.join(FEATURES_DIR, 'feature_scaler.npz')
    mean, std = save_feature_scaler(matrix, scaler_path)
    if verbose:
        print(f"  Saved: {scaler_path}")

    # 4. Build Faiss index
    if verbose:
        print("\n[4/5] Building Faiss IndexFlatIP...")
    # z-score scale
    safe_std = np.where(std < 1e-8, 1.0, std)
    scaled = ((matrix - mean) / safe_std).astype(np.float32)
    # L2 normalize → Inner Product = Cosine
    faiss.normalize_L2(scaled)
    index = faiss.IndexFlatIP(FEATURE_DIM)
    index.add(scaled)
    faiss_path = os.path.join(FEATURES_DIR, 'faiss.index')
    faiss.write_index(index, faiss_path)
    if verbose:
        print(f"  Index size: {index.ntotal} vectors")
        print(f"  Saved: {faiss_path}")

    # 5. Build file_index.json
    if verbose:
        print("\n[5/5] Building features/file_index.json...")
    meta_map = load_metadata_map()
    file_index: dict[int, dict] = {}
    for i, (fname, sp) in enumerate(zip(filenames, species_list)):
        meta = meta_map.get(fname, {})
        file_index[i] = {
            "file_id": meta.get("file_id", f"{sp}_{i:04d}"),
            "filepath": meta.get("filepath", f"data/processed/{fname}"),
            "filename": fname,
            "species": sp,
            "duration_sec": float(meta.get("duration_sec", 2.0)),
            "sample_rate": int(meta.get("sample_rate", 22050)),
            "source": str(meta.get("source", "unknown")),
            "quality": "kept",
        }

    index_path = os.path.join(FEATURES_DIR, 'file_index.json')
    with open(index_path, 'w', encoding='utf-8') as f:
        json.dump(file_index, f, indent=2, ensure_ascii=False)
    if verbose:
        print(f"  Entries: {len(file_index)}")
        print(f"  Saved: {index_path}")

    elapsed = time.time() - t_start
    if verbose:
        print(f"\n{'=' * 60}")
        print(f"  DONE in {elapsed:.1f}s")
        print(f"  feature_db.npy  : {matrix.shape}")
        print(f"  faiss.index     : {index.ntotal} vectors")
        print(f"  file_index.json : {len(file_index)} entries")
        print(f"{'=' * 60}")


if __name__ == '__main__':
    build_canonical()
