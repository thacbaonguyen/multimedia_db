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
from feature import N_MFCC, N_MELS, N_CHROMA

# ─────────────────────────────────────────────
# Feature weighting: cân bằng ảnh hưởng giữa các nhóm đặc trưng
# MFCC (26D) là đặc trưng phân biệt loài mạnh nhất nhưng chỉ chiếm 26/310
# Mel (256D) chiếm 82.6% vector → dominate cosine distance nếu không cân bằng
# ─────────────────────────────────────────────
FEATURE_WEIGHTS = np.concatenate([
    np.full(N_MFCC * 2, 3.0),     # MFCC 26D: ×3.0  (timbre, vocal tract)
    np.full(N_MELS * 2, 1.0),     # Mel  256D: ×1.0  (energy distribution)
    np.full(N_CHROMA * 2, 2.0),   # Chroma 24D: ×2.0 (pitch structure)
    np.full(2, 2.0),              # Centroid 2D: ×2.0 (brightness)
    np.full(2, 2.0),              # ZCR 2D: ×2.0 (voiced/unvoiced)
]).astype(np.float32)
assert len(FEATURE_WEIGHTS) == FEATURE_DIM, f"Weights mismatch: {len(FEATURE_WEIGHTS)} != {FEATURE_DIM}"

PROJECT_ROOT = os.path.join(SRC_DIR, '..')
FEATURES_DIR = os.path.join(PROJECT_ROOT, 'features')
METADATA_CSV = os.path.join(PROJECT_ROOT, 'data', 'metadata.csv')


def load_metadata_map(csv_path: str = METADATA_CSV) -> dict[str, dict]:
    """Load metadata.csv thành dict {filename: {file_id, filepath, ...}}"""
    meta_map: dict[str, dict] = {}
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            meta_map[row['filename']] = row.to_dict()
    return meta_map

# Tạo feature_db.npy + scaler + faiss.index + file_index.json
def build_canonical(verbose: bool = True) -> None:
    t_start = time.time()
    os.makedirs(FEATURES_DIR, exist_ok=True)

    # load all vectors từ PostgreSQL (quality='kept')
    ids, filenames, species_list, matrix = load_all_vectors(scaled=False)
    n_files = matrix.shape[0] # số hàng
    dim = matrix.shape[1] if n_files > 0 else FEATURE_DIM
    assert dim == FEATURE_DIM, f"Expected {FEATURE_DIM}D, got {dim}D"
    if verbose:
        print(f"  Loaded {n_files} vectors, dim={dim}")

    # save raw feature_db.npy
    db_path = os.path.join(FEATURES_DIR, 'feature_db.npy')
    np.save(db_path, matrix)

    # fit z-score scaler + save weights

    scaler_path = os.path.join(FEATURES_DIR, 'feature_scaler.npz')
    mean, std = save_feature_scaler(matrix, scaler_path, weights=FEATURE_WEIGHTS)
    if verbose:
        print(f"  Saved: {scaler_path}")
        print(f"  Weights: MFCC×3.0, Mel×1.0, Chroma×2.0, Centroid×2.0, ZCR×2.0")

    # build Faiss index
    # z-score scale → apply weights → L2 normalize → Inner Product = Cosine
    safe_std = np.where(std < 1e-8, 1.0, std)
    scaled = ((matrix - mean) / safe_std).astype(np.float32)
    scaled *= FEATURE_WEIGHTS  # <-- Feature weighting
    # L2 normalize → Inner Product = Cosine
    faiss.normalize_L2(scaled)
    index = faiss.IndexFlatIP(FEATURE_DIM)
    index.add(scaled)
    faiss_path = os.path.join(FEATURES_DIR, 'faiss.index')
    faiss.write_index(index, faiss_path)
    if verbose:
        print(f"  Index size: {index.ntotal} vectors")
        print(f"  Saved: {faiss_path}")

    # build file_index.json
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


if __name__ == '__main__':
    build_canonical()
