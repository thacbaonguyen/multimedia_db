
from __future__ import annotations

import json
import os
from typing import Optional

import faiss
import numpy as np


class AnimalSoundSearchEngine:
    def __init__(self, dimension: int = 310):
        self.dimension = dimension
        self.index: Optional[faiss.Index] = None
        self.file_index: dict[int, dict] = {}
        self.scaler_mean: Optional[np.ndarray] = None
        self.scaler_std: Optional[np.ndarray] = None
        self.feature_weights: Optional[np.ndarray] = None
        self.feature_db: Optional[np.ndarray] = None
        self.feature_db_norm: Optional[np.ndarray] = None
        self._loaded = False

    def load(
        self,
        index_path: str,
        file_index_path: str,
        scaler_path: str,
        feature_db_path: str | None = None,
    ) -> None:
        """Load Faiss index + file_index.json + scaler từ disk."""
        self.index = faiss.read_index(index_path)
        with open(file_index_path, 'r', encoding='utf-8') as f:
            raw = json.load(f)
            self.file_index = {int(k): v for k, v in raw.items()}
        scaler = np.load(scaler_path)
        self.scaler_mean = scaler['mean'].astype(np.float32)
        self.scaler_std = scaler['std'].astype(np.float32)
        self.feature_weights = scaler['weights'].astype(np.float32) if 'weights' in scaler else None
        if feature_db_path and os.path.exists(feature_db_path):
            self.feature_db = np.load(feature_db_path).astype(np.float32)
            self.feature_db_norm = self._prepare_database_numpy(self.feature_db)
        self._loaded = True

    def is_loaded(self) -> bool:
        return self._loaded and self.index is not None

    def _prepare_query(self, query_vector: np.ndarray) -> np.ndarray:
        assert self.scaler_mean is not None and self.scaler_std is not None
        safe_std = np.where(self.scaler_std < 1e-8, 1.0, self.scaler_std)
        q = ((query_vector - self.scaler_mean) / safe_std).reshape(1, -1).astype(np.float32)
        if self.feature_weights is not None:
            q *= self.feature_weights
        faiss.normalize_L2(q)
        return q

    def _prepare_database_numpy(self, matrix: np.ndarray) -> np.ndarray:
        """z-score scale -> apply weights -> L2 normalize bằng NumPy."""
        assert self.scaler_mean is not None and self.scaler_std is not None
        safe_std = np.where(self.scaler_std < 1e-8, 1.0, self.scaler_std)
        db = ((matrix - self.scaler_mean) / safe_std).astype(np.float32)
        if self.feature_weights is not None:
            db *= self.feature_weights

        norms = np.linalg.norm(db, axis=1, keepdims=True)
        safe_norms = np.where(norms < 1e-8, 1.0, norms)
        return (db / safe_norms).astype(np.float32)

    def _prepare_query_numpy(self, query_vector: np.ndarray) -> np.ndarray:
        """Chuẩn hóa query giống database, nhưng chỉ dùng NumPy."""
        assert self.scaler_mean is not None and self.scaler_std is not None
        safe_std = np.where(self.scaler_std < 1e-8, 1.0, self.scaler_std)
        q = ((query_vector - self.scaler_mean) / safe_std).astype(np.float32)
        if self.feature_weights is not None:
            q *= self.feature_weights

        norm = np.linalg.norm(q)
        if norm < 1e-8:
            return q
        return (q / norm).astype(np.float32)

    def _format_results(self, scores: np.ndarray, indices: np.ndarray) -> list[dict]:
        """Đổi scores/indices thành schema kết quả chung cho Faiss và NumPy."""
        results: list[dict] = []
        for rank, (score, idx) in enumerate(zip(scores, indices), 1):
            if int(idx) == -1:
                continue
            meta = self.file_index.get(int(idx), {})
            sim = round(float(np.clip(score, 0.0, 1.0)), 4)
            results.append({
                "rank": rank,
                "filepath": meta.get("filepath", "unknown"),
                "species": meta.get("species", "unknown"),
                "similarity_score": sim,
                "distance": round(1.0 - sim, 4),
            })
        return results

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> list[dict]:
        assert self.is_loaded(), "Engine chưa load. Gọi load() trước."
        assert query_vector.shape == (self.dimension,), \
            f"Expected ({self.dimension},), got {query_vector.shape}"

        q = self._prepare_query(query_vector)
        scores, indices = self.index.search(q, top_k)

        return self._format_results(scores[0], indices[0])

    def search_numpy_cosine(self, query_vector: np.ndarray, top_k: int = 5) -> list[dict]:
        """
        Search cosine thuần bằng NumPy, không dùng Faiss.

        Dùng cùng feature_db.npy + mean/std/weights như Faiss:
        database vectors và query đều được scale, weight, L2-normalize,
        sau đó tính scores = db_norm @ q.
        """
        assert self._loaded, "Engine chưa load. Gọi load() trước."
        assert self.feature_db_norm is not None, \
            "Chưa load feature_db.npy. Gọi load(..., feature_db_path=...) trước."
        assert query_vector.shape == (self.dimension,), \
            f"Expected ({self.dimension},), got {query_vector.shape}"

        q = self._prepare_query_numpy(query_vector)
        scores = self.feature_db_norm @ q
        k = min(top_k, scores.shape[0])
        if k <= 0:
            return []

        top_indices = np.argpartition(-scores, k - 1)[:k]
        top_indices = top_indices[np.argsort(-scores[top_indices])]
        return self._format_results(scores[top_indices], top_indices)

    def get_total_files(self) -> int:
        """Số lượng files trong index."""
        return self.index.ntotal if self.index else 0

    def get_species_list(self) -> list[str]:
        """Danh sách loài unique trong index."""
        return sorted(set(
            meta.get("species", "unknown")
            for meta in self.file_index.values()
        ))


def create_engine(features_dir: str = None) -> AnimalSoundSearchEngine:
    """
    Factory function: tạo và load engine từ features/ directory.
    Tiện dùng cho demo/tests.
    """
    if features_dir is None:
        features_dir = os.path.join(os.path.dirname(__file__), '..', 'features')

    engine = AnimalSoundSearchEngine()
    engine.load(
        index_path=os.path.join(features_dir, 'faiss.index'),
        file_index_path=os.path.join(features_dir, 'file_index.json'),
        scaler_path=os.path.join(features_dir, 'feature_scaler.npz'),
        feature_db_path=os.path.join(features_dir, 'feature_db.npy'),
    )
    return engine


if __name__ == '__main__':
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from feature import extract_from_file

    print("=== Test Search Engine ===")
    engine = create_engine()
    print(f"✓ Loaded: {engine.get_total_files()} files, species: {engine.get_species_list()}")

    # Test self-match: dùng file đầu tiên trong index
    if engine.file_index:
        first_meta = engine.file_index[0]
        test_file = os.path.join(os.path.dirname(__file__), '..', first_meta['filepath'])
        if os.path.exists(test_file):
            vec = extract_from_file(test_file, preprocess=True)
            results = engine.search(vec, top_k=5)
            print(f"\nQuery: {first_meta['filepath']}")
            for r in results:
                print(f"  #{r['rank']} {r['filepath']:<50s} "
                      f"species={r['species']:<8s} sim={r['similarity_score']:.4f}")
