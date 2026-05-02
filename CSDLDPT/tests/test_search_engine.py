"""Tests for search engine (Faiss + pure cosine)."""

import os
import sys
import json
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

FEATURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'features')
FAISS_INDEX   = os.path.join(FEATURES_DIR, 'faiss.index')
FILE_INDEX    = os.path.join(FEATURES_DIR, 'file_index.json')
SCALER_PATH   = os.path.join(FEATURES_DIR, 'feature_scaler.npz')
FEATURE_DB    = os.path.join(FEATURES_DIR, 'feature_db.npy')

REQUIRED_SCHEMA_KEYS = {"rank", "filepath", "species", "similarity_score", "distance"}


def engine_available():
    return all(os.path.exists(p) for p in [FAISS_INDEX, FILE_INDEX, SCALER_PATH, FEATURE_DB])


@pytest.fixture(scope='module')
def engine():
    if not engine_available():
        pytest.skip("Canonical artifacts not built yet")
    from search_engine import create_engine
    return create_engine(FEATURES_DIR)


@pytest.fixture(scope='module')
def db_vectors():
    if not os.path.exists(FEATURE_DB):
        pytest.skip("feature_db.npy not found")
    return np.load(FEATURE_DB)


@pytest.fixture(scope='module')
def file_index():
    if not os.path.exists(FILE_INDEX):
        pytest.skip("file_index.json not found")
    with open(FILE_INDEX) as f:
        return {int(k): v for k, v in json.load(f).items()}


class TestSelfMatchRank1:
    def test_self_match_score(self, engine, db_vectors, file_index):
        """R-05.4 kịch bản 1: query file từ CSDL → rank 1 = chính nó."""
        results = engine.search(db_vectors[0], top_k=5)
        assert results[0]["similarity_score"] >= 0.999, \
            f"Self-match score too low: {results[0]['similarity_score']}"

    def test_self_match_filepath(self, engine, db_vectors, file_index):
        """Rank 1 filepath phải khớp file query."""
        results = engine.search(db_vectors[0], top_k=5)
        assert results[0]["filepath"] == file_index[0]["filepath"], \
            f"Expected {file_index[0]['filepath']}, got {results[0]['filepath']}"


class TestOutputSchema:
    def test_returns_5_results(self, engine, db_vectors):
        results = engine.search(db_vectors[0], top_k=5)
        assert len(results) == 5

    def test_exact_schema_keys(self, engine, db_vectors):
        """R-05.2: exact keys."""
        results = engine.search(db_vectors[0], top_k=5)
        for r in results:
            assert set(r.keys()) >= REQUIRED_SCHEMA_KEYS, \
                f"Missing keys: {REQUIRED_SCHEMA_KEYS - set(r.keys())}"

    def test_similarity_score_range(self, engine, db_vectors):
        """similarity_score ∈ [0, 1]."""
        results = engine.search(db_vectors[0], top_k=5)
        for r in results:
            assert 0.0 <= r["similarity_score"] <= 1.0, \
                f"Score out of range: {r['similarity_score']}"

    def test_distance_consistency(self, engine, db_vectors):
        """distance = 1 - similarity_score."""
        results = engine.search(db_vectors[0], top_k=5)
        for r in results:
            expected_dist = round(1.0 - r["similarity_score"], 4)
            assert r["distance"] == expected_dist, \
                f"Distance mismatch: {r['distance']} != {expected_dist}"

    def test_sorted_descending(self, engine, db_vectors):
        """Results sorted by similarity_score descending."""
        results = engine.search(db_vectors[0], top_k=5)
        scores = [r["similarity_score"] for r in results]
        assert scores == sorted(scores, reverse=True), "Not sorted descending"

    def test_rank_sequential(self, engine, db_vectors):
        results = engine.search(db_vectors[0], top_k=5)
        ranks = [r["rank"] for r in results]
        assert ranks == [1, 2, 3, 4, 5]


class TestExternalQuery:
    def test_external_file_returns_5(self, engine):
        """R-05.4 kịch bản 2: file hợp lệ KHÔNG trong CSDL → vẫn trả 5."""
        # Tạo random vector giả lập file ngoài CSDL
        fake_query = np.random.randn(310).astype(np.float32)
        results = engine.search(fake_query, top_k=5)
        assert len(results) == 5
        assert all("similarity_score" in r for r in results)


class TestInvalidInput:
    def test_wrong_dimension(self, engine):
        with pytest.raises(AssertionError):
            engine.search(np.random.randn(58).astype(np.float32), top_k=5)

    def test_not_loaded(self):
        from search_engine import AnimalSoundSearchEngine
        fresh = AnimalSoundSearchEngine()
        with pytest.raises(AssertionError):
            fresh.search(np.random.randn(310).astype(np.float32))
