"""Tests for 310D feature extraction."""

import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from feature import (
    extract_all, extract_mfcc, extract_mel_spectrogram,
    extract_chroma, extract_spectral_centroid, extract_zcr,
    FEATURE_NAMES, FEATURE_DIM, SAMPLE_RATE,
)


@pytest.fixture
def random_audio():
    return np.random.randn(SAMPLE_RATE * 2).astype(np.float32)


class TestVectorDimension:
    def test_vector_shape_310(self, random_audio):
        vec = extract_all(random_audio)
        assert vec.shape == (310,), f"Expected (310,), got {vec.shape}"

    def test_feature_names_count(self):
        assert len(FEATURE_NAMES) == 310, f"Expected 310 names, got {len(FEATURE_NAMES)}"

    def test_feature_dim_constant(self):
        assert FEATURE_DIM == 310


class TestSubFeatureDimensions:
    def test_mfcc_26d(self, random_audio):
        result = extract_mfcc(random_audio)
        assert result.shape == (26,), f"MFCC: expected (26,), got {result.shape}"

    def test_mel_spectrogram_256d(self, random_audio):
        result = extract_mel_spectrogram(random_audio)
        assert result.shape == (256,), f"Mel: expected (256,), got {result.shape}"

    def test_chroma_24d(self, random_audio):
        result = extract_chroma(random_audio)
        assert result.shape == (24,), f"Chroma: expected (24,), got {result.shape}"

    def test_spectral_centroid_2d(self, random_audio):
        result = extract_spectral_centroid(random_audio)
        assert result.shape == (2,), f"Centroid: expected (2,), got {result.shape}"

    def test_zcr_2d(self, random_audio):
        result = extract_zcr(random_audio)
        assert result.shape == (2,), f"ZCR: expected (2,), got {result.shape}"

    def test_total_matches(self):
        """26 + 256 + 24 + 2 + 2 = 310"""
        assert 26 + 256 + 24 + 2 + 2 == 310


class TestFeatureValues:
    def test_no_nan(self, random_audio):
        vec = extract_all(random_audio)
        assert not np.any(np.isnan(vec)), "Vector contains NaN"

    def test_no_inf(self, random_audio):
        vec = extract_all(random_audio)
        assert not np.any(np.isinf(vec)), "Vector contains Inf"

    def test_dtype_float32(self, random_audio):
        vec = extract_all(random_audio)
        assert vec.dtype == np.float32


class TestFeatureNames:
    def test_mfcc_names(self):
        mfcc_names = [n for n in FEATURE_NAMES if n.startswith('mfcc_')]
        assert len(mfcc_names) == 26

    def test_mel_names(self):
        mel_names = [n for n in FEATURE_NAMES if n.startswith('mel_')]
        assert len(mel_names) == 256

    def test_chroma_names(self):
        chroma_names = [n for n in FEATURE_NAMES if n.startswith('chroma_')]
        assert len(chroma_names) == 24

    def test_centroid_names(self):
        centroid_names = [n for n in FEATURE_NAMES if n.startswith('spectral_centroid')]
        assert len(centroid_names) == 2

    def test_zcr_names(self):
        zcr_names = [n for n in FEATURE_NAMES if n.startswith('zcr_')]
        assert len(zcr_names) == 2
