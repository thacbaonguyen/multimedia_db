"""Tests for preprocessing pipeline."""

import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from preprocess import (
    load_audio, normalize_length, normalize_amplitude,
    preprocess_audio_for_features, load_excluded_filenames,
    N_SAMPLES, SAMPLE_RATE,
)
from exceptions import AudioFileNotFoundError, AudioFormatError


FIXTURES_DIR = os.path.join(os.path.dirname(__file__), 'fixtures')


class TestLoadAudio:
    def test_file_not_found(self):
        with pytest.raises(AudioFileNotFoundError):
            load_audio('/nonexistent/path.wav')

    def test_invalid_format(self):
        txt_path = os.path.join(FIXTURES_DIR, 'not_audio.txt')
        if os.path.exists(txt_path):
            with pytest.raises(AudioFormatError):
                load_audio(txt_path)


class TestNormalizeLength:
    def test_output_shape(self):
        y = np.random.randn(44100).astype(np.float32)
        result = normalize_length(y)
        assert len(result) == N_SAMPLES

    def test_zero_pad_short_signal(self):
        """File ngắn → zero-padded, KHÔNG tile/repeat."""
        y = np.ones(1000, dtype=np.float32) * 0.5
        result = normalize_length(y)
        assert len(result) == N_SAMPLES
        # Phần pad cuối phải là zeros
        assert np.all(result[1000:] == 0.0), "Zero-pad expected, not tile"

    def test_truncate_long_signal(self):
        y = np.random.randn(88200).astype(np.float32)
        result = normalize_length(y)
        assert len(result) == N_SAMPLES

    def test_all_silence(self):
        """Audio toàn silence → trả về zeros."""
        y = np.zeros(44100, dtype=np.float32)
        result = normalize_length(y)
        assert len(result) == N_SAMPLES
        assert np.all(result == 0.0)


class TestNormalizeAmplitude:
    def test_preserves_silence(self):
        y = np.zeros(N_SAMPLES, dtype=np.float32)
        result = normalize_amplitude(y)
        assert np.all(result == 0.0)

    def test_normalizes_rms(self):
        y = np.random.randn(N_SAMPLES).astype(np.float32)
        result = normalize_amplitude(y, target_db=-20.0)
        rms = np.sqrt(np.mean(result ** 2))
        expected_rms = 10 ** (-20.0 / 20.0)
        assert abs(rms - expected_rms) < 1e-4


class TestPreprocessAudioForFeatures:
    def test_from_real_file(self):
        """Test pipeline từ file thật nếu có."""
        test_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'balanced8_processed')
        if not os.path.exists(test_dir):
            pytest.skip("No processed data available")
        files = [f for f in os.listdir(test_dir) if f.endswith('.wav')]
        if not files:
            pytest.skip("No wav files")
        path = os.path.join(test_dir, files[0])
        y, sr = preprocess_audio_for_features(path)
        assert len(y) == N_SAMPLES
        assert sr == SAMPLE_RATE


class TestLoadExcludedFilenames:
    def test_loads_excluded(self):
        csv_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'excluded_files.csv')
        if not os.path.exists(csv_path):
            pytest.skip("No excluded_files.csv")
        excluded = load_excluded_filenames(csv_path)
        assert len(excluded) > 0
        # Should contain both raw and processed variants
        assert any('sounddino' in f for f in excluded)

    def test_missing_csv(self):
        result = load_excluded_filenames('/nonexistent.csv')
        assert result == set()
