"""
Custom exception classes for audio processing pipeline.

Provides specific error types for:
- Missing audio files
- Invalid audio formats
- Audio processing failures
"""


class AudioFileNotFoundError(FileNotFoundError):
    """File audio không tồn tại tại đường dẫn chỉ định."""


class AudioFormatError(ValueError):
    """File không phải định dạng audio hợp lệ (.wav, .mp3, .flac, .ogg)."""


class AudioProcessingError(RuntimeError):
    """Lỗi trong quá trình xử lý audio (load, trim, normalize)."""
