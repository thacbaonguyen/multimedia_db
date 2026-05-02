"""
CSDLDPT - Module Trực Quan Hóa Âm Thanh
Tạo hình ảnh waveform, spectrogram, MFCC heatmap, comparison chart, similarity bar.
Dùng cho báo cáo và demo UI.
"""

from __future__ import annotations

import os
from typing import Optional

import librosa
import librosa.display
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

# Consistent style
plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'font.size': 10,
})

SAMPLE_RATE = 22050
HOP_LENGTH = 512
N_FFT = 2048
N_MELS = 128
N_MFCC = 13


def save_waveform(
    y: np.ndarray,
    sr: int = SAMPLE_RATE,
    save_path: str = 'waveform.png',
    title: str = 'Waveform',
    figsize: tuple = (10, 3),
) -> str:
    """Lưu waveform plot."""
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    fig, ax = plt.subplots(figsize=figsize)
    librosa.display.waveshow(y, sr=sr, ax=ax, color='#2196F3', alpha=0.8)
    ax.set_title(title, fontweight='bold')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return save_path


def save_mel_spectrogram(
    y: np.ndarray,
    sr: int = SAMPLE_RATE,
    save_path: str = 'spectrogram.png',
    title: str = 'Mel Spectrogram',
    figsize: tuple = (10, 4),
) -> str:
    """Lưu Mel spectrogram image."""
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    fig, ax = plt.subplots(figsize=figsize)
    img = librosa.display.specshow(
        mel_db, sr=sr, hop_length=HOP_LENGTH,
        x_axis='time', y_axis='mel', ax=ax, cmap='magma',
    )
    fig.colorbar(img, ax=ax, format='%+2.0f dB', label='Power (dB)')
    ax.set_title(title, fontweight='bold')
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return save_path


def save_mfcc_heatmap(
    y: np.ndarray,
    sr: int = SAMPLE_RATE,
    save_path: str = 'mfcc.png',
    title: str = 'MFCC Heatmap',
    figsize: tuple = (10, 4),
) -> str:
    """Lưu MFCC heatmap (13 coefficients)."""
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    mfcc = librosa.feature.mfcc(
        y=y, sr=sr, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH,
    )
    fig, ax = plt.subplots(figsize=figsize)
    img = librosa.display.specshow(
        mfcc, sr=sr, hop_length=HOP_LENGTH,
        x_axis='time', ax=ax, cmap='coolwarm',
    )
    fig.colorbar(img, ax=ax, label='MFCC Value')
    ax.set_title(title, fontweight='bold')
    ax.set_ylabel('MFCC Coefficient')
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return save_path


def save_comparison(
    query_y: np.ndarray,
    result_ys: list[np.ndarray],
    labels: list[str],
    save_path: str = 'comparison.png',
    sr: int = SAMPLE_RATE,
    title: str = 'Query vs Top-5 Results',
) -> str:
    """So sánh waveform query với top-5 results."""
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    n = 1 + len(result_ys)
    fig, axes = plt.subplots(n, 1, figsize=(12, 2.5 * n), sharex=True)

    # Query
    librosa.display.waveshow(query_y, sr=sr, ax=axes[0], color='#F44336', alpha=0.8)
    axes[0].set_title('Query', fontweight='bold', color='#F44336')
    axes[0].set_ylabel('Amp')

    # Results
    colors = ['#2196F3', '#4CAF50', '#FF9800', '#9C27B0', '#00BCD4']
    for i, (ry, label) in enumerate(zip(result_ys, labels)):
        color = colors[i % len(colors)]
        librosa.display.waveshow(ry, sr=sr, ax=axes[i + 1], color=color, alpha=0.8)
        axes[i + 1].set_title(f'#{i + 1}: {label}', fontweight='bold', color=color)
        axes[i + 1].set_ylabel('Amp')

    axes[-1].set_xlabel('Time (s)')
    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.01)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return save_path


def save_similarity_bar(
    results: list[dict],
    save_path: str = 'similarity_bar.png',
    title: str = 'Top-5 Similarity Scores',
) -> str:
    """Horizontal bar chart cho similarity scores top-5."""
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    labels = [f"#{r['rank']} {r['species']}" for r in results]
    scores = [r['similarity_score'] for r in results]
    colors = ['#F44336', '#FF9800', '#FFC107', '#4CAF50', '#2196F3']

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(labels[::-1], scores[::-1], color=colors[:len(scores)][::-1], height=0.6)

    for bar, score in zip(bars, scores[::-1]):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                f'{score:.4f}', va='center', fontweight='bold')

    ax.set_xlim(0, 1.05)
    ax.set_xlabel('Similarity Score')
    ax.set_title(title, fontweight='bold')
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return save_path
