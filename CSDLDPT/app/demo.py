"""
CSDLDPT - Demo UI: Animal Sound Retrieval
Gradio interface với pure cosine similarity (Faiss).
Entry point: python app/demo.py
"""

from __future__ import annotations

import os
import sys
import tempfile

import gradio as gr
import librosa
import librosa.display
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Path setup
APP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(APP_DIR, '..')
SRC_DIR = os.path.join(PROJECT_ROOT, 'src')
sys.path.insert(0, SRC_DIR)

from feature import extract_all, FEATURE_DIM, SAMPLE_RATE
from preprocess import preprocess_audio_for_features
from search_engine import create_engine
from database import get_db_stats, log_search
from visualization import save_waveform, save_mel_spectrogram

FEATURES_DIR  = os.path.join(PROJECT_ROOT, 'features')
PROCESSED_DIR = os.path.join(PROJECT_ROOT, 'data', 'balanced8_processed')

# Load engine 1 lần khi start
engine = create_engine(FEATURES_DIR)

# ─────────────────────────────────────────────
# Core search function
# ─────────────────────────────────────────────

def do_search(audio_path):
    """Xử lý query audio → trả về kết quả search."""
    if audio_path is None:
        return ("⚠️ Vui lòng upload hoặc thu âm.",
                None, None, None, None, None, None, None, None, None, None, None, None)

    # 1. Preprocess + extract features
    y_query, sr = preprocess_audio_for_features(audio_path)
    query_vec = extract_all(y_query, sr)

    # 2. Search
    results = engine.search(query_vec, top_k=5)

    # 3. Log search
    if results:
        log_search(
            os.path.basename(audio_path),
            results[0]['filepath'],
            results[0]['similarity_score'],
        )

    # 4. Query visualization
    query_spec = _plot_spectrogram(y_query, sr, 'Query Spectrogram')
    query_wave = _plot_waveform(y_query, sr, 'Query Waveform')

    # 5. Result table HTML
    html = _build_results_html(results)

    # 6. Audio players + spectrograms cho top-5
    audios = []
    specs = []
    for r in results:
        fpath = os.path.join(PROJECT_ROOT, r['filepath'])
        if os.path.exists(fpath):
            audios.append(fpath)
            y_r, _ = preprocess_audio_for_features(fpath)
            specs.append(_plot_spectrogram(
                y_r, sr,
                f"#{r['rank']} {r['species']} (sim={r['similarity_score']:.4f})"
            ))
        else:
            audios.append(None)
            specs.append(None)

    # 7. Comparison chart
    comparison = _plot_comparison(y_query, results, sr)

    # Pad to 5
    while len(audios) < 5:
        audios.append(None)
    while len(specs) < 5:
        specs.append(None)

    return (html, query_wave, query_spec,
            audios[0], specs[0],
            audios[1], specs[1],
            audios[2], specs[2],
            audios[3], specs[3],
            audios[4], specs[4],
            comparison)


# ─────────────────────────────────────────────
# Plotting helpers
# ─────────────────────────────────────────────

def _plot_spectrogram(y, sr, title):
    fig, ax = plt.subplots(figsize=(8, 3))
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    librosa.display.specshow(mel_db, sr=sr, x_axis='time', y_axis='mel',
                             ax=ax, cmap='magma')
    ax.set_title(title, fontsize=11, fontweight='bold')
    fig.tight_layout()
    tmp = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
    fig.savefig(tmp.name, dpi=120, bbox_inches='tight')
    plt.close(fig)
    return tmp.name


def _plot_waveform(y, sr, title):
    fig, ax = plt.subplots(figsize=(8, 2))
    librosa.display.waveshow(y, sr=sr, ax=ax, color='#2196F3', alpha=0.8)
    ax.set_title(title, fontsize=11, fontweight='bold')
    fig.tight_layout()
    tmp = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
    fig.savefig(tmp.name, dpi=120, bbox_inches='tight')
    plt.close(fig)
    return tmp.name


def _plot_comparison(y_query, results, sr):
    """Bar chart similarity + mini waveforms."""
    labels = [f"#{r['rank']} {r['species']}" for r in results]
    scores = [r['similarity_score'] for r in results]
    colors = ['#F44336', '#FF9800', '#FFC107', '#4CAF50', '#2196F3']

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(labels[::-1], scores[::-1],
                   color=colors[:len(scores)][::-1], height=0.6)
    for bar, score in zip(bars, scores[::-1]):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                f'{score:.4f}', va='center', fontweight='bold', fontsize=10)
    ax.set_xlim(0, 1.08)
    ax.set_xlabel('Similarity Score')
    ax.set_title('Top-5 Cosine Similarity', fontweight='bold')
    fig.tight_layout()
    tmp = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
    fig.savefig(tmp.name, dpi=120, bbox_inches='tight')
    plt.close(fig)
    return tmp.name


def _build_results_html(results):
    rows = ""
    for r in results:
        bar_w = max(0, min(200, int(r['similarity_score'] * 200)))
        fname = os.path.basename(r['filepath'])
        rows += f"""
        <tr>
          <td style='padding:8px 10px;font-size:16px;font-weight:700;color:#6366f1'>#{r['rank']}</td>
          <td style='padding:8px 10px;font-family:monospace;font-size:12px'>{fname}</td>
          <td style='padding:8px 10px;font-weight:600;color:#2563eb'>{r['species']}</td>
          <td style='padding:8px 10px'>
            <div style='background:#e5e7eb;border-radius:4px;height:14px;width:200px;display:inline-block;vertical-align:middle'>
              <div style='background:linear-gradient(90deg,#6366f1,#3b82f6);height:14px;border-radius:4px;width:{bar_w}px'></div>
            </div>
            <span style='font-size:13px;font-weight:600;margin-left:6px'>{r['similarity_score']:.4f}</span>
          </td>
        </tr>"""

    return f"""
    <div style='font-family:system-ui,sans-serif'>
      <table style='border-collapse:collapse;width:100%;background:#fff;
                    border:1px solid #e5e7eb;border-radius:8px;overflow:hidden'>
        <thead>
          <tr style='background:#f8fafc'>
            <th style='padding:10px 10px;text-align:left;color:#475569'>Rank</th>
            <th style='padding:10px 10px;text-align:left;color:#475569'>File</th>
            <th style='padding:10px 10px;text-align:left;color:#475569'>Species</th>
            <th style='padding:10px 10px;text-align:left;color:#475569'>Similarity</th>
          </tr>
        </thead>
        <tbody>{rows}</tbody>
      </table>
      <p style='font-size:12px;color:#94a3b8;margin-top:8px'>
        Method: Pure Cosine Similarity (Faiss IndexFlatIP) · Vector: {FEATURE_DIM}D
      </p>
    </div>"""


# ─────────────────────────────────────────────
# Stats
# ─────────────────────────────────────────────

def get_stats_html():
    try:
        stats = get_db_stats()
        chips = "".join(
            f"<span style='display:inline-block;background:#eef2ff;color:#4f46e5;"
            f"border-radius:16px;padding:4px 12px;margin:3px;font-size:13px;"
            f"font-weight:500'>{sp} ({cnt})</span>"
            for sp, cnt in stats['species_list']
        )
        return (
            f"<div style='font-family:system-ui,sans-serif'>"
            f"<b style='font-size:18px'>{stats['total_files']}</b> files · "
            f"<b style='font-size:18px'>{stats['n_species']}</b> species · "
            f"<b style='font-size:18px'>{FEATURE_DIM}D</b> features<br><br>"
            f"{chips}</div>"
        )
    except Exception as e:
        return f"<p style='color:red'>DB error: {e}</p>"


# ─────────────────────────────────────────────
# Gradio UI
# ─────────────────────────────────────────────

DEMO_CSS = """
.gradio-container { max-width: 1200px !important; }
h1 { text-align: center; }
"""

with gr.Blocks(title="CSDLDPT — Animal Sound Retrieval") as demo:

    gr.Markdown("""
    # 🔊 Animal Sound Retrieval System
    **CSDLDPT — Cơ Sở Dữ Liệu Đa Phương Tiện**

    Upload hoặc thu âm tiếng động vật → hệ thống trả về **Top-5** file tương tự nhất trong CSDL.
    """)

    with gr.Row():
        with gr.Column(scale=1):
            audio_in = gr.Audio(
                label="🎤 Input Audio",
                type="filepath",
                sources=["upload", "microphone"],
            )
            btn = gr.Button("🔍 Search", variant="primary", size="lg")
            gr.Markdown("### 📊 Database Info")
            stats_html = gr.HTML(value=get_stats_html())

        with gr.Column(scale=2):
            result_html = gr.HTML(label="Results")
            with gr.Row():
                query_wave = gr.Image(label="Query Waveform", height=160)
                query_spec = gr.Image(label="Query Spectrogram", height=160)

    gr.Markdown("### 🎵 Top-5 Results — Audio Players & Spectrograms")

    # Top-5 result audio + spectrogram
    result_audios = []
    result_specs = []
    for i in range(5):
        with gr.Row():
            a = gr.Audio(label=f"#{i+1} Audio", type="filepath", interactive=False)
            s = gr.Image(label=f"#{i+1} Spectrogram", height=180)
            result_audios.append(a)
            result_specs.append(s)

    gr.Markdown("### 📈 Similarity Comparison")
    comparison_img = gr.Image(label="Top-5 Similarity Chart", height=300)

    # Outputs list
    outputs = [
        result_html, query_wave, query_spec,
        result_audios[0], result_specs[0],
        result_audios[1], result_specs[1],
        result_audios[2], result_specs[2],
        result_audios[3], result_specs[3],
        result_audios[4], result_specs[4],
        comparison_img,
    ]

    btn.click(do_search, inputs=[audio_in], outputs=outputs)

    gr.Markdown(f"""
    ---
    **Features:** MFCC (26) · Mel Spectrogram (256) · Chroma (24) · Spectral Centroid (2) · ZCR (2) = **{FEATURE_DIM}D**
    · **Search:** Pure Cosine Similarity via Faiss IndexFlatIP
    · **Database:** PostgreSQL (metadata) + NumPy (vectors)
    """)


if __name__ == '__main__':
    print("=" * 50)
    print("  CSDLDPT — Animal Sound Retrieval Demo")
    print(f"  Database: {engine.get_total_files()} files")
    print(f"  Species: {engine.get_species_list()}")
    print(f"  Feature dim: {FEATURE_DIM}D")
    print("=" * 50)
    demo.launch(server_name='0.0.0.0', server_port=7860, share=False,
                theme=gr.themes.Soft(), css=DEMO_CSS)
