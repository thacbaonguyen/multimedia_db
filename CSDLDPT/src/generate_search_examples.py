"""
CSDLDPT - Tạo kết quả trung gian cho 2 kịch bản search.

Output → features/intermediate/
  search_scenario_1_in_db/     — query file CÓ trong CSDL
  search_scenario_2_external/  — query file KHÔNG trong CSDL
  feature_examples/            — minh họa spectrogram 2+ loài cho feature_justification.md
"""

from __future__ import annotations

import csv
import json
import os
import sys

import librosa
import numpy as np

SRC_DIR = os.path.dirname(__file__)
sys.path.insert(0, SRC_DIR)

from feature import extract_all, extract_from_file, FEATURE_DIM, SAMPLE_RATE
from preprocess import preprocess_audio_for_features
from search_engine import create_engine
from visualization import (
    save_waveform, save_mel_spectrogram, save_mfcc_heatmap,
    save_comparison, save_similarity_bar,
)

PROJECT_ROOT  = os.path.join(SRC_DIR, '..')
FEATURES_DIR  = os.path.join(PROJECT_ROOT, 'features')
INTER_DIR     = os.path.join(FEATURES_DIR, 'intermediate')
PROCESSED_DIR = os.path.join(PROJECT_ROOT, 'data', 'balanced8_processed')


def generate_feature_examples(output_dir: str, verbose: bool = True) -> None:
    """Tạo spectrogram/waveform/MFCC minh họa cho ≥ 2 loài khác nhau."""
    examples_dir = os.path.join(output_dir, 'feature_examples')
    os.makedirs(examples_dir, exist_ok=True)

    # Chọn 1 file mỗi loài cho minh họa
    target_species = ['cat', 'dog', 'frog', 'cow']
    files = sorted(os.listdir(PROCESSED_DIR))

    for sp in target_species:
        sp_files = [f for f in files if f.startswith(f'{sp}_')]
        if not sp_files:
            continue
        chosen = sp_files[0]
        path = os.path.join(PROCESSED_DIR, chosen)
        y, sr = preprocess_audio_for_features(path)

        save_waveform(y, sr, os.path.join(examples_dir, f'{sp}_waveform.png'),
                      title=f'Waveform — {sp.capitalize()}')
        save_mel_spectrogram(y, sr, os.path.join(examples_dir, f'{sp}_spectrogram.png'),
                             title=f'Mel Spectrogram — {sp.capitalize()}')
        save_mfcc_heatmap(y, sr, os.path.join(examples_dir, f'{sp}_mfcc.png'),
                          title=f'MFCC — {sp.capitalize()}')

        if verbose:
            print(f"  ✓ {sp}: waveform + spectrogram + mfcc")


def generate_scenario(
    scenario_name: str,
    query_path: str,
    output_dir: str,
    engine,
    verbose: bool = True,
) -> None:
    """Tạo đầy đủ artifacts cho 1 kịch bản search."""
    scenario_dir = os.path.join(output_dir, scenario_name)
    os.makedirs(scenario_dir, exist_ok=True)

    # 1. Preprocess + extract query
    y_query, sr = preprocess_audio_for_features(query_path)
    query_vec = extract_all(y_query, sr)
    query_filename = os.path.basename(query_path)
    query_species = query_filename.split('_')[0]

    # 2. Search
    results = engine.search(query_vec, top_k=5)

    # 3. Save query artifacts
    np.save(os.path.join(scenario_dir, 'query_vector.npy'), query_vec)
    save_waveform(y_query, sr, os.path.join(scenario_dir, 'query_waveform.png'),
                  title=f'Query: {query_filename}')
    save_mel_spectrogram(y_query, sr, os.path.join(scenario_dir, 'query_spectrogram.png'),
                         title=f'Query Spectrogram: {query_filename}')

    # 4. Query info JSON
    query_info = {
        'filename': query_filename,
        'species': query_species,
        'vector_dim': int(query_vec.shape[0]),
        'vector_sample_first_10': query_vec[:10].tolist(),
        'vector_mean': float(query_vec.mean()),
        'vector_std': float(query_vec.std()),
    }
    with open(os.path.join(scenario_dir, 'query_info.json'), 'w') as f:
        json.dump(query_info, f, indent=2)

    # 5. Ranking JSON (top-5)
    with open(os.path.join(scenario_dir, 'ranking.json'), 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # 6. Ranking full CSV (cosine vs ALL files in DB)
    feature_db = np.load(os.path.join(FEATURES_DIR, 'feature_db.npy'))
    scaler = np.load(os.path.join(FEATURES_DIR, 'feature_scaler.npz'))
    mean, std = scaler['mean'], scaler['std']
    weights = scaler['weights'] if 'weights' in scaler else np.ones(feature_db.shape[1], dtype=np.float32)
    safe_std = np.where(std < 1e-8, 1.0, std)

    # Scale + weight query and DB exactly like search_engine/build_canonical
    q_scaled = ((query_vec - mean) / safe_std).astype(np.float32)
    q_scaled *= weights
    q_norm = q_scaled / (np.linalg.norm(q_scaled) + 1e-8)
    db_scaled = ((feature_db - mean) / safe_std).astype(np.float32)
    db_scaled *= weights
    db_norms = np.linalg.norm(db_scaled, axis=1, keepdims=True)
    db_norms = np.where(db_norms < 1e-8, 1.0, db_norms)
    db_normed = db_scaled / db_norms
    cosines = db_normed @ q_norm

    with open(os.path.join(FEATURES_DIR, 'file_index.json'), 'r') as f:
        file_index = {int(k): v for k, v in json.load(f).items()}

    full_ranking_path = os.path.join(scenario_dir, 'ranking_full.csv')
    with open(full_ranking_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['rank', 'filename', 'species', 'cosine_similarity', 'distance'])
        sorted_indices = np.argsort(cosines)[::-1]
        for rank, idx in enumerate(sorted_indices, 1):
            meta = file_index.get(int(idx), {})
            cos_val = float(cosines[idx])
            sim = round(float(np.clip(cos_val, 0, 1)), 6)
            writer.writerow([
                rank,
                meta.get('filename', 'unknown'),
                meta.get('species', 'unknown'),
                round(sim, 6),
                round(1 - sim, 6),
            ])

    # 7. Result spectrograms + comparison chart
    result_ys = []
    result_labels = []
    for r in results:
        rpath = os.path.join(PROJECT_ROOT, r['filepath'])
        if os.path.exists(rpath):
            ry, _ = preprocess_audio_for_features(rpath)
            result_ys.append(ry)
            result_labels.append(f"{r['species']} ({r['similarity_score']:.4f})")
            save_mel_spectrogram(
                ry, sr,
                os.path.join(scenario_dir, f"result_{r['rank']}_spectrogram.png"),
                title=f"#{r['rank']} {r['species']} (sim={r['similarity_score']:.4f})",
            )

    if result_ys:
        save_comparison(y_query, result_ys, result_labels,
                        os.path.join(scenario_dir, 'comparison_chart.png'))

    # 8. Similarity bar chart
    save_similarity_bar(results, os.path.join(scenario_dir, 'similarity_bar.png'))

    if verbose:
        print(f"  ✓ {scenario_name}: {len(results)} results")
        print(f"    Query: {query_filename} ({query_species})")
        print(f"    Top-1: {results[0]['filepath']} (sim={results[0]['similarity_score']:.4f})")
        print(f"    Full ranking: {len(cosines)} files")


def main():
    print("=" * 60)
    print("  Generate Intermediate Results")
    print("=" * 60)

    os.makedirs(INTER_DIR, exist_ok=True)
    engine = create_engine(FEATURES_DIR)

    # 1. Feature examples (minh họa cho báo cáo)
    print("\n[1/3] Feature examples cho 4 loài...")
    generate_feature_examples(INTER_DIR)

    # 2. Scenario 1: file CÓ trong CSDL
    print("\n[2/3] Search scenario 1 — file CÓ trong CSDL...")
    # Chọn file cat đầu tiên
    db_files = sorted(f for f in os.listdir(PROCESSED_DIR)
                      if f.startswith('cat_') and f.endswith('.wav'))
    if db_files:
        query_in_db = os.path.join(PROCESSED_DIR, db_files[0])
        generate_scenario('search_scenario_1_in_db', query_in_db, INTER_DIR, engine)

    # 3. Scenario 2: file KHÔNG trong CSDL
    print("\n[3/3] Search scenario 2 — file KHÔNG trong CSDL...")
    # Dùng file dog cuối cùng, tạo bản biến đổi nhẹ để mô phỏng "external"
    dog_files = sorted(f for f in os.listdir(PROCESSED_DIR)
                       if f.startswith('dog_') and f.endswith('.wav'))
    if dog_files:
        # Lấy file cuối, thêm noise nhẹ để giả lập external
        ext_src = os.path.join(PROCESSED_DIR, dog_files[-1])
        y_ext, sr_ext = preprocess_audio_for_features(ext_src)
        # Thêm noise để tạo "file mới" không giống hệt file trong DB
        np.random.seed(42)
        y_ext_noisy = y_ext + np.random.randn(len(y_ext)).astype(np.float32) * 0.02

        # Lưu tạm
        import soundfile as sf
        ext_path = os.path.join(INTER_DIR, 'query_external_dog.wav')
        sf.write(ext_path, y_ext_noisy, sr_ext)
        generate_scenario('search_scenario_2_external', ext_path, INTER_DIR, engine)

    print(f"\n{'=' * 60}")
    print(f"  DONE — Output: {INTER_DIR}")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
