"""
CSDLDPT - Evaluate feature-space quality for similarity retrieval.

This script keeps the task as content-based retrieval. Species labels are used
only as evaluation metadata to measure whether same-species audio tends to be
closer in the learned feature space.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from itertools import combinations
from typing import Any

import numpy as np

from feature import FEATURE_DIM, N_CHROMA, N_MFCC, N_MELS


SRC_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.join(SRC_DIR, '..')
FEATURES_DIR = os.path.join(PROJECT_ROOT, 'features')


@dataclass(frozen=True)
class FeatureSlices:
    mfcc: slice
    mel: slice
    chroma: slice
    centroid: slice
    zcr: slice


def get_feature_slices() -> FeatureSlices:
    """Return fixed slices for the current 310D feature layout."""
    mfcc_end = N_MFCC * 2
    mel_end = mfcc_end + N_MELS * 2
    chroma_end = mel_end + N_CHROMA * 2
    centroid_end = chroma_end + 2
    zcr_end = centroid_end + 2
    assert zcr_end == FEATURE_DIM, f"Feature layout mismatch: {zcr_end} != {FEATURE_DIM}"
    return FeatureSlices(
        mfcc=slice(0, mfcc_end),
        mel=slice(mfcc_end, mel_end),
        chroma=slice(mel_end, chroma_end),
        centroid=slice(chroma_end, centroid_end),
        zcr=slice(centroid_end, zcr_end),
    )


def load_artifacts(features_dir: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Load feature matrix, scaler, weights, and species labels."""
    feature_db_path = os.path.join(features_dir, 'feature_db.npy')
    scaler_path = os.path.join(features_dir, 'feature_scaler.npz')
    file_index_path = os.path.join(features_dir, 'file_index.json')

    matrix = np.load(feature_db_path).astype(np.float32)
    scaler = np.load(scaler_path)
    mean = scaler['mean'].astype(np.float32)
    std = scaler['std'].astype(np.float32)
    weights = (
        scaler['weights'].astype(np.float32)
        if 'weights' in scaler
        else np.ones(matrix.shape[1], dtype=np.float32)
    )

    with open(file_index_path, 'r', encoding='utf-8') as f:
        file_index = json.load(f)
    species = [file_index[str(i)]['species'] for i in range(len(file_index))]
    assert matrix.shape[0] == len(species), "feature_db.npy and file_index.json length mismatch"
    assert matrix.shape[1] == FEATURE_DIM, f"Expected {FEATURE_DIM}D, got {matrix.shape[1]}D"
    return matrix, mean, std, weights, species


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    """L2-normalize rows, leaving all-zero rows safe."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms < 1e-8, 1.0, norms)
    return matrix / norms


def transform_features(
    matrix: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """Apply z-score, feature weights, and L2 normalization."""
    safe_std = np.where(std < 1e-8, 1.0, std)
    scaled = ((matrix - mean) / safe_std).astype(np.float32)
    weighted = scaled * weights.astype(np.float32)
    return normalize_rows(weighted).astype(np.float32)


def precision_at_k(similarity: np.ndarray, labels: np.ndarray, top_k: int) -> tuple[float, float, dict[str, float]]:
    """Return top-1 same-species accuracy and Precision@K, excluding self matches."""
    ranking = np.argsort(-similarity, axis=1)[:, 1:top_k + 1]
    p_at_k: list[float] = []
    top1: list[bool] = []
    by_species: dict[str, list[float]] = {}

    for row_idx, neighbors in enumerate(ranking):
        same = labels[neighbors] == labels[row_idx]
        score = float(np.mean(same))
        p_at_k.append(score)
        top1.append(bool(same[0]))
        by_species.setdefault(str(labels[row_idx]), []).append(score)

    species_scores = {
        species: float(np.mean(scores))
        for species, scores in sorted(by_species.items())
    }
    return float(np.mean(top1)), float(np.mean(p_at_k)), species_scores


def overlap_pairs(similarity: np.ndarray, labels: np.ndarray, limit: int = 10) -> list[dict[str, Any]]:
    """Return inter-class species pairs with the highest average cosine similarity."""
    species = sorted(set(str(label) for label in labels))
    pairs: list[tuple[float, str, str]] = []
    for left, right in combinations(species, 2):
        pair_sim = similarity[np.ix_(labels == left, labels == right)]
        pairs.append((float(np.mean(pair_sim)), left, right))
    pairs.sort(reverse=True)
    return [
        {'left': left, 'right': right, 'mean_cosine': score}
        for score, left, right in pairs[:limit]
    ]


def evaluate_variant(
    name: str,
    matrix: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    weights: np.ndarray,
    labels: np.ndarray,
    top_k: int,
) -> dict[str, Any]:
    """Evaluate one weighting variant."""
    normalized = transform_features(matrix, mean, std, weights)
    similarity = normalized @ normalized.T

    same_class = labels[:, None] == labels[None, :]
    np.fill_diagonal(same_class, False)
    different_class = labels[:, None] != labels[None, :]

    top1, p_at_k, by_species = precision_at_k(similarity, labels, top_k)
    return {
        'name': name,
        'mean_intra_class_cosine': float(np.mean(similarity[same_class])),
        'mean_inter_class_cosine': float(np.mean(similarity[different_class])),
        'gap': float(np.mean(similarity[same_class]) - np.mean(similarity[different_class])),
        'top1_same_species_accuracy': top1,
        f'precision_at_{top_k}': p_at_k,
        f'precision_at_{top_k}_by_species': by_species,
        'highest_inter_class_pairs': overlap_pairs(similarity, labels),
    }


def build_weight_variants(current_weights: np.ndarray) -> dict[str, np.ndarray]:
    """Create small, explainable ablation variants while preserving 310D shape."""
    slices = get_feature_slices()

    variants: dict[str, np.ndarray] = {'baseline_current': current_weights.copy()}

    no_chroma = current_weights.copy()
    no_chroma[slices.chroma] = 0.0
    variants['no_chroma'] = no_chroma

    low_chroma = current_weights.copy()
    low_chroma[slices.chroma] = 0.5
    variants['low_chroma_0_5'] = low_chroma

    chroma_one = current_weights.copy()
    chroma_one[slices.chroma] = 1.0
    variants['low_chroma_1_0'] = chroma_one

    mel_075 = current_weights.copy()
    mel_075[slices.mel] = 0.75
    variants['mel_downweight_0_75'] = mel_075

    mel_05 = current_weights.copy()
    mel_05[slices.mel] = 0.5
    variants['mel_downweight_0_5'] = mel_05

    mfcc_focus = current_weights.copy()
    mfcc_focus[slices.mfcc] = 4.0
    mfcc_focus[slices.mel] = 0.75
    mfcc_focus[slices.chroma] = 1.0
    variants['mfcc_focus'] = mfcc_focus

    return variants


def format_percent(value: float) -> str:
    return f"{value * 100:.2f}%"


def print_report(results: list[dict[str, Any]], top_k: int) -> None:
    """Print a compact Markdown-friendly report."""
    print("# Feature Space Evaluation")
    print()
    print("Species labels are used only for evaluation; retrieval remains pure cosine similarity.")
    print()
    print("| Variant | Intra cosine | Inter cosine | Gap | Top-1 same species | Precision@{} |".format(top_k))
    print("|---|---:|---:|---:|---:|---:|")
    for result in results:
        print(
            f"| {result['name']} | "
            f"{result['mean_intra_class_cosine']:.4f} | "
            f"{result['mean_inter_class_cosine']:.4f} | "
            f"{result['gap']:.4f} | "
            f"{format_percent(result['top1_same_species_accuracy'])} | "
            f"{format_percent(result[f'precision_at_{top_k}'])} |"
        )

    baseline = results[0]
    print()
    print("## Precision@{} by species (baseline_current)".format(top_k))
    print()
    print("| Species | Precision@{} |".format(top_k))
    print("|---|---:|")
    for species, score in baseline[f'precision_at_{top_k}_by_species'].items():
        print(f"| {species} | {format_percent(score)} |")

    print()
    print("## Highest inter-class overlap pairs (baseline_current)")
    print()
    print("| Pair | Mean cosine |")
    print("|---|---:|")
    for pair in baseline['highest_inter_class_pairs']:
        print(f"| {pair['left']} / {pair['right']} | {pair['mean_cosine']:.4f} |")


def main() -> None:
    parser = argparse.ArgumentParser(description='Evaluate feature-space retrieval quality.')
    parser.add_argument('--features-dir', default=FEATURES_DIR, help='Directory containing canonical artifacts.')
    parser.add_argument('--top-k', type=int, default=5, help='K for Precision@K.')
    parser.add_argument('--json-output', help='Optional path to write full metrics as JSON.')
    args = parser.parse_args()

    matrix, mean, std, weights, species = load_artifacts(args.features_dir)
    labels = np.array(species)
    variants = build_weight_variants(weights)
    results = [
        evaluate_variant(name, matrix, mean, std, variant_weights, labels, args.top_k)
        for name, variant_weights in variants.items()
    ]

    print_report(results, args.top_k)

    if args.json_output:
        with open(args.json_output, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)


if __name__ == '__main__':
    main()
