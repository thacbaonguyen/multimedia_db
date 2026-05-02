"""
CSDLDPT - Animal sound retrieval with classifier-guided reranking.
"""

from __future__ import annotations

from functools import lru_cache
import os
import pickle
import sys

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.dirname(__file__))

from feature import extract_from_file
from database import (
    apply_feature_scaler,
    load_all_vectors,
    load_feature_scaler,
    log_search,
    DB_PATH,
)


PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "svm_classifier.pkl")

COSINE_WEIGHT = 0.75
CLASSIFIER_WEIGHT = 0.25
UNKNOWN_PROB_THRESHOLD = 0.45
UNKNOWN_SCORE_THRESHOLD = 0.55
UNKNOWN_MARGIN_THRESHOLD = 0.03


def normalize_l2(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms < 1e-8, 1.0, norms)
    return matrix / norms


def cosine_similarity_batch(query_vec: np.ndarray, db_matrix: np.ndarray) -> np.ndarray:
    q_norm = query_vec / (np.linalg.norm(query_vec) + 1e-8)
    db_norm = normalize_l2(db_matrix)
    return db_norm @ q_norm


@lru_cache(maxsize=1)
def load_classifier(model_path: str = MODEL_PATH):
    if not os.path.exists(model_path):
        return None
    with open(model_path, "rb") as handle:
        return pickle.load(handle)


def get_classifier_info(model, query_vec_raw: np.ndarray, top_n: int = 2):
    if model is None:
        return {
            "predicted_species": None,
            "candidate_species": [],
            "class_probabilities": {},
            "top_probability": None,
        }

    predicted = model.predict(query_vec_raw.reshape(1, -1))[0]
    class_probabilities = {}
    candidate_species = [predicted]
    top_probability = None

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(query_vec_raw.reshape(1, -1))[0]
        classes = list(model.named_steps["svc"].classes_)
        class_probabilities = {species: float(prob) for species, prob in zip(classes, probabilities)}
        order = np.argsort(probabilities)[::-1]
        candidate_species = [classes[idx] for idx in order[:top_n]]
        top_probability = float(probabilities[order[0]])
    else:
        candidate_species = [predicted]

    if predicted not in candidate_species:
        candidate_species = [predicted] + candidate_species

    unique_candidate_species = []
    for species in candidate_species:
        if species not in unique_candidate_species:
            unique_candidate_species.append(species)

    return {
        "predicted_species": predicted,
        "candidate_species": unique_candidate_species[:top_n],
        "class_probabilities": class_probabilities,
        "top_probability": top_probability,
    }


def normalize_cosine_scores(scores: np.ndarray) -> np.ndarray:
    return np.clip((scores + 1.0) / 2.0, 0.0, 1.0)


def assess_unknown(top_probability: float | None, results: list[dict]) -> tuple[bool, str]:
    if not results:
        return True, "no_results"

    top_score = results[0]["hybrid_score"]
    second_score = results[1]["hybrid_score"] if len(results) > 1 else results[0]["hybrid_score"]
    margin = top_score - second_score

    low_prob = top_probability is not None and top_probability < UNKNOWN_PROB_THRESHOLD
    low_score = top_score < UNKNOWN_SCORE_THRESHOLD
    small_margin = len(results) > 1 and margin < UNKNOWN_MARGIN_THRESHOLD

    reasons = []
    if low_prob:
        reasons.append("low_classifier_confidence")
    if low_score:
        reasons.append("low_hybrid_score")
    if small_margin and low_prob:
        reasons.append("small_top_margin")

    if low_score or low_prob:
        return True, ",".join(reasons) if reasons else "low_confidence"
    return False, "confident"


def search(
    query_path: str,
    top_k: int = 5,
    db_path: str = DB_PATH,
    verbose: bool = True,
    exclude_self: bool = False,
    log_result: bool = True,
    classifier_guided: bool = True,
    candidate_top_classes: int = 2,
):
    if verbose:
        print(f"\n[SEARCH] Query: {os.path.basename(query_path)}")
        print("  Step 1: extract query features")

    query_vec_raw = extract_from_file(query_path)
    if verbose:
        print(f"  -> Vector shape: {query_vec_raw.shape}, norm: {np.linalg.norm(query_vec_raw):.4f}")

    if verbose:
        print("  Step 2: load indexed vectors")
    ids, filenames, species_list, db_matrix = load_all_vectors(db_path, scaled=True)
    if len(ids) == 0:
        print("  [ERROR] Database is empty. Run indexing first.")
        return []

    mean, std = load_feature_scaler()
    query_vec = apply_feature_scaler(query_vec_raw, mean, std)
    if verbose:
        print(f"  -> Database: {len(ids)} files, matrix shape: {db_matrix.shape}")

    candidate_indices = np.arange(len(ids))
    classifier_info = {
        "predicted_species": None,
        "candidate_species": [],
        "class_probabilities": {},
        "top_probability": None,
    }

    if classifier_guided:
        classifier = load_classifier()
        classifier_info = get_classifier_info(classifier, query_vec_raw, top_n=candidate_top_classes)
        candidate_species = classifier_info["candidate_species"]
        if candidate_species:
            candidate_indices = np.array(
                [idx for idx, species in enumerate(species_list) if species in candidate_species],
                dtype=int,
            )
        if verbose:
            if classifier_info["predicted_species"] is not None:
                print(f"  Step 3: classifier prediction -> {classifier_info['predicted_species']}")
                print(f"  -> Candidate classes: {candidate_species}")
                print(f"  -> Candidate files: {len(candidate_indices)}")
                if classifier_info["top_probability"] is not None:
                    print(f"  -> Top class probability: {classifier_info['top_probability']:.4f}")
            else:
                print("  Step 3: no classifier found, search full database")
    elif verbose:
        print("  Step 3: classifier guidance disabled")

    if verbose:
        print("  Step 4: cosine similarity on candidate set")
    candidate_matrix = db_matrix[candidate_indices]
    cosine_scores = cosine_similarity_batch(query_vec, candidate_matrix)
    cosine_scores_01 = normalize_cosine_scores(cosine_scores)
    if verbose:
        print(
            f"  -> Cosine min: {cosine_scores.min():.4f} | "
            f"max: {cosine_scores.max():.4f} | mean: {cosine_scores.mean():.4f}"
        )

    class_probabilities = classifier_info["class_probabilities"]
    classifier_scores = np.array(
        [class_probabilities.get(species_list[idx], 0.0) for idx in candidate_indices],
        dtype=np.float32,
    )
    hybrid_scores = COSINE_WEIGHT * cosine_scores_01 + CLASSIFIER_WEIGHT * classifier_scores

    if verbose:
        print("  Step 5: hybrid reranking (cosine + classifier)")
        print(
            f"  -> Weights: cosine={COSINE_WEIGHT:.2f}, "
            f"classifier={CLASSIFIER_WEIGHT:.2f}"
        )

    sorted_local_idx = np.argsort(hybrid_scores)[::-1]
    sorted_idx = [int(candidate_indices[idx]) for idx in sorted_local_idx]
    if exclude_self:
        query_name = os.path.basename(query_path)
        sorted_idx = [idx for idx in sorted_idx if filenames[idx] != query_name]
    sorted_idx = sorted_idx[:top_k]

    candidate_lookup = {int(global_idx): int(local_idx) for local_idx, global_idx in enumerate(candidate_indices)}

    results = []
    for rank, idx in enumerate(sorted_idx, 1):
        local_idx = candidate_lookup[idx]
        results.append(
            {
                "rank": rank,
                "id": ids[idx],
                "filename": filenames[idx],
                "species": species_list[idx],
                "score": float(cosine_scores[local_idx]),
                "cosine_score": float(cosine_scores_01[local_idx]),
                "classifier_score": float(classifier_scores[local_idx]),
                "hybrid_score": float(hybrid_scores[local_idx]),
                "predicted_species": classifier_info["predicted_species"],
                "candidate_species": classifier_info["candidate_species"],
                "top_probability": classifier_info["top_probability"],
            }
        )

    is_unknown, unknown_reason = assess_unknown(classifier_info["top_probability"], results)
    for row in results:
        row["is_unknown"] = is_unknown
        row["unknown_reason"] = unknown_reason

    if results and log_result:
        log_search(os.path.basename(query_path), results[0]["filename"], results[0]["hybrid_score"], db_path)

    if verbose:
        print(f"\n  {'Rank':<6} {'File':<40} {'Species':<12} {'Hybrid'}")
        print("  " + "-" * 86)
        for row in results:
            bar = "#" * max(0, int(row["hybrid_score"] * 20))
            print(
                f"  #{row['rank']:<5} {row['filename']:<40} "
                f"{row['species']:<12} {row['hybrid_score']:.4f} {bar}"
            )
        if classifier_info["predicted_species"] is not None:
            print(f"\n  Classifier prediction: {classifier_info['predicted_species']}")
            print(f"  Candidate classes: {', '.join(classifier_info['candidate_species'])}")
            if classifier_info["top_probability"] is not None:
                print(f"  Top class probability: {classifier_info['top_probability']:.4f}")
        print(f"  Unknown flag: {is_unknown} ({unknown_reason})")

    return results


def search_and_evaluate(query_path: str, true_species: str, top_k: int = 5, db_path: str = DB_PATH):
    results = search(
        query_path,
        top_k=top_k,
        db_path=db_path,
        verbose=False,
        exclude_self=True,
        log_result=False,
    )
    if not results:
        return results, 0.0, False

    hits = sum(1 for row in results if row["species"] == true_species)
    precision_at_k = hits / top_k
    hit_at_1 = results[0]["species"] == true_species
    return results, precision_at_k, hit_at_1


if __name__ == "__main__":
    import glob

    processed_dir = os.path.join(os.path.dirname(__file__), "..", "data", "balanced8_processed")
    files = sorted(glob.glob(os.path.join(processed_dir, "*.wav")))
    if files:
        search(files[5], top_k=5, verbose=True, exclude_self=True, log_result=False)
