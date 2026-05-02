"""
Train a simple supervised classifier on the existing 58-D audio features.

This script:
1. Loads indexed feature vectors from SQLite / .npy
2. Splits data into train/val/test with stratification
3. Trains a small SVM baseline with light hyperparameter search
4. Saves the best model and evaluation reports
"""

from __future__ import annotations

import json
import os
import pickle
import sys
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

SRC_DIR = os.path.dirname(__file__)
sys.path.insert(0, SRC_DIR)

from database import DB_PATH, load_all_vectors


PROJECT_ROOT = os.path.join(SRC_DIR, "..")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
REPORT_DIR = os.path.join(PROJECT_ROOT, "reports")
MODEL_PATH = os.path.join(MODEL_DIR, "svm_classifier.pkl")
METRICS_PATH = os.path.join(REPORT_DIR, "svm_metrics.json")
CONFUSION_PNG = os.path.join(REPORT_DIR, "svm_confusion_matrix.png")
CONFUSION_CSV = os.path.join(REPORT_DIR, "svm_confusion_matrix.csv")

RANDOM_STATE = 42


def ensure_dirs() -> None:
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(REPORT_DIR, exist_ok=True)


def split_dataset(features: np.ndarray, labels: list[str], filenames: list[str]):
    indices = np.arange(len(labels))

    train_idx, temp_idx = train_test_split(
        indices,
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=labels,
    )

    temp_labels = [labels[i] for i in temp_idx]
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=0.50,
        random_state=RANDOM_STATE,
        stratify=temp_labels,
    )

    def pack(idx: np.ndarray):
        return (
            features[idx],
            [labels[i] for i in idx],
            [filenames[i] for i in idx],
        )

    return {
        "train": pack(train_idx),
        "val": pack(val_idx),
        "test": pack(test_idx),
    }


def build_candidates():
    configs = [
        {"C": 1.0, "gamma": "scale"},
        {"C": 3.0, "gamma": "scale"},
        {"C": 10.0, "gamma": "scale"},
        {"C": 3.0, "gamma": 0.01},
        {"C": 10.0, "gamma": 0.01},
    ]
    for config in configs:
        yield Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "svc",
                    SVC(
                        kernel="rbf",
                        C=config["C"],
                        gamma=config["gamma"],
                        probability=True,
                        decision_function_shape="ovr",
                    ),
                ),
            ]
        ), config


def evaluate_predictions(y_true: list[str], y_pred: list[str]) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
    }


def save_confusion_matrix(y_true: list[str], y_pred: list[str], labels: list[str]) -> None:
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    np.savetxt(CONFUSION_CSV, matrix, fmt="%d", delimiter=",", header=",".join(labels), comments="")

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(matrix, cmap="Blues")
    ax.figure.colorbar(im, ax=ax)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("SVM Confusion Matrix (Test)")

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center", color="black", fontsize=8)

    fig.tight_layout()
    plt.savefig(CONFUSION_PNG, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main():
    ensure_dirs()

    ids, filenames, labels, features = load_all_vectors(db_path=DB_PATH, scaled=False)
    if len(ids) == 0:
        raise RuntimeError("Database is empty. Run indexing before training.")

    label_list = sorted(set(labels))
    print(f"Loaded {len(ids)} samples from {DB_PATH}")
    print(f"Classes: {label_list}")
    print(f"Feature shape: {features.shape}")
    print(f"Class counts: {dict(Counter(labels))}")

    splits = split_dataset(features, labels, filenames)
    x_train, y_train, _ = splits["train"]
    x_val, y_val, _ = splits["val"]
    x_test, y_test, test_files = splits["test"]

    print("\nSplit sizes")
    print(f"Train: {len(y_train)}")
    print(f"Val  : {len(y_val)}")
    print(f"Test : {len(y_test)}")

    best_model = None
    best_config = None
    best_val_score = -1.0
    candidate_results = []

    print("\nTraining candidates")
    for model, config in build_candidates():
        model.fit(x_train, y_train)
        val_pred = model.predict(x_val)
        val_metrics = evaluate_predictions(y_val, val_pred)
        candidate_results.append({"config": config, "val": val_metrics})
        print(f"Config {config} -> val accuracy={val_metrics['accuracy']:.4f}, macro_f1={val_metrics['macro_f1']:.4f}")

        if val_metrics["macro_f1"] > best_val_score:
            best_val_score = val_metrics["macro_f1"]
            best_model = model
            best_config = config

    print(f"\nBest config: {best_config}")

    x_trainval = np.vstack([x_train, x_val])
    y_trainval = y_train + y_val

    final_model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "svc",
                SVC(
                    kernel="rbf",
                    C=best_config["C"],
                    gamma=best_config["gamma"],
                    probability=True,
                    decision_function_shape="ovr",
                ),
            ),
        ]
    )
    final_model.fit(x_trainval, y_trainval)

    test_pred = final_model.predict(x_test)
    test_metrics = evaluate_predictions(y_test, test_pred)
    print("\nTest metrics")
    print(f"Accuracy   : {test_metrics['accuracy']:.4f}")
    print(f"Macro F1   : {test_metrics['macro_f1']:.4f}")
    print(f"Weighted F1: {test_metrics['weighted_f1']:.4f}")

    save_confusion_matrix(y_test, test_pred, label_list)

    error_examples = []
    for file_name, true_label, pred_label in zip(test_files, y_test, test_pred):
        if true_label != pred_label:
            error_examples.append(
                {"filename": file_name, "true": true_label, "pred": pred_label}
            )
        if len(error_examples) >= 20:
            break

    metrics = {
        "db_path": DB_PATH,
        "n_samples": len(ids),
        "classes": label_list,
        "class_counts": dict(Counter(labels)),
        "split_sizes": {
            "train": len(y_train),
            "val": len(y_val),
            "test": len(y_test),
        },
        "best_config": best_config,
        "candidate_results": candidate_results,
        "test_metrics": test_metrics,
        "error_examples": error_examples,
        "artifacts": {
            "model_path": MODEL_PATH,
            "confusion_png": CONFUSION_PNG,
            "confusion_csv": CONFUSION_CSV,
        },
    }

    with open(MODEL_PATH, "wb") as handle:
        pickle.dump(final_model, handle)

    with open(METRICS_PATH, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, ensure_ascii=True, indent=2)

    print(f"\nSaved model   : {MODEL_PATH}")
    print(f"Saved metrics : {METRICS_PATH}")
    print(f"Saved matrix  : {CONFUSION_PNG}")


if __name__ == "__main__":
    main()
