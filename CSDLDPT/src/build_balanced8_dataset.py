"""
Build an 8-class dataset by splitting Chicken into Hen and Rooster.

Sources for Hen/Rooster:
  - ESC-50 local download
  - SLLM-multi-hop/AnimalQA
  - DynamicSuperbPrivate/EnvironmentalSoundClassification_ESC50-Animals_TTS
"""

from __future__ import annotations

import csv
import shutil
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BALANCED7_DIR = PROJECT_ROOT / "data" / "balanced_raw"
BALANCED8_DIR = PROJECT_ROOT / "data" / "balanced8_raw"
BALANCED8_META = PROJECT_ROOT / "data" / "balanced8_metadata.csv"

TARGET_COUNTS = {
    "Dog": 150,
    "Cat": 145,
    "Cow": 120,
    "Frog": 120,
    "Sheep": 121,
    "Monkey": 150,
    "Hen": 100,
    "Rooster": 100,
}

LICENSES = {
    "balanced7_existing": "mixed-existing",
    "esc50": "cc-by-nc-3.0",
    "animalqa": "unspecified",
    "dynamicsuperb_private": "unspecified",
}


def reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def add_metadata(rows: list[dict[str, str]], filename: str, species: str, source: str, origin_label: str, original_ref: str):
    rows.append(
        {
            "filename": filename,
            "species": species,
            "source": source,
            "origin_label": origin_label,
            "license": LICENSES[source],
            "original_ref": original_ref,
        }
    )


def copy_existing_species(species: str, source_species: str, limit: int, rows: list[dict[str, str]]) -> int:
    src_dir = BALANCED7_DIR / source_species
    dst_dir = BALANCED8_DIR / species
    dst_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    for src_path in sorted(src_dir.glob("*.wav"))[:limit]:
        dst_name = src_path.name
        shutil.copy2(src_path, dst_dir / dst_name)
        add_metadata(rows, dst_name, species, "balanced7_existing", source_species, str(src_path))
        copied += 1
    return copied


def import_esc50_label(label: str, species: str, limit: int, rows: list[dict[str, str]]) -> int:
    meta_path = hf_hub_download(
        repo_id="TigreGotico/ESC-50",
        filename="meta/esc50.csv",
        repo_type="dataset",
    )
    df = pd.read_csv(meta_path)
    subset = df[df["category"] == label].head(limit)

    dst_dir = BALANCED8_DIR / species
    dst_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for _, row in subset.iterrows():
        src_path = Path(
            hf_hub_download(
                repo_id="TigreGotico/ESC-50",
                filename=row["filename"],
                repo_type="dataset",
            )
        )
        dst_name = f"esc50_{label}_{row['filename']}"
        shutil.copy2(src_path, dst_dir / dst_name)
        add_metadata(rows, dst_name, species, "esc50", label, str(src_path))
        written += 1
    return written


def import_animalqa_label(label: str, species: str, limit: int, rows: list[dict[str, str]]) -> int:
    parquet_path = hf_hub_download(
        repo_id="SLLM-multi-hop/AnimalQA",
        filename="data/test-00000-of-00001.parquet",
        repo_type="dataset",
    )
    df = pd.read_parquet(parquet_path, columns=["file", "audio", "single_answer"])
    clean_label = df["single_answer"].str.replace(r"^\([a-d]\)\s+", "", regex=True)
    subset = df[clean_label == label].head(limit)

    dst_dir = BALANCED8_DIR / species
    dst_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for _, row in subset.iterrows():
        dst_name = f"animalqa_{label}_{Path(row['file']).name}"
        (dst_dir / dst_name).write_bytes(row["audio"]["bytes"])
        add_metadata(rows, dst_name, species, "animalqa", label, f"SLLM-multi-hop/AnimalQA:{row['file']}")
        written += 1
    return written


def import_dynamicprivate_label(label: str, species: str, limit: int, rows: list[dict[str, str]]) -> int:
    parquet_path = hf_hub_download(
        repo_id="DynamicSuperbPrivate/EnvironmentalSoundClassification_ESC50-Animals_TTS",
        filename="data/test-00000-of-00001.parquet",
        repo_type="dataset",
    )
    df = pd.read_parquet(parquet_path, columns=["file", "audio", "label"])
    subset = df[df["label"] == label].head(limit)

    dst_dir = BALANCED8_DIR / species
    dst_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for _, row in subset.iterrows():
        dst_name = f"dspvt_{label}_{Path(row['file']).name}"
        (dst_dir / dst_name).write_bytes(row["audio"]["bytes"])
        add_metadata(
            rows,
            dst_name,
            species,
            "dynamicsuperb_private",
            label,
            f"DynamicSuperbPrivate/EnvironmentalSoundClassification_ESC50-Animals_TTS:{row['file']}",
        )
        written += 1
    return written


def count_species(species: str) -> int:
    return len(list((BALANCED8_DIR / species).glob("*.wav")))


def write_metadata(rows: list[dict[str, str]]) -> None:
    with BALANCED8_META.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["filename", "species", "source", "origin_label", "license", "original_ref"],
        )
        writer.writeheader()
        writer.writerows(rows)


def build():
    reset_dir(BALANCED8_DIR)
    rows: list[dict[str, str]] = []

    for species in ["Dog", "Cat", "Cow", "Frog", "Sheep", "Monkey"]:
        copied = copy_existing_species(species, species, TARGET_COUNTS[species], rows)
        print(f"{species}: copied {copied} existing files")

    hen_written = 0
    hen_written += import_esc50_label("hen", "Hen", 40, rows)
    hen_written += import_animalqa_label("hen", "Hen", 40, rows)
    hen_written += import_dynamicprivate_label("hen", "Hen", TARGET_COUNTS["Hen"] - hen_written, rows)
    print(f"Hen: imported {hen_written}")

    rooster_written = 0
    rooster_written += import_esc50_label("rooster", "Rooster", 40, rows)
    rooster_written += import_animalqa_label("rooster", "Rooster", 40, rows)
    rooster_written += import_dynamicprivate_label("rooster", "Rooster", TARGET_COUNTS["Rooster"] - rooster_written, rows)
    print(f"Rooster: imported {rooster_written}")

    write_metadata(rows)

    print("\nBalanced8 dataset summary")
    total = 0
    for species in TARGET_COUNTS:
        count = count_species(species)
        total += count
        print(f"{species}: {count}")
    print(f"TOTAL: {total}")
    print(f"Metadata: {BALANCED8_META}")


if __name__ == "__main__":
    build()
