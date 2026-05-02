"""
Import a curated set of SoundDino samples into the balanced8 dataset.

The goal is not bulk collection, but adding harder out-of-domain vocalization
examples for the classes that currently drift the most on external queries.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path

import librosa
import requests
import soundfile as sf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "balanced8_raw"
METADATA_PATH = PROJECT_ROOT / "data" / "balanced8_metadata.csv"
DOWNLOAD_DIR = PROJECT_ROOT / "data" / "_sounddino_tmp"

BASE_URL = "https://sounddino.com"

CURATED_ITEMS = {
    "Cow": [
        ("repeated-mooing", "/mp3/74/repeated-mooing.mp3"),
        ("village-birds-cicadas-lowing-cows", "/mp3/35/village-birds-cicadas-lowing-cows.mp3"),
        ("nature-birds-lowing-cows-in-the-pasture", "/mp3/42/nature-birds-lowing-cows-in-the-pasture.mp3"),
        ("the-lowing-of-a-cow", "/mp3/40/the-lowing-of-a-cow.mp3"),
        ("learn-to-name-a-cow", "/mp3/38/learn-to-name-a-cow.mp3"),
        ("village-birds-cows-goats-chickens-in-the-distance", "/mp3/35/village-birds-cows-goats-chickens-in-the-distance.mp3"),
        ("on-the-farm-roosters-cows-birds-in-the-background", "/mp3/31/on-the-farm-roosters-cows-birds-in-the-background.mp3"),
        ("the-sound-of-a-cow-being-milked", "/mp3/31/the-sound-of-a-cow-being-milked.mp3"),
        ("milking-a-cow-against-the-background-of-mooing", "/mp3/31/milking-a-cow-against-the-background-of-mooing.mp3"),
        ("cows-and-bulls", "/mp3/31/cows-and-bulls.mp3"),
        ("cows-on-the-farm", "/mp3/31/cows-on-the-farm.mp3"),
    ],
    "Sheep": [
        ("prolonged-bleating-of-an-old-sheep", "/mp3/45/prolonged-bleating-of-an-old-sheep.mp3"),
        ("the-sound-of-sheep-in-a-barn", "/mp3/45/the-sound-of-sheep-in-a-barn.mp3"),
        ("a-young-sheep-makes-sounds", "/mp3/45/a-young-sheep-makes-sounds.mp3"),
        ("the-bleating-of-several-sheep", "/mp3/74/the-bleating-of-several-sheep.mp3"),
        ("a-large-flock-of-sheep-is-led-to-pasture", "/mp3/74/a-large-flock-of-sheep-is-led-to-pasture.mp3"),
        ("the-bleating-of-a-flock-of-sheep", "/mp3/74/the-bleating-of-a-flock-of-sheep.mp3"),
        ("repeated-bleating-of-a-sheep", "/mp3/74/repeated-bleating-of-a-sheep.mp3"),
        ("a-whole-flock-of-sheep-in-a-barn", "/mp3/74/a-whole-flock-of-sheep-in-a-barn.mp3"),
        ("sheep-and-sheep-in-a-herd", "/mp3/43/sheep-and-sheep-in-a-herd.mp3"),
        ("young-lamb", "/mp3/18/the-sound-that-a-young-lamb-makes.mp3"),
    ],
    "Dog": [
        ("barking-dog", "/mp3/18/the-sound-of-a-barking-dog.mp3"),
        ("several-dogs-are-barking", "/mp3/18/several-dogs-are-barking.mp3"),
        ("dogs-barking-at-night", "/mp3/18/the-sound-of-dogs-barking-at-night-somewhere-in-the-private-sector.mp3"),
        ("dog-howling", "/mp3/18/dog-howling.mp3"),
        ("barking-and-growling-domestic-dog", "/mp3/18/live-sound-of-barking-and-growling-of-a-domestic-dog.mp3"),
        ("two-dogs-barking-at-a-stranger", "/mp3/18/two-dogs-barking-at-a-stranger.mp3"),
        ("dog-growling", "/mp3/18/the-sound-of-a-dog-growling.mp3"),
        ("dog-growls-viciously-3d-sound", "/mp3/18/dog-growls-viciously---3d-sound.mp3"),
        ("guard-dog-on-chain", "/mp3/18/barking-and-growling-of-a-dog-that-stands-on-a-chain-and-guards-a-private-house.mp3"),
        ("aggressive-growl-small-dog", "/mp3/18/aggressive-growl-of-a-small-dog.mp3"),
        ("roar-of-the-dog", "/mp3/18/roar-of-the-dog.mp3"),
    ],
    "Cat": [
        ("domestic-cat-asks-for-food", "/mp3/74/domestic-cat-asks-for-food.mp3"),
        ("cry-of-a-hungry-cat", "/mp3/74/cry-of-a-hungry-cat.mp3"),
        ("big-wild-cat-cry", "/mp3/74/the-cry-of-a-big-wild-cat.mp3"),
        ("wounded-cat-cry-for-help", "/mp3/74/a-pitiful-cry-for-help-from-a-wounded-cat.mp3"),
        ("angry-cat-cry", "/mp3/74/an-angry-cat-cry.mp3"),
        ("furious-cat-hiss", "/mp3/74/furious-cat-hiss.mp3"),
        ("nervous-cry-of-an-angry-cat", "/mp3/74/nervous-cry-of-an-angry-cat.mp3"),
        ("meowing-cat-in-trouble", "/mp3/74/meowing-of-a-cat-in-trouble.mp3"),
        ("angry-cat-scream-after-fight", "/mp3/74/angry-cat-scream-after-a-fight.mp3"),
        ("march-cat-screaming", "/mp3/74/march-cat-screaming.mp3"),
        ("persian-cat-meowing", "/mp3/30/persian-cat-meowing.mp3"),
        ("stressed-cat-meows-and-hisses", "/mp3/19/the-cat-is-stressed-she-meows-and-hisses.mp3"),
        ("single-cat-hissing", "/mp3/19/single-cat-hissing-sound.mp3"),
        ("cat-meows", "/mp3/84/cat-meows.mp3"),
    ],
}


def load_existing_metadata() -> list[dict[str, str]]:
    if not METADATA_PATH.exists():
        return []
    with METADATA_PATH.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_metadata(rows: list[dict[str, str]]) -> None:
    with METADATA_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["filename", "species", "source", "origin_label", "license", "original_ref"],
        )
        writer.writeheader()
        writer.writerows(rows)


def convert_mp3_to_wav(mp3_path: Path, wav_path: Path) -> None:
    y, sr = librosa.load(str(mp3_path), sr=None, mono=True)
    sf.write(str(wav_path), y, sr)


def import_curated_items() -> dict[str, int]:
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    metadata_rows = load_existing_metadata()
    existing_files = {row["filename"] for row in metadata_rows}

    imported_counts = {species: 0 for species in CURATED_ITEMS}
    failed_items: list[tuple[str, str, str]] = []

    for species, items in CURATED_ITEMS.items():
        target_dir = RAW_DIR / species
        target_dir.mkdir(parents=True, exist_ok=True)

        for slug, relative_url in items:
            file_name = f"sounddino_{slug}.wav"
            mp3_url = f"{BASE_URL}{relative_url}"
            wav_path = target_dir / file_name

            if file_name in existing_files:
                continue
            if wav_path.exists():
                metadata_rows.append(
                    {
                        "filename": file_name,
                        "species": species,
                        "source": "sounddino_curated",
                        "origin_label": species,
                        "license": "site-download",
                        "original_ref": mp3_url,
                    }
                )
                existing_files.add(file_name)
                continue

            mp3_path = DOWNLOAD_DIR / f"{slug}.mp3"

            try:
                response = requests.get(mp3_url, timeout=30)
                response.raise_for_status()
                mp3_path.write_bytes(response.content)
                convert_mp3_to_wav(mp3_path, wav_path)
                metadata_rows.append(
                    {
                        "filename": file_name,
                        "species": species,
                        "source": "sounddino_curated",
                        "origin_label": species,
                        "license": "site-download",
                        "original_ref": mp3_url,
                    }
                )
                existing_files.add(file_name)
                imported_counts[species] += 1
            except Exception as exc:
                failed_items.append((species, slug, str(exc)))

    write_metadata(metadata_rows)
    if failed_items:
        print("\nFailed SoundDino items")
        for species, slug, message in failed_items:
            print(f"{species} | {slug} | {message}")
    return imported_counts


def main() -> None:
    counts = import_curated_items()
    total = sum(counts.values())
    print(f"Imported {total} curated SoundDino files")
    for species, count in counts.items():
        print(f"{species}: {count}")


if __name__ == "__main__":
    main()
