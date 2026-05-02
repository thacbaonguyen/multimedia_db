"""
CSDLDPT - Module Quản Lý Cơ Sở Dữ Liệu
PostgreSQL lưu metadata + numpy binary lưu vector đặc trưng + Faiss cho search
"""

from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from datetime import datetime
from typing import Generator, Optional

import numpy as np
import psycopg2
from dotenv import load_dotenv

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

# Load .env từ project root
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": int(os.getenv("DB_PORT", "5432")),
    "dbname": os.getenv("DB_NAME", "animal_sounds"),
    "user": os.getenv("DB_USER", "csdldpt"),
    "password": os.getenv("DB_PASSWORD", "csdldpt123"),
}

FEATURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'features')
SCALER_PATH  = os.path.join(FEATURES_DIR, 'feature_scaler.npz')
SAMPLE_RATE  = 22050
FEATURE_DIM  = 310


# ─────────────────────────────────────────────
# Connection Management
# ─────────────────────────────────────────────

@contextmanager
def get_connection() -> Generator:
    """Context manager cho PostgreSQL connection."""
    conn = psycopg2.connect(**DB_CONFIG)
    try:
        yield conn
    finally:
        conn.close()


def check_connection() -> None:
    """Kiểm tra PostgreSQL sẵn sàng cho pipeline/demo."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT 1")
            cur.fetchone()


# ─────────────────────────────────────────────
# Khởi tạo CSDL
# ─────────────────────────────────────────────

def init_db() -> None:
    """Tạo tables nếu chưa có (backup cho scripts/init.sql)."""
    with get_connection() as conn:
        cur = conn.cursor()

        cur.execute("""
            CREATE TABLE IF NOT EXISTS audio_files (
                id           SERIAL PRIMARY KEY,
                file_id      TEXT NOT NULL UNIQUE,
                filename     TEXT NOT NULL UNIQUE,
                species      TEXT NOT NULL,
                filepath     TEXT NOT NULL,
                duration_sec REAL,
                sample_rate  INTEGER DEFAULT 22050,
                source       TEXT,
                quality      TEXT DEFAULT 'kept',
                indexed_at   TIMESTAMP DEFAULT NOW(),
                feat_path    TEXT
            )
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS species_stats (
                species          TEXT PRIMARY KEY,
                file_count       INTEGER,
                avg_duration_sec REAL,
                updated_at       TIMESTAMP DEFAULT NOW()
            )
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS search_log (
                id          SERIAL PRIMARY KEY,
                query_file  TEXT,
                top1_result TEXT,
                top1_score  REAL,
                searched_at TIMESTAMP DEFAULT NOW()
            )
        """)

        # Indexes
        cur.execute("CREATE INDEX IF NOT EXISTS idx_audio_species ON audio_files(species)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_audio_filename ON audio_files(filename)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_audio_file_id ON audio_files(file_id)")

        conn.commit()
    print("CSDL PostgreSQL khởi tạo thành công.")


# ─────────────────────────────────────────────
# Thêm / đọc dữ liệu
# ─────────────────────────────────────────────

def insert_record(
    file_id: str,
    filename: str,
    species: str,
    filepath: str,
    duration_sec: float,
    source: str,
    feature_vec: np.ndarray,
    quality: str = "kept",
    features_dir: str = FEATURES_DIR,
) -> None:
    """Thêm 1 file vào CSDL: metadata vào PostgreSQL + vector vào .npy"""
    os.makedirs(features_dir, exist_ok=True)

    # Lưu vector đặc trưng ra file .npy
    feat_name = filename.replace('.wav', '.npy')
    feat_path = os.path.join(features_dir, feat_name)
    np.save(feat_path, feature_vec.astype(np.float32))

    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO audio_files
                (file_id, filename, species, filepath, duration_sec,
                 sample_rate, source, quality, feat_path)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (filename) DO UPDATE SET
                file_id=EXCLUDED.file_id,
                species=EXCLUDED.species,
                filepath=EXCLUDED.filepath,
                duration_sec=EXCLUDED.duration_sec,
                sample_rate=EXCLUDED.sample_rate,
                source=EXCLUDED.source,
                quality=EXCLUDED.quality,
                feat_path=EXCLUDED.feat_path,
                indexed_at=NOW()
        """, (file_id, filename, species, filepath, duration_sec,
              SAMPLE_RATE, source, quality, feat_path))
        conn.commit()


def get_all_records() -> list[tuple]:
    """Lấy toàn bộ metadata từ CSDL (chỉ quality='kept')."""
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT id, file_id, filename, species, filepath,
                   duration_sec, source, feat_path
            FROM audio_files
            WHERE quality = 'kept'
            ORDER BY id
        """)
        return cur.fetchall()


def truncate_all() -> None:
    """Reset toàn bộ dữ liệu trong DB (dùng khi rebuild)."""
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("""
            TRUNCATE audio_files, species_stats, search_log
            RESTART IDENTITY CASCADE
        """)
        conn.commit()
    print("Đã xóa sạch dữ liệu trong DB.")


# ─────────────────────────────────────────────
# Feature vector operations
# ─────────────────────────────────────────────

def save_feature_scaler(matrix: np.ndarray, scaler_path: str = SCALER_PATH) -> tuple[np.ndarray, np.ndarray]:
    """Fit và lưu z-score scaler cho từng chiều đặc trưng."""
    os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
    if matrix.size == 0:
        mean = np.zeros(FEATURE_DIM, dtype=np.float32)
        std = np.ones(FEATURE_DIM, dtype=np.float32)
    else:
        mean = matrix.mean(axis=0).astype(np.float32)
        std = matrix.std(axis=0).astype(np.float32)
        std = np.where(std < 1e-8, 1.0, std).astype(np.float32)
    np.savez(scaler_path, mean=mean, std=std)
    return mean, std


def load_feature_scaler(scaler_path: str = SCALER_PATH) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Đọc scaler đã fit từ tập index."""
    if not os.path.exists(scaler_path):
        return None, None
    data = np.load(scaler_path)
    return data['mean'].astype(np.float32), data['std'].astype(np.float32)


def apply_feature_scaler(
    matrix: np.ndarray,
    mean: Optional[np.ndarray],
    std: Optional[np.ndarray],
) -> np.ndarray:
    """Chuẩn hóa z-score cho vector hoặc ma trận đặc trưng."""
    if mean is None or std is None:
        return matrix
    safe_std = np.where(std < 1e-8, 1.0, std)
    return (matrix - mean) / safe_std


def load_all_vectors(
    scaled: bool = False,
    scaler_path: str = SCALER_PATH,
) -> tuple[list[int], list[str], list[str], np.ndarray]:
    """
    Tải toàn bộ vector đặc trưng từ đĩa vào RAM.
    Chỉ lấy records có quality='kept'.
    Trả về: (ids, filenames, species_list, matrix NxD)
    """
    records = get_all_records()
    ids, filenames, species_list, vecs = [], [], [], []

    for row in records:
        rid, file_id, fname, sp, fpath, dur, source, feat_path = row
        if feat_path and os.path.exists(feat_path):
            vec = np.load(feat_path)
            ids.append(rid)
            filenames.append(fname)
            species_list.append(sp)
            vecs.append(vec)

    matrix = np.vstack(vecs).astype(np.float32) if vecs else np.zeros((0, FEATURE_DIM), dtype=np.float32)
    if scaled and len(vecs) > 0:
        mean, std = load_feature_scaler(scaler_path)
        if mean is None:
            mean, std = save_feature_scaler(matrix, scaler_path)
        matrix = apply_feature_scaler(matrix, mean, std).astype(np.float32)
    return ids, filenames, species_list, matrix


# ─────────────────────────────────────────────
# Statistics & Logging
# ─────────────────────────────────────────────

def update_species_stats() -> None:
    """Cập nhật bảng thống kê loài."""
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO species_stats (species, file_count, avg_duration_sec, updated_at)
            SELECT species, COUNT(*), AVG(duration_sec), NOW()
            FROM audio_files
            WHERE quality = 'kept'
            GROUP BY species
            ON CONFLICT (species) DO UPDATE SET
                file_count = EXCLUDED.file_count,
                avg_duration_sec = EXCLUDED.avg_duration_sec,
                updated_at = EXCLUDED.updated_at
        """)
        conn.commit()


def get_db_stats() -> dict:
    """Trả về thống kê tổng quan CSDL."""
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT COUNT(*), COUNT(DISTINCT species)
            FROM audio_files WHERE quality = 'kept'
        """)
        total, n_species = cur.fetchone()
        cur.execute("SELECT species, file_count FROM species_stats ORDER BY species")
        species_counts = cur.fetchall()
    return {
        'total_files' : total,
        'n_species'   : n_species,
        'species_list': species_counts,
    }


def log_search(query_file: str, top1_result: str, top1_score: float) -> None:
    """Ghi lịch sử tìm kiếm."""
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO search_log (query_file, top1_result, top1_score, searched_at)
            VALUES (%s, %s, %s, NOW())
        """, (query_file, top1_result, top1_score))
        conn.commit()


if __name__ == '__main__':
    try:
        check_connection()
        print("✓ PostgreSQL kết nối thành công")
        init_db()
        stats = get_db_stats()
        print(f"Tổng files: {stats['total_files']} | Số loài: {stats['n_species']}")
    except Exception as e:
        print(f"✗ Lỗi kết nối PostgreSQL: {e}")
        print("  Hãy chạy: cp .env.example .env && docker compose up -d")
