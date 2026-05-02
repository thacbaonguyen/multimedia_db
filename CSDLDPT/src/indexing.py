"""
CSDLDPT - Pipeline Lập Chỉ Mục Toàn Bộ Dữ Liệu
Chạy script này 1 lần để xây dựng CSDL từ đầu.
"""

import os
import sys
import time
import librosa
import numpy as np

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

# Thêm src vào path
SRC_DIR = os.path.dirname(__file__)
sys.path.insert(0, SRC_DIR)

from preprocess import preprocess_all, PROCESSED_DIR, RAW_DIR
from feature    import extract_from_file
from database   import (
    init_db,
    insert_record,
    update_species_stats,
    get_db_stats,
    load_all_vectors,
    save_feature_scaler,
    DB_PATH,
    SCALER_PATH,
)


def run_indexing(raw_dir=RAW_DIR, processed_dir=PROCESSED_DIR, db_path=DB_PATH, verbose=True):
    """
    Pipeline đầy đủ:
    1. Tiền xử lý âm thanh (raw → processed)
    2. Khởi tạo CSDL
    3. Trích xuất đặc trưng + lập chỉ mục từng file
    4. Cập nhật thống kê
    """
    t_start = time.time()
    print("=" * 60)
    print("  CSDLDPT - Lập Chỉ Mục Tiếng Động Vật")
    print("=" * 60)

    # ── Bước 1: Tiền xử lý ──────────────────────────────────
    print("\n[1/4] Tiền xử lý dữ liệu âm thanh...")
    records = preprocess_all(raw_dir, processed_dir, verbose=False)
    print(f"  ✓ {len(records)} files đã được chuẩn hóa")

    # ── Bước 2: Khởi tạo CSDL ───────────────────────────────
    print("\n[2/4] Khởi tạo CSDL SQLite...")
    init_db(db_path)
    print(f"  ✓ CSDL: {db_path}")

    # ── Bước 3: Trích xuất đặc trưng & lập chỉ mục ──────────
    print(f"\n[3/4] Trích xuất đặc trưng + lập chỉ mục {len(records)} files...")
    ok, fail = 0, 0
    for i, rec in enumerate(records):
        fname   = rec['filename']
        species = rec['species']
        fpath   = os.path.join(processed_dir, fname)

        try:
            y, sr = librosa.load(fpath, sr=22050, mono=True)
            duration = len(y) / sr
            vec = extract_from_file(fpath)
            insert_record(fname, species, fpath, duration, vec, db_path)
            ok += 1
        except Exception as e:
            print(f"  [LỖI] {fname}: {e}")
            fail += 1

        if verbose and (i + 1) % 100 == 0:
            pct = (i + 1) / len(records) * 100
            print(f"  [{i+1:>3}/{len(records)}] {pct:.0f}% — OK: {ok}, Lỗi: {fail}")

    print(f"  ✓ Lập chỉ mục xong: {ok} thành công, {fail} lỗi")

    # ── Bước 4: Thống kê ────────────────────────────────────
    print("\n[4/4] Fit scaler và cập nhật thống kê loài...")
    _, _, _, db_matrix = load_all_vectors(db_path, scaled=False)
    save_feature_scaler(db_matrix, SCALER_PATH)
    print(f"  ✓ Scaler: {SCALER_PATH}")

    update_species_stats(db_path)
    stats = get_db_stats(db_path)

    elapsed = time.time() - t_start
    print("\n" + "=" * 60)
    print(f"  HOÀN THÀNH trong {elapsed:.1f}s")
    print(f"  Tổng files: {stats['total_files']}")
    print(f"  Số loài   : {stats['n_species']}")
    print("=" * 60)
    print("\n  Chi tiết từng loài:")
    for sp, cnt in stats['species_list']:
        bar = "▪" * cnt
        print(f"  {sp:<20} {cnt:>3} files  {bar}")

    return stats


if __name__ == '__main__':
    run_indexing()
