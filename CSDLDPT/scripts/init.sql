-- CSDLDPT - Khởi tạo schema PostgreSQL
-- File này tự động chạy khi Docker volume pgdata được tạo lần đầu.
-- Reset: docker compose down -v && docker compose up -d

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
);

CREATE TABLE IF NOT EXISTS species_stats (
    species          TEXT PRIMARY KEY,
    file_count       INTEGER,
    avg_duration_sec REAL,
    updated_at       TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS search_log (
    id          SERIAL PRIMARY KEY,
    query_file  TEXT,
    top1_result TEXT,
    top1_score  REAL,
    searched_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_audio_species ON audio_files(species);
CREATE INDEX IF NOT EXISTS idx_audio_filename ON audio_files(filename);
CREATE INDEX IF NOT EXISTS idx_audio_file_id ON audio_files(file_id);
