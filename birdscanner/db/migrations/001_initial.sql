-- Migration 001: initial schema
CREATE TABLE IF NOT EXISTS detections (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp      DATETIME NOT NULL,
    species        TEXT NOT NULL,
    confidence     REAL NOT NULL,
    image_path     TEXT NOT NULL,
    thumbnail_path TEXT NOT NULL,
    track_id       INTEGER,
    stable_frames  INTEGER,
    duration_sec   REAL,
    uploaded_at    DATETIME
);
