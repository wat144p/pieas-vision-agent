"""SQLite database helpers for the PIEAS Visual Intelligence Agent."""

import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "pipeline.db"


def get_connection() -> sqlite3.Connection:
    """Return a connection to the SQLite database."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db() -> None:
    """Create the images and descriptions tables if they don't exist."""
    conn = get_connection()
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS images (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            hash TEXT UNIQUE NOT NULL,
            source_url TEXT NOT NULL,
            filepath TEXT NOT NULL,
            latitude REAL,
            longitude REAL,
            scraped_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            analyzed INTEGER DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS descriptions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_hash TEXT UNIQUE NOT NULL,
            description_json TEXT NOT NULL,
            analyzed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (image_hash) REFERENCES images(hash)
        );
        """
    )
    conn.commit()
    conn.close()


def image_exists(hash_hex: str) -> bool:
    """Check if an image with this SHA-256 hash already exists in the DB."""
    conn = get_connection()
    cur = conn.execute("SELECT 1 FROM images WHERE hash = ?", (hash_hex,))
    exists = cur.fetchone() is not None
    conn.close()
    return exists


def insert_image_record(hash_hex: str, source_url: str, filepath: str,
                        latitude: float = None, longitude: float = None) -> None:
    """Insert a new image record with optional geolocation."""
    conn = get_connection()
    try:
        conn.execute(
            "INSERT INTO images (hash, source_url, filepath, latitude, longitude) "
            "VALUES (?, ?, ?, ?, ?)",
            (hash_hex, source_url, filepath, latitude, longitude),
        )
        conn.commit()
    except sqlite3.IntegrityError:
        pass
    finally:
        conn.close()


def mark_analyzed(hash_hex: str) -> None:
    """Set analyzed=1 for the given image hash."""
    conn = get_connection()
    conn.execute("UPDATE images SET analyzed = 1 WHERE hash = ?", (hash_hex,))
    conn.commit()
    conn.close()


def store_description(image_hash: str, description_json: str) -> None:
    """Insert or replace a description record."""
    conn = get_connection()
    conn.execute(
        "INSERT OR REPLACE INTO descriptions (image_hash, description_json) "
        "VALUES (?, ?)",
        (image_hash, description_json),
    )
    conn.commit()
    conn.close()


def get_unanalyzed_images() -> list:
    """Return list of (hash, filepath) for images not yet analyzed."""
    conn = get_connection()
    cur = conn.execute(
        "SELECT hash, filepath FROM images WHERE analyzed = 0"
    )
    rows = cur.fetchall()
    conn.close()
    return [(row["hash"], row["filepath"]) for row in rows]