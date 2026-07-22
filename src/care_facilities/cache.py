"""A simple disk-backed cache with TTL, used to wrap external HTTP calls.

Backed by SQLite so it's safe (enough) to share across processes/threads
without pulling in an extra dependency. Not designed for high write
concurrency -- just enough to avoid refetching the same zipcode/state/
address data on every run.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Callable

from . import config

_lock = threading.Lock()
_connection: sqlite3.Connection | None = None


def _ensure_cache_dir() -> Path:
    cache_dir = config.cache_dir_path()
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def _get_connection() -> sqlite3.Connection:
    global _connection
    if _connection is not None:
        return _connection

    with _lock:
        if _connection is not None:
            return _connection

        cache_dir = _ensure_cache_dir()
        db_path = cache_dir / "cache.sqlite3"

        conn = sqlite3.connect(str(db_path), check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS cache (
                cache_key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                expires_at REAL NOT NULL
            )
            """
        )
        conn.commit()
        _connection = conn
        return _connection


def get(key: str) -> Any | None:
    """Return the cached value for `key`, or None if missing/expired."""
    conn = _get_connection()
    with _lock:
        cursor = conn.execute(
            "SELECT value, expires_at FROM cache WHERE cache_key = ?", (key,)
        )
        row = cursor.fetchone()

    if row is None:
        return None

    value_json, expires_at = row
    if expires_at < time.time():
        # Expired; best-effort cleanup, but don't fail the read if this errors.
        try:
            with _lock:
                conn.execute("DELETE FROM cache WHERE cache_key = ?", (key,))
                conn.commit()
        except sqlite3.Error:
            pass
        return None

    return json.loads(value_json)


def set(key: str, value: Any, ttl_seconds: float) -> None:
    """Store `value` under `key`, expiring after `ttl_seconds` seconds."""
    conn = _get_connection()
    expires_at = time.time() + ttl_seconds
    value_json = json.dumps(value)

    with _lock:
        conn.execute(
            """
            INSERT INTO cache (cache_key, value, expires_at)
            VALUES (?, ?, ?)
            ON CONFLICT(cache_key) DO UPDATE SET
                value = excluded.value,
                expires_at = excluded.expires_at
            """,
            (key, value_json, expires_at),
        )
        conn.commit()


def cached_call(key: str, ttl_seconds: float, fn: Callable[[], Any]) -> Any:
    """Return the cached value for `key` if present and fresh; otherwise call
    `fn()`, cache the result, and return it.

    This is the primary entry point later stages should use to wrap external
    HTTP calls (CMS API, NPPES API, Census geocoder, etc.).
    """
    cached_value = get(key)
    if cached_value is not None:
        return cached_value

    value = fn()
    set(key, value, ttl_seconds)
    return value
