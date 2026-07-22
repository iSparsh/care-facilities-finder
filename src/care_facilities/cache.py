"""HTTP-response cache with TTL.

Uses Redis when `REDIS_URL` is set (production on Render Key Value); otherwise
falls back to a local SQLite file under `.cache/` (local/dev). Same public
API either way: `get`, `set`, `cached_call`.

Not designed for high write concurrency -- just enough to avoid refetching
the same zipcode/state/address data on every run. Cache misses and Redis
blips degrade to a live fetch rather than crashing the pipeline.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Callable

from . import config

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_sqlite_connection: sqlite3.Connection | None = None
_redis_client: Any | None = None
_redis_init_attempted = False

# Prefix so we don't collide with other keys if the Redis instance is shared.
_REDIS_KEY_PREFIX = "care_facilities:"


def _use_redis() -> bool:
    return bool(config.REDIS_URL)


def _get_redis():
    """Lazily connect to Redis. Returns None if unset or unreachable."""
    global _redis_client, _redis_init_attempted

    if not _use_redis():
        return None

    if _redis_client is not None:
        return _redis_client

    with _lock:
        if _redis_client is not None:
            return _redis_client
        if _redis_init_attempted and _redis_client is None:
            # Previous connect failed; don't hammer Redis on every cache miss.
            # A process restart (or clearing this flag in tests) retries.
            return None
        _redis_init_attempted = True
        try:
            import redis

            client = redis.Redis.from_url(
                config.REDIS_URL,
                decode_responses=True,
                socket_connect_timeout=2,
                socket_timeout=2,
            )
            client.ping()
            _redis_client = client
            return _redis_client
        except Exception as exc:  # noqa: BLE001 - cache must never take down the app
            logger.warning("Redis unavailable (%s); cache misses until restart", exc)
            _redis_client = None
            return None


def _ensure_cache_dir() -> Path:
    cache_dir = config.cache_dir_path()
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def _get_sqlite() -> sqlite3.Connection:
    global _sqlite_connection
    if _sqlite_connection is not None:
        return _sqlite_connection

    with _lock:
        if _sqlite_connection is not None:
            return _sqlite_connection

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
        _sqlite_connection = conn
        return _sqlite_connection


def _redis_key(key: str) -> str:
    return f"{_REDIS_KEY_PREFIX}{key}"


def get(key: str) -> Any | None:
    """Return the cached value for `key`, or None if missing/expired."""
    client = _get_redis()
    if client is not None:
        try:
            raw = client.get(_redis_key(key))
            if raw is None:
                return None
            return json.loads(raw)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Redis get failed for %s (%s)", key, exc)
            return None

    conn = _get_sqlite()
    with _lock:
        cursor = conn.execute(
            "SELECT value, expires_at FROM cache WHERE cache_key = ?", (key,)
        )
        row = cursor.fetchone()

    if row is None:
        return None

    value_json, expires_at = row
    if expires_at < time.time():
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
    value_json = json.dumps(value)
    ttl = max(1, int(ttl_seconds))

    client = _get_redis()
    if client is not None:
        try:
            client.set(_redis_key(key), value_json, ex=ttl)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Redis set failed for %s (%s)", key, exc)
        return

    conn = _get_sqlite()
    expires_at = time.time() + ttl_seconds
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


def reset_clients_for_tests() -> None:
    """Drop cached clients so tests can switch backends cleanly."""
    global _redis_client, _redis_init_attempted, _sqlite_connection
    with _lock:
        if _redis_client is not None:
            try:
                _redis_client.close()
            except Exception:  # noqa: BLE001
                pass
        _redis_client = None
        _redis_init_attempted = False
        if _sqlite_connection is not None:
            try:
                _sqlite_connection.close()
            except Exception:  # noqa: BLE001
                pass
        _sqlite_connection = None
