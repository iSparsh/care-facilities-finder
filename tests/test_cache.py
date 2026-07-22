"""Tests for the disk/Redis cache (`care_facilities.cache`).

Uses a temp SQLite dir by default. Redis path is covered with a fake client
so we don't need a live Redis in CI.
"""

from __future__ import annotations

import json
import time

import pytest

from care_facilities import cache, config as config_module


class _FakeRedis:
    def __init__(self):
        self.store: dict[str, tuple[str, float | None]] = {}

    def ping(self):
        return True

    def get(self, key: str):
        entry = self.store.get(key)
        if entry is None:
            return None
        value, expires_at = entry
        if expires_at is not None and expires_at < time.time():
            del self.store[key]
            return None
        return value

    def set(self, key: str, value: str, ex: int | None = None):
        expires_at = time.time() + ex if ex is not None else None
        self.store[key] = (value, expires_at)
        return True

    def close(self):
        pass


@pytest.fixture(autouse=True)
def _isolated_sqlite(tmp_path, monkeypatch):
    monkeypatch.setattr(config_module, "CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr(config_module, "REDIS_URL", None)
    cache.reset_clients_for_tests()
    yield
    cache.reset_clients_for_tests()


def test_sqlite_round_trip():
    cache.set("k1", {"hello": "world"}, ttl_seconds=60)
    assert cache.get("k1") == {"hello": "world"}


def test_sqlite_expired_returns_none():
    cache.set("k2", [1, 2, 3], ttl_seconds=1)
    # Force expiry by writing an already-expired row via the public API
    # isn't possible; poke the DB expires_at directly.
    conn = cache._get_sqlite()
    conn.execute(
        "UPDATE cache SET expires_at = ? WHERE cache_key = ?",
        (time.time() - 10, "k2"),
    )
    conn.commit()
    assert cache.get("k2") is None


def test_cached_call_stores_and_reuses(monkeypatch):
    calls = {"n": 0}

    def _fn():
        calls["n"] += 1
        return {"n": calls["n"]}

    assert cache.cached_call("ck", 60, _fn) == {"n": 1}
    assert cache.cached_call("ck", 60, _fn) == {"n": 1}
    assert calls["n"] == 1


def test_redis_round_trip(monkeypatch):
    fake = _FakeRedis()

    def _from_url(*args, **kwargs):
        return fake

    monkeypatch.setattr(config_module, "REDIS_URL", "redis://example:6379")
    cache.reset_clients_for_tests()

    import redis as redis_mod

    monkeypatch.setattr(redis_mod.Redis, "from_url", staticmethod(_from_url))

    cache.set("rk", {"via": "redis"}, ttl_seconds=60)
    assert cache.get("rk") == {"via": "redis"}
    stored = fake.store[cache._redis_key("rk")][0]
    assert json.loads(stored) == {"via": "redis"}


def test_redis_unavailable_falls_through_to_fetch(monkeypatch):
    monkeypatch.setattr(config_module, "REDIS_URL", "redis://example:6379")
    cache.reset_clients_for_tests()

    import redis as redis_mod

    def _boom(*args, **kwargs):
        raise ConnectionError("nope")

    monkeypatch.setattr(redis_mod.Redis, "from_url", staticmethod(_boom))

    calls = {"n": 0}

    def _fn():
        calls["n"] += 1
        return "ok"

    # Connect fails -> get returns None -> fn runs. set also no-ops safely.
    assert cache.cached_call("miss", 60, _fn) == "ok"
    assert calls["n"] == 1
