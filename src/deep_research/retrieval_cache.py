"""Namespaced TTL/LRU retrieval snapshots and cancellation-isolated in-process single flight."""

import asyncio
import json
import sqlite3
import time
from dataclasses import asdict
from pathlib import Path

from deep_research.harness_store import digest, encode
from deep_research.retrieval import (
    MAX_DOCUMENT_BYTES,
    SearchHit,
    SourceDocument,
    _query_arguments,
)


class CachedRetriever:
    def __init__(self, inner, path: Path, namespace: str, *, ttl=300, force_refresh=False):
        if not namespace or not 0 < ttl <= 3600:
            raise ValueError("Cache namespace and TTL required")
        self.inner, self.path, self.namespace = inner, path, namespace
        self.ttl, self.force_refresh = ttl, force_refresh
        self.flights = {}
        self.stats = dict(hits=0, misses=0, joined=0, upstream=0, invalid=0)
        path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(path) as db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS retrieval_cache(key TEXT PRIMARY KEY,"
                "payload TEXT NOT NULL, checksum TEXT NOT NULL, expires REAL NOT NULL,"
                "touched REAL NOT NULL)"
            )

    @staticmethod
    def validate(kind, value):
        if kind == "search":
            if not isinstance(value, list) or len(value) > 10:
                raise ValueError("Invalid cached search")
            for hit in value:
                if set(hit) != {"url", "title"} or not all(
                    isinstance(x, str) for x in hit.values()
                ):
                    raise ValueError("Invalid cached hit")
        else:
            if set(value) != {"url", "title", "text", "truncated"}:
                raise ValueError("Invalid cached document")
            if (
                not all(isinstance(value[k], str) for k in ("url", "title", "text"))
                or type(value["truncated"]) is not bool
                or len(value["text"].encode()) > MAX_DOCUMENT_BYTES
            ):
                raise ValueError("Invalid cached document")
        return value

    def read(self, key, kind):
        with sqlite3.connect(self.path) as db:
            row = db.execute(
                "SELECT payload,checksum,expires FROM retrieval_cache WHERE key=?", (key,)
            ).fetchone()
            if not row or row[2] <= time.time():
                return None
            try:
                value = self.validate(kind, json.loads(row[0]))
                if digest(value) != row[1]:
                    raise ValueError
            except (ValueError, TypeError, KeyError, UnicodeError):
                self.stats["invalid"] += 1
                db.execute("DELETE FROM retrieval_cache WHERE key=?", (key,))
                return None
            db.execute("UPDATE retrieval_cache SET touched=? WHERE key=?", (time.time(), key))
            return value

    def write(self, key, kind, value):
        self.validate(kind, value)
        payload = encode(value)
        with sqlite3.connect(self.path) as db:
            now = time.time()
            db.execute(
                "INSERT OR REPLACE INTO retrieval_cache VALUES(?,?,?,?,?)",
                (key, payload, digest(value), now + self.ttl, now),
            )
            db.execute("DELETE FROM retrieval_cache WHERE expires<=?", (now,))
            while True:
                count, size = db.execute(
                    "SELECT COUNT(*),COALESCE(SUM(length(CAST(payload AS BLOB))),"
                    "0) FROM retrieval_cache"
                ).fetchone()
                if count <= 1000 and size <= 32 * 1024 * 1024:
                    break
                db.execute(
                    "DELETE FROM retrieval_cache WHERE key=(SELECT key FROM retrieval_cache "
                    "ORDER BY touched LIMIT 1)"
                )

    async def cached(self, kind, args, call):
        key = digest(dict(version=1, namespace=self.namespace, kind=kind, args=args))
        if not self.force_refresh:
            value = self.read(key, kind)
            if value is not None:
                self.stats["hits"] += 1
                return value
        if key in self.flights:
            self.stats["joined"] += 1
            return await asyncio.shield(self.flights[key])
        self.stats["misses"] += 1

        async def load():
            self.stats["upstream"] += 1
            value = await call()
            self.write(key, kind, value)
            return value

        task = asyncio.create_task(load())
        self.flights[key] = task

        def done(t):
            self.flights.pop(key, None)
            if not t.cancelled():
                t.exception()  # Consume exceptions if the last waiting caller was cancelled.

        task.add_done_callback(done)
        return await asyncio.shield(task)

    async def search(self, query, limit=3):
        query = _query_arguments(query, limit)

        async def load():
            return [asdict(x) for x in await self.inner.search(query, limit=limit)]

        value = await self.cached("search", dict(query=query, limit=limit), load)
        return [SearchHit(**x) for x in value]

    async def fetch(self, hit):
        async def load():
            result = asdict(await self.inner.fetch(hit))
            if result["url"] != hit.url:
                raise ValueError("Fetched URL differs from requested identity")
            return result

        value = await self.cached("fetch", asdict(hit), load)
        if value["url"] != hit.url:
            raise ValueError("Fetched URL differs from checkpoint identity")
        return SourceDocument(**value)

    async def close(self):
        tasks = list(self.flights.values())
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
