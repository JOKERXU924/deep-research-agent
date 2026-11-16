import asyncio
import sqlite3

import pytest

from deep_research.retrieval import SearchHit, SourceDocument
from deep_research.retrieval_cache import CachedRetriever


class Source:
    def __init__(self):
        self.calls = 0

    async def search(self, query, limit=3):
        self.calls += 1
        await asyncio.sleep(0.01)
        return [SearchHit("https://example.org/" + query, "source")][:limit]

    async def fetch(self, hit):
        self.calls += 1
        return SourceDocument(hit.url, hit.title, "snapshot")


async def test_singleflight_shields_other_waiter_from_cancellation(tmp_path):
    source = Source()
    cache = CachedRetriever(source, tmp_path / "c.sqlite3", "namespace")
    a, b = (asyncio.create_task(cache.search("query")) for _ in range(2))
    await asyncio.sleep(0)
    a.cancel()
    with pytest.raises(asyncio.CancelledError):
        await a
    assert (await b)[0].title == "source" and source.calls == 1
    assert cache.stats["joined"] == 1
    await cache.close()


async def test_ttl_namespace_force_refresh_and_corruption(tmp_path):
    path = tmp_path / "c.sqlite3"
    source = Source()
    cache = CachedRetriever(source, path, "v1")
    await cache.search("query")
    await cache.search("query")
    assert source.calls == 1
    await CachedRetriever(source, path, "v2").search("query")
    assert source.calls == 2
    await CachedRetriever(source, path, "v1", force_refresh=True).search("query")
    assert source.calls == 3
    with sqlite3.connect(path) as db:
        db.execute("UPDATE retrieval_cache SET expires=0")
    await cache.search("query")
    assert source.calls == 4
    with sqlite3.connect(path) as db:
        db.execute("UPDATE retrieval_cache SET payload='[]'")
    await cache.search("query")
    assert source.calls == 5 and cache.stats["invalid"] == 1


async def test_errors_not_cached_and_close_joins_inflight(tmp_path):
    class Broken:
        async def search(self, query, limit=3):
            raise OSError("offline")

    cache = CachedRetriever(Broken(), tmp_path / "c.sqlite3", "broken")
    for _ in range(2):
        with pytest.raises(OSError):
            await cache.search("query")
    assert cache.stats["upstream"] == 2
    await cache.close()
