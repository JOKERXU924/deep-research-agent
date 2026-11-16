import asyncio
import sqlite3

import numpy as np
import pytest

from deep_research.embedding_cache import PersistentEncoder


class Encoder:
    encoder_id = "fixture-v1"
    dimensions = 3
    kind = "semantic"

    def __init__(self):
        self.calls = []

    async def encode_documents(self, texts):
        self.calls.append(("document", texts))
        await asyncio.sleep(0)
        return np.array([[len(t), 1, 2] for t in texts], dtype=np.float32)

    async def encode_queries(self, texts):
        self.calls.append(("query", texts))
        await asyncio.sleep(0)
        return np.array([[len(t), 2, 3] for t in texts], dtype=np.float32)


async def test_persist_deduplicate_batch_and_role_isolation(tmp_path):
    base = Encoder()
    path = tmp_path / "cache.sqlite3"
    cache = PersistentEncoder(base, path, batch_size=1)
    first = await cache.encode_documents(["a", "bb", "a"])
    assert len(base.calls) == 2 and np.array_equal(first[0], first[2])
    reopened = PersistentEncoder(base, path)
    assert np.array_equal(await reopened.encode_documents(["a", "bb", "a"]), first)
    assert reopened.stats["encoded_texts"] == 0 and reopened.stats["hits"] == 2
    query = await reopened.encode_queries(["a"])
    assert not np.array_equal(query[0], first[0])
    base.encoder_id = "fixture-v2"
    other = PersistentEncoder(base, path)
    await other.encode_documents(["a"])
    assert other.stats["misses"] == 1


async def test_corrupt_row_is_reencoded_and_cache_is_bounded(tmp_path):
    base = Encoder()
    path = tmp_path / "cache.sqlite3"
    cache = PersistentEncoder(base, path, max_entries=2)
    await cache.encode_documents(["a"])
    with sqlite3.connect(path) as c:
        c.execute("update vectors set value=?", (b"bad",))
    await cache.encode_documents(["a"])
    assert cache.stats["corrupt_rows"] == 1 and cache.stats["encoded_texts"] == 2
    await cache.encode_documents(["b", "c"])
    with sqlite3.connect(path) as c:
        assert c.execute("select count(*) from vectors").fetchone()[0] <= 2


async def test_concurrent_calls_do_not_duplicate_inference(tmp_path):
    base = Encoder()
    cache = PersistentEncoder(base, tmp_path / "cache.sqlite3")
    await asyncio.gather(cache.encode_documents(["a"]), cache.encode_documents(["a"]))
    assert len(base.calls) == 1


async def test_cancel_does_not_write_an_unfinished_batch(tmp_path):
    base = Encoder()
    ready = asyncio.Event()

    async def wait(texts):
        ready.set()
        await asyncio.sleep(10)

    base.encode_documents = wait
    path = tmp_path / "cache.sqlite3"
    cache = PersistentEncoder(base, path)
    task = asyncio.create_task(cache.encode_documents(["a"]))
    await ready.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    with sqlite3.connect(path) as c:
        assert c.execute("select count(*) from vectors").fetchone()[0] == 0
