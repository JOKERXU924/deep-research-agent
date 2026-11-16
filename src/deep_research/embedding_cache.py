"""Bounded persistent float32 cache; no raw texts, pickle, or cross-role reuse."""

import asyncio
import hashlib
import json
import sqlite3
import time

import numpy as np

from deep_research.embeddings import _texts
from deep_research.errors import ResearchError


class PersistentEncoder:
    def __init__(self, encoder, path, *, batch_size=64, max_entries=20000):
        if not 1 <= batch_size <= 64 or not 1 <= max_entries <= 20000:
            raise ValueError("Invalid embedding cache bounds")
        self.encoder, self.path = encoder, path
        self.encoder_id, self.dimensions, self.kind = (
            encoder.encoder_id,
            encoder.dimensions,
            encoder.kind,
        )
        if not 1 <= self.dimensions <= 4096:
            raise ValueError("Invalid embedding dimensions")
        self.namespace = hashlib.sha256(
            json.dumps(["float32-v1", self.encoder_id, self.dimensions, self.kind]).encode()
        ).hexdigest()
        self.batch_size, self.max_entries, self.lock = batch_size, max_entries, asyncio.Lock()
        self.stats = dict(hits=0, misses=0, encoded_texts=0, batches=0, corrupt_rows=0)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with sqlite3.connect(path) as c:
                c.execute(
                    "CREATE TABLE IF NOT EXISTS vectors(namespace TEXT, role TEXT, digest TEXT, "
                    "value BLOB, checksum TEXT, touched REAL, PRIMARY KEY(namespace,role,digest))"
                )
        except sqlite3.Error:
            raise ResearchError("embedding_cache", "向量缓存无法初始化。") from None

    def validate(self, value, count):
        raw = np.asarray(value)
        if (
            raw.shape != (count, self.dimensions)
            or not np.issubdtype(raw.dtype, np.floating)
            or not np.isfinite(raw).all()
            or not np.isfinite(np.linalg.norm(raw, axis=1)).all()
            or (np.linalg.norm(raw, axis=1) == 0).any()
        ):
            raise ResearchError("embedding_output", "缓存编码器返回无效向量。")
        return raw.astype("<f4")

    async def encode(self, texts, role):
        texts = _texts(texts)
        if not texts:
            return np.empty((0, self.dimensions), dtype=np.float32)
        async with self.lock:
            keys = list(dict.fromkeys(hashlib.sha256(t.encode()).hexdigest() for t in texts))
            text_by_key = {hashlib.sha256(t.encode()).hexdigest(): t for t in texts}
            found, missing = {}, []
            try:
                with sqlite3.connect(self.path) as c:
                    for key in keys:
                        row = c.execute(
                            "SELECT value,checksum FROM vectors WHERE namespace=? AND role=? AND "
                            "digest=?",
                            (self.namespace, role, key),
                        ).fetchone()
                        valid = (
                            row is not None
                            and isinstance(row[0], bytes)
                            and len(row[0]) == self.dimensions * 4
                            and hashlib.sha256(row[0]).hexdigest() == row[1]
                        )
                        if valid:
                            data = np.frombuffer(row[0], dtype="<f4").copy()
                            valid = bool(np.isfinite(data).all() and np.linalg.norm(data) > 0)
                        if valid:
                            found[key] = data
                            self.stats["hits"] += 1
                            c.execute(
                                "UPDATE vectors SET touched=? WHERE namespace=? "
                                "AND role=? AND digest=?",
                                (time.time(), self.namespace, role, key),
                            )
                        else:
                            if row is not None:
                                self.stats["corrupt_rows"] += 1
                                c.execute(
                                    "DELETE FROM vectors WHERE namespace=? AND role=? AND digest=?",
                                    (self.namespace, role, key),
                                )
                            missing.append(key)
                    self.stats["misses"] += len(missing)
                # No database lock is held during native inference. Cancel propagates.
                for offset in range(0, len(missing), self.batch_size):
                    batch = missing[offset : offset + self.batch_size]
                    method = (
                        self.encoder.encode_queries
                        if role == "query"
                        else self.encoder.encode_documents
                    )
                    vectors = self.validate(
                        await method([text_by_key[k] for k in batch]), len(batch)
                    )
                    await asyncio.sleep(0)  # Do not persist an incomplete cancelled operation.
                    with sqlite3.connect(self.path) as c:
                        for key, vector in zip(batch, vectors, strict=True):
                            blob = vector.tobytes()
                            c.execute(
                                "INSERT OR REPLACE INTO vectors VALUES(?,?,?,?,?,?)",
                                (
                                    self.namespace,
                                    role,
                                    key,
                                    blob,
                                    hashlib.sha256(blob).hexdigest(),
                                    time.time(),
                                ),
                            )
                            found[key] = vector.copy()
                        c.execute(
                            "DELETE FROM vectors WHERE rowid IN (SELECT rowid FROM vectors ORDER "
                            "BY touched DESC LIMIT -1 OFFSET ?)",
                            (self.max_entries,),
                        )
                    self.stats["encoded_texts"] += len(batch)
                    self.stats["batches"] += 1
            except sqlite3.Error:
                raise ResearchError("embedding_cache", "向量缓存读写失败。") from None
            return np.stack([found[hashlib.sha256(t.encode()).hexdigest()] for t in texts])

    async def encode_queries(self, texts):
        return await self.encode(texts, "query")

    async def encode_documents(self, texts):
        return await self.encode(texts, "document")
