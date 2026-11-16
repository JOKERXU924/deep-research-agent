"""Resumable external-action checkpoints around the existing research pipeline.

Recovery replays deterministic local computation; completed external actions are
restored, not dispatched. This is not continuation of an asyncio stack, and cannot
promise exactly-once execution by a remote model provider.
"""

import asyncio
import contextlib
import json
import time
from dataclasses import asdict
from datetime import UTC, datetime

from deep_research.harness_store import encode, fail
from deep_research.retrieval import SearchHit, SourceDocument
from deep_research.schemas import ModelResponse


class HarnessRuntime:
    def __init__(self, store, rid, config):
        self.store, self.rid, self.config = store, rid, config
        self.owner = None
        self.heartbeat_task = None
        self.model_index = 0
        self.locks = {}
        self.timestamps = {}
        self.fetch_order = []
        self.fetch_position = 0
        self.condition = asyncio.Condition()
        self.stats = dict(reused_model=0, reused_search=0, reused_fetch=0)
        self.failure = None

    async def __aenter__(self):
        self.owner = self.store.claim(self.rid, self.config)
        with self.store.connect() as db:
            rows = db.execute(
                "SELECT a.key FROM harness_events e JOIN harness_actions a "
                "ON a.run_id=e.run_id AND a.key=json_extract(e.data,'$.key') "
                "WHERE e.run_id=? AND e.kind='action_completed' AND a.kind='fetch' "
                "AND a.status='completed' ORDER BY e.seq",
                (self.rid,),
            ).fetchall()
        self.fetch_order = list(dict.fromkeys(r[0] for r in rows))

        async def heartbeat():
            while True:
                await asyncio.sleep(5)
                self.store.heartbeat(self.rid, self.owner)

        self.heartbeat_task = asyncio.create_task(heartbeat())
        return self

    async def __aexit__(self, *args):
        self.heartbeat_task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await self.heartbeat_task

        def release():
            if self.store.get(self.rid)["owner"] == self.owner:
                self.store.release(self.rid, self.owner)

        if args[0] is not None:
            with contextlib.suppress(Exception):
                release()
        else:
            release()

    async def execute(self, kind, args, call):
        if self.failure is not None:
            raise self.failure
        try:
            return await self._execute(kind, args, call)
        except Exception as error:
            if kind == "model" or getattr(error, "code", "").startswith("harness_"):
                self.failure = fail("execution_stopped")
            raise

    async def _execute(self, kind, args, call):
        if self.heartbeat_task.done():
            self.heartbeat_task.result()
        key, _ = self.store.action(self.rid, kind, args)
        lock = self.locks.setdefault(key, asyncio.Lock())
        async with lock:
            key, old = self.store.action(self.rid, kind, args)
            if old and old["status"] == "completed":
                if kind == "fetch" and key in self.fetch_order:
                    async with self.condition:
                        await self.condition.wait_for(
                            lambda: (
                                self.fetch_position >= len(self.fetch_order)
                                or self.fetch_order[self.fetch_position] == key
                                or key in self.fetch_order[: self.fetch_position]
                            )
                        )
                        if (
                            self.fetch_position < len(self.fetch_order)
                            and self.fetch_order[self.fetch_position] == key
                        ):
                            self.fetch_position += 1
                            self.condition.notify_all()
                self.store.reused(self.rid, self.owner, key, kind)
                self.stats["reused_" + kind] += 1
                return old["output"]
            # Any drift at a model ordinal stops before another paid request.
            if kind == "model":
                with self.store.connect() as db:
                    existing = db.execute(
                        "SELECT args FROM harness_actions WHERE run_id=? AND kind='model'",
                        (self.rid,),
                    ).fetchall()
                if any(json.loads(r[0])["ordinal"] == args["ordinal"] for r in existing):
                    raise fail("request_drift")
            key = self.store.begin(
                self.rid, self.owner, kind, args, limits=self.config["action_limits"]
            )
            try:
                result = await call()
                if kind == "fetch" and self.fetch_order:
                    async with self.condition:
                        await self.condition.wait_for(
                            lambda: self.fetch_position >= len(self.fetch_order)
                        )
                self.store.finish(self.rid, self.owner, key, result)
                return result
            except BaseException as error:
                with contextlib.suppress(Exception):
                    self.store.action_error(
                        self.rid,
                        self.owner,
                        key,
                        kind,
                        getattr(error, "code", type(error).__name__),
                    )
                raise

    def provider(self, inner):
        runtime = self

        class Provider:
            async def complete(self, request):
                runtime.model_index += 1
                stage = "model"
                try:
                    schema_text = request.messages[0].content.split("JSON Schema. ", 1)[-1]
                    schema_text = schema_text.split("\n", 1)[-1]
                    stage = json.loads(schema_text).get("title", "model")
                except (ValueError, IndexError):
                    pass
                point_ids = []
                try:
                    payload = json.loads(request.messages[-1].content)
                    points = payload.get("requirements", payload.get("points", []))
                    point_ids = sorted(
                        {
                            p.get("point_id", p.get("id"))
                            for p in points
                            if isinstance(p, dict)
                            and isinstance(p.get("point_id", p.get("id")), str)
                        }
                    )[:8]
                except (ValueError, TypeError, AttributeError):
                    pass
                args = dict(
                    ordinal=runtime.model_index,
                    stage=stage,
                    point_ids=point_ids,
                    request=request.model_dump(mode="json"),
                )

                async def call():
                    return (await inner.complete(request)).model_dump(mode="json")

                value = await runtime.execute("model", args, call)
                return ModelResponse.model_validate(value)

        return Provider()

    def retriever(self, inner):
        runtime = self

        class Retriever:
            async def search(self, query, limit=3):
                async def call():
                    return [asdict(x) for x in await inner.search(query, limit=limit)]

                rows = await runtime.execute("search", dict(query=query, limit=limit), call)
                return [SearchHit(**x) for x in rows]

            async def fetch(self, hit):
                async def call():
                    return dict(
                        document=asdict(await inner.fetch(hit)),
                        retrieved_at=datetime.now(UTC).isoformat(),
                    )

                value = await runtime.execute("fetch", asdict(hit), call)
                runtime.timestamps[hit.url] = value["retrieved_at"]
                return SourceDocument(**value["document"])

        return Retriever()

    def source_timestamp(self, url):
        return self.timestamps[url]

    def remaining(self):
        value = self.store.get(self.rid)["deadline"] - time.time()
        if value <= 0:
            raise fail("deadline")
        return value

    def complete(self, result):
        if self.failure is not None:
            raise self.failure
        # Public result stays small; raw model responses remain local checkpoints.
        self.store.release(self.rid, self.owner, result=json.loads(encode(result)))
