"""Free crash/recovery experiment. Synthetic latency and fixtures, not model-quality evidence."""

import argparse
import asyncio
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from uuid import uuid4

from deep_research.errors import ResearchError
from deep_research.harness import HarnessRuntime
from deep_research.harness_runner import engine_fingerprint, run_harness
from deep_research.harness_store import HarnessStore, digest
from deep_research.retrieval import SearchHit, SourceDocument
from deep_research.retrieval_cache import CachedRetriever
from deep_research.schemas import ChatMessage, ModelRequest, ModelResponse

ROOT = Path(__file__).resolve().parents[1]
REQUEST = ModelRequest(
    messages=(ChatMessage(role="user", content="fixed fixture plan"),), max_output_tokens=32
)
CONFIG = dict(version=1, fixture="six-source-crash", action_limits=dict(model=9, search=6, fetch=8))


class Source:
    def __init__(self):
        self.fetches, self.searches = 0, 0

    async def search(self, query, limit=3):
        self.searches += 1
        await asyncio.sleep(0.01)
        return [SearchHit("https://fixture.example/" + query, query)]

    async def fetch(self, hit):
        self.fetches += 1
        await asyncio.sleep(0.01)
        return SourceDocument(hit.url, hit.title, "Evidence snapshot for " + hit.title)


class Provider:
    def __init__(self):
        self.calls = 0

    async def complete(self, request):
        self.calls += 1
        return ModelResponse(
            provider="fixture",
            model="fixture",
            request_id="fixed",
            content='{"plan":"six sources"}',
            latency_ms=0,
        )


def hits():
    return [SearchHit(f"https://fixture.example/{i}", str(i)) for i in range(6)]


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


async def worker(db, rid, unknown):
    store = HarnessStore(db)
    async with HarnessRuntime(store, rid, CONFIG) as runtime:
        if unknown:
            store.begin(
                rid,
                runtime.owner,
                "model",
                dict(ordinal=1, request="uncertain dispatch"),
                limits=CONFIG["action_limits"],
            )
            os._exit(74)
        await runtime.provider(Provider()).complete(REQUEST)
        retriever = runtime.retriever(Source())
        for hit in hits()[:4]:
            await retriever.fetch(hit)
        # Deliberately bypass finally blocks, heartbeat cleanup and lease release.
        os._exit(73)


async def main():
    out = ROOT / "outputs/research-harness-20261104" / str(uuid4())
    await asyncio.to_thread(out.mkdir, parents=True)
    store = HarnessStore(out / "checkpoint.sqlite3")
    first, unknown = (store.create(CONFIG) for _ in range(2))
    save(
        out / "protocol.json",
        dict(
            engine=engine_fingerprint(),
            script_sha256=hashlib.sha256(
                await asyncio.to_thread(Path(__file__).read_bytes)
            ).hexdigest(),
            sources=6,
            crash_after_saved_fetches=4,
            comparison="same saved prefix; full restart vs checkpoint recovery",
            live_provider=False,
            paid_calls=0,
        ),
    )
    processes = []
    for rid, flag in ((first, []), (unknown, ["--unknown"])):
        processes.append(
            await asyncio.to_thread(
                subprocess.Popen,
                [sys.executable, __file__, "--worker", str(out / "checkpoint.sqlite3"), rid, *flag],
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
        )
    codes = await asyncio.gather(*(asyncio.to_thread(p.wait, timeout=15) for p in processes))
    assert codes == [73, 74]
    before = store.get(first)["deadline"]
    try:
        store.claim(first, CONFIG)
    except ResearchError as error:
        assert error.code == "harness_busy"
    else:
        raise AssertionError("Live lease must exclude another owner")
    print("Workers force-exited; waiting for their original leases to expire.", flush=True)
    await asyncio.sleep(
        max(0, max(store.get(rid)["lease"] for rid in (first, unknown)) - time.time()) + 0.05
    )
    provider, source = Provider(), Source()
    async with HarnessRuntime(store, first, CONFIG) as runtime:
        response = await runtime.provider(provider).complete(REQUEST)
        assert response.content == '{"plan":"six sources"}'
        retriever = runtime.retriever(source)
        documents = [await retriever.fetch(hit) for hit in hits()]
        assert all(d.text == "Evidence snapshot for " + d.title for d in documents)
        runtime.complete(dict(snapshot_hash=digest([d.text for d in documents])))
        reuse = dict(runtime.stats)
    assert source.fetches == 2 and provider.calls == 0 and reuse["reused_fetch"] == 4
    assert store.get(first)["deadline"] == before
    try:
        async with HarnessRuntime(store, unknown, CONFIG):
            raise AssertionError("Unknown model request must never be resent")
    except ResearchError as error:
        assert error.code == "harness_model_outcome_unknown"
    assert store.get(unknown)["status"] == "blocked"
    baseline = Source()
    for hit in hits():
        await baseline.fetch(hit)
    cache_source = Source()
    cache = CachedRetriever(cache_source, out / "cache.sqlite3", "fixture-v1")
    results = await asyncio.gather(*(cache.search("same-query") for _ in range(16)))
    assert all(r == results[0] for r in results) and cache_source.searches == 1
    await cache.search("same-query")
    assert cache_source.searches == 1 and cache.stats["hits"] == 1
    await cache.close()
    db, report_out = out / "integrated.sqlite3", out / "reports"
    # Actual old DAG + review loop, with a task cancellation after durable fetch.
    task = asyncio.create_task(
        run_harness(db=db, output_dir=report_out, stop_after_fetch=1, on_progress=lambda _: None)
    )
    try:
        await task
    except asyncio.CancelledError:
        pass
    integrated = HarnessStore(db.with_suffix(".harness.sqlite3"))
    integrated_id = integrated.list_runs()[0]["id"]
    report = await run_harness(
        db=db, output_dir=report_out, resume=integrated_id, on_progress=lambda _: None
    )
    assert report["review_status"] == "passed"
    result = dict(
        verified=True,
        crash_exit_codes=codes,
        original_deadline_preserved=True,
        completed_snapshot_reused=reuse,
        recovery_fetches=dict(
            full_restart=baseline.fetches,
            checkpoint_recovery=source.fetches,
            reduction_pct=100 * (1 - source.fetches / baseline.fetches),
        ),
        including_saved_prefix=dict(
            full_restart=4 + baseline.fetches, checkpoint_recovery=6, reduction_pct=40.0
        ),
        concurrent_same_query=dict(
            requests=16, upstream=cache_source.searches, reduction_pct=93.75, stats=cache.stats
        ),
        unknown_model_recovery="blocked_without_resend",
        integrated_report=report,
        paid_calls=0,
        limitations=[
            "Synthetic local retrieval and scripted model; not production latency",
            "Forced process exit proves recovery at completed external-action boundaries",
            "Report fixture checks integration, not semantic research accuracy",
        ],
    )
    save(out / "result.json", result)
    print(
        json.dumps(
            dict(
                output_dir=str(out), **{k: v for k, v in result.items() if k != "integrated_report"}
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", nargs=2)
    parser.add_argument("--unknown", action="store_true")
    args = parser.parse_args()
    if args.worker:
        asyncio.run(worker(Path(args.worker[0]), args.worker[1], args.unknown))
    else:
        asyncio.run(main())
