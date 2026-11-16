import asyncio
import json
import sqlite3

import pytest

from deep_research.errors import ResearchError
from deep_research.harness import HarnessRuntime
from deep_research.harness_runner import run_harness
from deep_research.harness_store import HarnessStore
from deep_research.retrieval import SearchHit, SourceDocument
from deep_research.schemas import ChatMessage, ModelRequest, ModelResponse


def config():
    return dict(version=1, action_limits=dict(model=2, search=6, fetch=8))


def request(text="question"):
    return ModelRequest(messages=(ChatMessage(role="user", content=text),), max_output_tokens=10)


class Provider:
    def __init__(self, fail=False):
        self.calls = 0
        self.fail = fail

    async def complete(self, req):
        self.calls += 1
        if self.fail:
            raise OSError("lost response")
        return ModelResponse(
            provider="fixture",
            model="fixture",
            request_id=str(self.calls),
            content="{}",
            latency_ms=0,
        )


def test_lease_exclusion_fencing_and_original_deadline(tmp_path):
    store = HarnessStore(tmp_path / "h.sqlite3")
    rid = store.create(config())
    original = store.get(rid)["deadline"]
    owner = store.claim(rid, config())
    with pytest.raises(ResearchError, match="busy"):
        store.claim(rid, config())
    with store.connect() as db:
        db.execute("UPDATE harness_runs SET lease=0 WHERE id=?", (rid,))
    other = store.claim(rid, config())
    with pytest.raises(ResearchError, match="lease_lost"):
        store.begin(rid, owner, "search", {}, limits=config()["action_limits"])
    assert store.get(rid)["deadline"] == original
    store.release(rid, other)
    with pytest.raises(ResearchError, match="configuration_changed"):
        store.claim(rid, dict(config(), version=2))


async def test_completed_model_reused_and_drift_stops_before_dispatch(tmp_path):
    store = HarnessStore(tmp_path / "h.sqlite3")
    rid = store.create(config())
    first = Provider()
    async with HarnessRuntime(store, rid, config()) as runtime:
        original = await runtime.provider(first).complete(request())
    second = Provider()
    async with HarnessRuntime(store, rid, config()) as runtime:
        actual = await runtime.provider(second).complete(request())
        assert actual == original and runtime.stats["reused_model"] == 1
    assert first.calls == 1 and second.calls == 0
    async with HarnessRuntime(store, rid, config()) as runtime:
        with pytest.raises(ResearchError, match="request_drift"):
            await runtime.provider(second).complete(request("changed"))
    assert second.calls == 0


async def test_unknown_model_outcome_blocks_resume_and_retains_reservation(tmp_path):
    store = HarnessStore(tmp_path / "h.sqlite3")
    rid = store.create(config())
    provider = Provider(fail=True)
    with pytest.raises(OSError):
        async with HarnessRuntime(store, rid, config()) as runtime:
            await runtime.provider(provider).complete(request())
    assert store.get(rid)["status"] == "blocked"
    with pytest.raises(ResearchError, match="model_outcome_unknown"):
        async with HarnessRuntime(store, rid, config()):
            pass
    assert provider.calls == 1
    assert store.trace(rid)["actions"][0]["attempts"] == 1


async def test_budget_survives_reconstruction(tmp_path):
    store = HarnessStore(tmp_path / "h.sqlite3")
    rid = store.create(config())
    p = Provider()
    async with HarnessRuntime(store, rid, config()) as runtime:
        provider = runtime.provider(p)
        await provider.complete(request("one"))
        await provider.complete(request("two"))
    async with HarnessRuntime(store, rid, config()) as runtime:
        provider = runtime.provider(p)
        await provider.complete(request("one"))
        await provider.complete(request("two"))
        with pytest.raises(ResearchError, match="budget"):
            await provider.complete(request("three"))
    assert p.calls == 2


async def test_corrupt_checkpoint_never_reused(tmp_path):
    store = HarnessStore(tmp_path / "h.sqlite3")
    rid = store.create(config())
    async with HarnessRuntime(store, rid, config()) as runtime:
        await runtime.provider(Provider()).complete(request())
    with store.connect() as db:
        db.execute("UPDATE harness_actions SET output='{}' WHERE run_id=?", (rid,))
    async with HarnessRuntime(store, rid, config()) as runtime:
        with pytest.raises(ResearchError, match="checkpoint_corrupt"):
            await runtime.provider(Provider()).complete(request())


async def test_fetched_checkpoints_replay_in_original_completion_order(tmp_path):
    store = HarnessStore(tmp_path / "h.sqlite3")
    rid = store.create(config())

    class Retriever:
        async def fetch(self, hit):
            return SourceDocument(hit.url, hit.title, "evidence")

    a, b = SearchHit("https://a.org", "a"), SearchHit("https://b.org", "b")
    async with HarnessRuntime(store, rid, config()) as runtime:
        r = runtime.retriever(Retriever())
        await r.fetch(b)
        await r.fetch(a)
        stamps = dict(runtime.timestamps)
    order = []
    async with HarnessRuntime(store, rid, config()) as runtime:
        r = runtime.retriever(Retriever())

        async def fetch(hit):
            await r.fetch(hit)
            order.append(hit.title)

        await asyncio.wait_for(asyncio.gather(fetch(a), fetch(b)), timeout=1)
        assert order == ["b", "a"] and runtime.timestamps == stamps


async def test_integrated_cancel_resume_review_and_artifact_validation(tmp_path):
    db, out = tmp_path / "research.sqlite3", tmp_path / "out"
    task = asyncio.create_task(
        run_harness(db=db, output_dir=out, stop_after_fetch=1, on_progress=lambda _: None)
    )
    with pytest.raises(asyncio.CancelledError):
        await task
    store = HarnessStore(db.with_suffix(".harness.sqlite3"))
    rid = store.list_runs()[0]["id"]
    before = store.get(rid)["deadline"]
    saved = store.trace(rid)
    completed_fetches = sum(
        a["kind"] == "fetch" and a["status"] == "completed" for a in saved["actions"]
    )
    result = await run_harness(db=db, output_dir=out, resume=rid, on_progress=lambda _: None)
    assert result["checkpoint_reuse"]["reused_fetch"] == completed_fetches > 0
    assert result["checkpoint_reuse"]["reused_model"] >= 1
    assert result["review_status"] == "passed" and store.get(rid)["deadline"] == before
    with sqlite3.connect(db) as connection:
        bundles = connection.execute(
            "SELECT report_json FROM research_runs WHERE status='completed'"
        )
        assert any(json.loads(r[0])["report"] for r in bundles)
    completed = await run_harness(db=db, output_dir=out, resume=rid, on_progress=lambda _: None)
    assert completed["reused_completed"] is True
    from pathlib import Path

    await asyncio.to_thread(Path(result["report_path"]).write_text, "changed", encoding="utf-8")
    with pytest.raises(ResearchError, match="artifact_changed"):
        await run_harness(db=db, output_dir=out, resume=rid, on_progress=lambda _: None)


async def test_resume_after_red_response_keeps_repair_loop_consistent(tmp_path, monkeypatch):
    original = HarnessRuntime._execute
    fired = False

    async def stop_after_red(self, kind, args, call):
        nonlocal fired
        value = await original(self, kind, args, call)
        if kind == "model" and args["ordinal"] == 3 and not fired:
            fired = True
            raise asyncio.CancelledError
        return value

    monkeypatch.setattr(HarnessRuntime, "_execute", stop_after_red)
    db, out = tmp_path / "r.sqlite3", tmp_path / "out"
    with pytest.raises(asyncio.CancelledError):
        await run_harness(db=db, output_dir=out, on_progress=lambda _: None)
    monkeypatch.setattr(HarnessRuntime, "_execute", original)
    store = HarnessStore(db.with_suffix(".harness.sqlite3"))
    rid = store.list_runs()[0]["id"]
    result = await run_harness(db=db, output_dir=out, resume=rid, on_progress=lambda _: None)
    assert result["checkpoint_reuse"]["reused_model"] == 3
    assert result["review_status"] == "passed"


async def test_drift_failure_cannot_be_caught_to_continue_or_complete(tmp_path):
    store = HarnessStore(tmp_path / "h.sqlite3")
    rid = store.create(config())
    p = Provider()
    async with HarnessRuntime(store, rid, config()) as runtime:
        await runtime.provider(p).complete(request())
    async with HarnessRuntime(store, rid, config()) as runtime:
        provider = runtime.provider(p)
        with pytest.raises(ResearchError, match="request_drift"):
            await provider.complete(request("changed"))
        with pytest.raises(ResearchError, match="execution_stopped"):
            await provider.complete(request("next"))
        with pytest.raises(ResearchError, match="execution_stopped"):
            runtime.complete({})
    assert p.calls == 1
