"""Run the offline research, recovery, and request coalescing fixtures."""

import argparse
import asyncio
import json
import tempfile
from pathlib import Path

from deep_research.harness_runner import run_harness
from deep_research.harness_store import HarnessStore
from deep_research.research import validate_citations
from deep_research.research_models import ResearchReport
from deep_research.retrieval import SearchHit
from deep_research.retrieval_cache import CachedRetriever


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def report_summary(workspace, store, result, label):
    report_path = Path(result["report_path"])
    evidence_path = report_path.with_name("evidence.json")
    bundle = json.loads(evidence_path.read_text(encoding="utf-8"))
    report = ResearchReport.model_validate(bundle["report"])
    validate_citations(report, bundle["sources"], bundle["evidence"])
    run_id = result["harness_id"]
    trace = store.trace(run_id)
    require(trace["status"] == "completed", "Harness run did not finish")
    require(result["review_status"] == "passed", "Scripted review did not reach its final state")
    trace_path = workspace / f"{label}-trace.json"
    save_json(trace_path, trace)
    return {
        "harness_id": run_id,
        "report": report_path.relative_to(workspace).as_posix(),
        "evidence": evidence_path.relative_to(workspace).as_posix(),
        "trace": trace_path.relative_to(workspace).as_posix(),
        "review_status": result["review_status"],
        "citation_integrity_verified": True,
        "checkpoint_reuse": result["checkpoint_reuse"],
    }


async def coalescing_demo(workspace):
    callers = 16
    arrived = 0
    all_arrived = asyncio.Event()

    class FixtureRetriever:
        async def search(self, query, limit=3):
            await all_arrived.wait()
            return [SearchHit("https://fixture.example/demo", "Offline fixture")][:limit]

    cache = CachedRetriever(FixtureRetriever(), workspace / "coalescing.sqlite3", "public-demo")

    async def lookup():
        nonlocal arrived
        arrived += 1
        if arrived == callers:
            all_arrived.set()
        return await cache.search("same query", limit=1)

    try:
        async with asyncio.timeout(5):
            results = await asyncio.gather(*(lookup() for _ in range(callers)))
        require(all(item == results[0] for item in results), "Coalesced results differ")
        require(cache.stats["upstream"] == 1, "Identical requests reached upstream twice")
        require(cache.stats["joined"] == callers - 1, "Concurrent callers did not share the task")
        require(await cache.search("same query", limit=1) == results[0], "Cached result differs")
        require(cache.stats["hits"] == 1, "A later request did not read the cache")
        return {"concurrent_callers": callers, **cache.stats}
    finally:
        await cache.close()


async def run_demo(workspace):
    workspace = await asyncio.to_thread(workspace.resolve)
    # The dashboard discovers this database next to its own database.
    db = workspace / "harness-research.sqlite3"
    reports = workspace / "reports"
    store = HarnessStore(db.with_suffix(".harness.sqlite3"))
    options = {"db": db, "output_dir": reports, "on_progress": lambda _: None}

    uninterrupted = await run_harness(**options, force_refresh=True)
    normal_summary = report_summary(workspace, store, uninterrupted, "normal")

    task = asyncio.create_task(run_harness(**options, force_refresh=True, stop_after_fetch=1))
    try:
        await task
    except asyncio.CancelledError:
        if asyncio.current_task().cancelling():
            raise
    else:
        raise RuntimeError("The planned interruption did not occur")

    interrupted = [row for row in store.list_runs() if row["status"] == "interrupted"]
    require(len(interrupted) == 1, "Expected one interrupted run")
    run = interrupted[0]
    trace = store.trace(run["id"])
    save_json(workspace / "interrupted-trace.json", trace)
    saved_fetches = sum(
        row["kind"] == "fetch" and row["status"] == "completed" for row in trace["actions"]
    )
    recovered = await run_harness(**options, resume=run["id"])
    recovered_summary = report_summary(workspace, store, recovered, "resumed")
    reused = recovered["checkpoint_reuse"]
    require(reused["reused_fetch"] == saved_fetches > 0, "Saved fetches were not reused")
    require(reused["reused_model"] >= 1, "Saved model output was not reused")
    require(store.get(run["id"])["deadline"] == run["deadline"], "Resume reset the deadline")

    return {
        "mode": "offline_fixture",
        "normal": normal_summary,
        "recovery": {
            **recovered_summary,
            "interrupted_trace": "interrupted-trace.json",
            "saved_fetches_before_resume": saved_fetches,
            "original_deadline_preserved": True,
        },
        "coalescing": await coalescing_demo(workspace),
        "scope": "Local fixtures exercise execution behavior and citation integrity.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/public-demo"))
    args = parser.parse_args()
    parent = args.output_dir.resolve()
    parent.mkdir(parents=True, exist_ok=True)
    workspace = Path(tempfile.mkdtemp(prefix="run-", dir=parent))
    summary = asyncio.run(run_demo(workspace))
    save_json(workspace / "summary.json", summary)
    print(json.dumps({"output_dir": str(workspace), **summary}, indent=2))


if __name__ == "__main__":
    main()
