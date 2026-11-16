"""Integrated harness entry; ordinary CLI/Web behaviour remains unchanged."""

import asyncio
import hashlib
import time
from contextlib import AsyncExitStack
from pathlib import Path

from deep_research.dag_demo import DagDemoRetriever
from deep_research.dag_journal import DagJournal
from deep_research.dag_research import DagOptions, run_dag_pipeline
from deep_research.demo import DEMO_QUESTION
from deep_research.gateway import ModelGateway
from deep_research.harness import HarnessRuntime
from deep_research.harness_store import HarnessStore, digest, fail
from deep_research.model_budget import ModelBudgetLimits
from deep_research.research_store import ResearchStore
from deep_research.retrieval_cache import CachedRetriever
from deep_research.review_demo import ReviewDemoProvider
from deep_research.review_journal import ReviewJournal
from deep_research.review_models import ReviewOptions


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class HarnessDemoProvider(ReviewDemoProvider):
    """Stateless fixture decisions survive replay; never a factual-quality judge."""

    def _red(self, context):
        text = context["report"]["sections"][0]["claims"][0]["text"]
        self.red_count = 2 if text.startswith("教学修订") else 1
        return super()._red(context)


def engine_fingerprint():
    files = sorted(Path(__file__).parent.rglob("*.py"))
    return digest({p.relative_to(Path(__file__).parent).as_posix(): file_hash(p) for p in files})


async def run_harness(
    *,
    db: Path,
    output_dir: Path,
    question=DEMO_QUESTION,
    resume=None,
    seconds=600,
    stop_after_fetch=0,
    force_refresh=False,
    on_progress=print,
):
    store = HarnessStore(db.with_suffix(".harness.sqlite3"))
    output_absolute = str(await asyncio.to_thread(output_dir.resolve))
    engine = await asyncio.to_thread(engine_fingerprint)
    if resume:
        saved = store.get(resume)
        config = saved["config"]
        question = config["question"]
        if config["engine"] != engine:
            raise fail("configuration_changed")
        if config["output_dir"] != output_absolute:
            raise fail("configuration_changed")
        if saved["status"] == "completed":
            artifact = Path(saved["result"]["report_path"])
            if (
                not await asyncio.to_thread(artifact.is_file)
                or await asyncio.to_thread(file_hash, artifact) != saved["result"]["report_sha256"]
            ):
                raise fail("artifact_changed")
            return dict(harness_id=resume, reused_completed=True, **saved["result"])
    else:
        if question != DEMO_QUESTION:
            raise fail("offline_question")
        config = dict(
            version=1,
            engine=engine,
            question=question,
            output_dir=output_absolute,
            force_refresh=force_refresh,
            action_limits=dict(model=9, search=6, fetch=8),
        )
    rid = resume or store.create(config, seconds=seconds)
    on_progress("Harness ID: " + rid)
    async with AsyncExitStack() as stack:
        runtime = await stack.enter_async_context(HarnessRuntime(store, rid, config))
        provider, retriever = HarnessDemoProvider("repair"), DagDemoRetriever()
        namespace = digest(dict(source="harness-offline-v1", version=1))
        cache = CachedRetriever(
            retriever,
            db.with_suffix(".retrieval-cache.sqlite3"),
            namespace,
            force_refresh=config["force_refresh"],
        )
        stack.push_async_callback(cache.close)
        wrapped = runtime.retriever(cache)
        root_task = asyncio.current_task()
        fetched = 0

        class InterruptingRetriever:
            async def search(self, query, limit=3):
                return await wrapped.search(query, limit=limit)

            async def fetch(self, hit):
                nonlocal fetched
                value = await wrapped.fetch(hit)
                fetched += 1
                if stop_after_fetch and fetched == stop_after_fetch:
                    root_task.cancel()
                    await asyncio.sleep(0)
                return value

        # Each reconstruction has its own old-style run ID; the stable harness ID
        # groups attempts and owns the original deadline/checkpoints/action budget.
        with ResearchStore(db) as research, DagJournal(db) as journal, ReviewJournal(db) as reviews:
            deadline = min(240, runtime.remaining())
            result = await run_dag_pipeline(
                question,
                gateway=ModelGateway(runtime.provider(provider)),
                retriever=InterruptingRetriever(),
                store=research,
                journal=journal,
                review_journal=reviews,
                review_options=ReviewOptions(max_rounds=1),
                output_dir=output_dir,
                options=DagOptions(total_timeout_seconds=deadline, max_replans=0),
                source_timestamp=runtime.source_timestamp,
                model_budget=ModelBudgetLimits(max_calls=9),
                source_mode="offline_demo",
                on_progress=on_progress,
            )
            bundle = research.get_report(result["run_id"])
            summary = dict(
                run_id=result["run_id"],
                report_path=result["report_path"],
                report_sha256=await asyncio.to_thread(file_hash, Path(result["report_path"])),
                review_status=(bundle.get("review") or {}).get("status"),
                checkpoint_reuse=runtime.stats,
                retrieval_cache=cache.stats,
                research_map=bundle.get("research_map"),
                gap_retrieval=bundle.get("gap_retrieval"),
                deadline_remaining=max(0, store.get(rid)["deadline"] - time.time()),
            )
            runtime.complete(summary)
    return dict(harness_id=rid, **summary)
