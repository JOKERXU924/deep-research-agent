"""M6 persistent manager plus the actual M2--M5 pipeline, using offline providers."""

import asyncio
from uuid import uuid4

import httpx
import pytest

import deep_research.web_runner as runner_module
from deep_research.dag_journal import DagJournal
from deep_research.errors import ResearchError
from deep_research.research_store import ResearchStore
from deep_research.review_demo import ReviewDemoProvider
from deep_research.web_jobs import TERMINAL, WebJobStore, WebManager
from deep_research.web_models import WebRequest


@pytest.fixture
def paths(tmp_path, monkeypatch):
    monkeypatch.setattr(runner_module, "load_settings", lambda *_: pytest.fail("secret access"))
    monkeypatch.setattr(httpx, "AsyncClient", lambda *a, **kw: pytest.fail("network access"))
    return {"db": tmp_path / "integration.sqlite3", "output_dir": tmp_path / "reports"}


async def finished(manager, job_id):
    async with asyncio.timeout(5):
        while manager.get(job_id)["status"] not in TERMINAL:  # noqa: ASYNC110
            await asyncio.sleep(0.005)
    return manager.get(job_id)


@pytest.mark.parametrize("compression", [False, True])
@pytest.mark.parametrize(
    "scenario", ["passed", "repair", "oscillation", "stalled", "max-rounds", "format", "deadline"]
)
async def test_real_pipeline_all_scenarios_survive_persistent_web_summary(
    paths, scenario, compression
):
    with WebJobStore(paths["db"]) as store:
        manager = WebManager(store, runner_module.make_runner(**paths))
        await manager.start()
        try:
            job, _ = manager.submit(
                WebRequest(scenario=scenario, compression=compression), str(uuid4())
            )
            done = await finished(manager, job["id"])
            assert done["status"] == "completed", done
            assert done["error_code"] is None
            if compression:
                expected = "skipped_deadline" if scenario == "deadline" else "completed"
                assert done["result"]["compression"]["status"] == expected
            if scenario == "deadline":
                assert done["result"]["degraded"]
                assert done["result"]["review"]["status"] == "interrupted"
                assert done["result"]["report_kind"] == "evidence_digest"
                assert "global_timeout" in done["result"]["reasons"]
            assert manager.events(job["id"])[-1]["kind"] == "completed"
            with ResearchStore(paths["db"]) as research:
                assert research.get_run(done["run_id"])["status"] == "completed"
                assert research.get_report(done["run_id"]) is not None
        finally:
            await manager.close()


@pytest.mark.parametrize("compression", [False, True])
async def test_no_review_pipeline_also_survives_manager_summary(paths, compression):
    with WebJobStore(paths["db"]) as store:
        manager = WebManager(store, runner_module.make_runner(**paths))
        await manager.start()
        try:
            job, _ = manager.submit(WebRequest(review=False, compression=compression), str(uuid4()))
            done = await finished(manager, job["id"])
            assert done["status"] == "completed"
            assert "review" not in done["result"]
        finally:
            await manager.close()


@pytest.mark.parametrize(
    "message_prefix,expected_calls", [("DAG 任务", 0), ("按依赖启动", 1), ("M5：", 2)]
)
async def test_progress_audit_failure_stops_actual_pipeline_before_next_call(
    paths, monkeypatch, message_prefix, expected_calls
):
    calls = []

    class CountingProvider(ReviewDemoProvider):
        async def complete(self, request):
            calls.append(request)
            return await super().complete(request)

    monkeypatch.setattr(runner_module, "ReviewDemoProvider", CountingProvider)
    with WebJobStore(paths["db"]) as store:
        original = store.progress

        def progress(job_id, message):
            if message.startswith(message_prefix):
                raise ResearchError("web_io", "fixture progress audit failure")
            original(job_id, message)

        monkeypatch.setattr(store, "progress", progress)
        manager = WebManager(store, runner_module.make_runner(**paths))
        await manager.start()
        try:
            job, _ = manager.submit(WebRequest(), str(uuid4()))
            done = await finished(manager, job["id"])
            assert done["status"] == "failed"
            assert done["error_code"] == "web_io"
            assert len(calls) == expected_calls
            with ResearchStore(paths["db"]) as research:
                assert research.get_run(done["run_id"])["status"] == "cancelled"
                assert research.get_report(done["run_id"]) is None
            with DagJournal(paths["db"]) as journal:
                ledger = journal.snapshot(done["run_id"])["model_calls"]
                assert len(ledger) == expected_calls
                assert all(item["status"] == "completed" for item in ledger)
            with pytest.raises(ResearchError) as caught:
                manager.submit(WebRequest(), str(uuid4()))
            assert caught.value.code == "web_unavailable"
        finally:
            await manager.close()
