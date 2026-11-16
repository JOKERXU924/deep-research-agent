"""Report commit wins over cancellation arriving during async client cleanup.

The provider/retriever below are local fixtures; no settings or network are read.
"""

import asyncio
from contextlib import asynccontextmanager

import httpx
import pytest

from deep_research import web_runner as runner_module
from deep_research.dag_demo import DagDemoRetriever
from deep_research.errors import ResearchError
from deep_research.memory_store import MemoryStore
from deep_research.research_store import ResearchStore
from deep_research.review_demo import ReviewDemoProvider
from deep_research.web_app import create_app


@pytest.fixture
def controlled_live(tmp_path, monkeypatch):
    @asynccontextmanager
    async def setup(*, memory_failure=False, block_first_call=False):
        db = tmp_path / "late-cancel.sqlite3"
        closing, release_cleanup = asyncio.Event(), asyncio.Event()
        call_started, release_call = asyncio.Event(), asyncio.Event()
        cleanup_cancelled, cleanup_completed = [], []

        class Provider(ReviewDemoProvider):
            def __init__(self, settings):
                super().__init__("passed")
                self.first = True

            async def complete(self, request):
                if self.first:
                    self.first = False
                    call_started.set()
                    if block_first_call:
                        await release_call.wait()
                return await super().complete(request)

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

        class Retriever(DagDemoRetriever):
            def __init__(self, language):
                super().__init__()

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                closing.set()
                try:
                    await release_cleanup.wait()
                except asyncio.CancelledError:
                    cleanup_cancelled.append(True)
                    raise
                cleanup_completed.append(True)

        monkeypatch.setattr(runner_module, "load_settings", lambda *_: object())
        monkeypatch.setattr(runner_module, "DeepSeekProvider", Provider)
        monkeypatch.setattr(runner_module, "WikipediaRetriever", Retriever)
        if memory_failure:

            def fail_publication(*_, **__):
                raise ResearchError("memory_io", "fixture publication failure")

            monkeypatch.setattr(MemoryStore, "ingest_run", fail_publication)
        runner = runner_module.make_runner(db=db, output_dir=tmp_path / "out", allow_live=True)
        app = create_app(db=db, output_dir=tmp_path / "out", runner=runner, allow_live=True)
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1:8765"
            ) as client:
                try:
                    yield {
                        "app": app,
                        "client": client,
                        "db": db,
                        "closing": closing,
                        "release_cleanup": release_cleanup,
                        "call_started": call_started,
                        "release_call": release_call,
                        "cleanup_cancelled": cleanup_cancelled,
                        "cleanup_completed": cleanup_completed,
                    }
                finally:
                    release_cleanup.set()
                    release_call.set()

    return setup


async def submit(client):
    from uuid import uuid4

    config = (await client.get("/api/config")).json()
    headers = {"X-CSRF-Token": config["csrf_token"], "Idempotency-Key": str(uuid4())}
    response = await client.post(
        "/api/jobs",
        headers=headers,
        json={
            "mode": "live",
            # Exercise the real topic gate against both offline demo documents.
            "question": "异步等待与并发限制",
            "confirm_live": True,
            "memory_scope": "late-cancel-test",
        },
    )
    assert response.status_code == 202
    return response.json()["id"], headers


async def terminal(client, job_id):
    async with asyncio.timeout(3):
        while True:
            job = (await client.get(f"/api/jobs/{job_id}")).json()
            if job["status"] in {"completed", "failed", "cancelled", "interrupted"}:
                return job
            await asyncio.sleep(0.005)


@pytest.mark.parametrize("memory_failure", [False, True])
async def test_cancel_after_report_commit_preserves_cleanup_and_publication_result(
    controlled_live, memory_failure
):
    async with controlled_live(memory_failure=memory_failure) as context:
        client = context["client"]
        job_id, headers = await submit(client)
        await asyncio.wait_for(context["closing"].wait(), 3)
        before = (await client.get(f"/api/jobs/{job_id}")).json()
        assert before["status"] == "running"
        with ResearchStore(context["db"]) as research:
            assert research.get_run(before["run_id"])["status"] == "completed"
        for _ in range(2):
            cancelled = (await client.post(f"/api/jobs/{job_id}/cancel", headers=headers)).json()
            assert cancelled["status"] == "running"
            assert cancelled["cancel_requested"] is False
        assert context["cleanup_cancelled"] == []
        context["release_cleanup"].set()
        final = await terminal(client, job_id)
        assert final["status"] == "completed" and final["cancel_requested"] is False
        assert final["result"]["memory_write"]["status"] == (
            "failed" if memory_failure else "saved"
        )
        assert context["cleanup_completed"] == [True]
        assert (await client.get(f"/api/jobs/{job_id}/report")).status_code == 200
        assert (await client.get(f"/api/jobs/{job_id}/download/json")).status_code == 200
        events = context["app"].state.manager.events(job_id)
        assert events[-1]["kind"] == "completed"
        assert not any(item["kind"] in {"cancelling", "cancelled"} for item in events)


async def test_cancel_before_report_commit_remains_cancelling_until_cleanup_finishes(
    controlled_live,
):
    async with controlled_live(block_first_call=True) as context:
        client = context["client"]
        job_id, headers = await submit(client)
        await asyncio.wait_for(context["call_started"].wait(), 3)
        accepted = (await client.post(f"/api/jobs/{job_id}/cancel", headers=headers)).json()
        assert accepted["status"] == "cancelling" and accepted["cancel_requested"]
        await asyncio.wait_for(context["closing"].wait(), 3)
        during = (await client.get(f"/api/jobs/{job_id}")).json()
        assert during["status"] == "cancelling"
        again = (await client.post(f"/api/jobs/{job_id}/cancel", headers=headers)).json()
        assert again["status"] == "cancelling" and context["cleanup_cancelled"] == []
        context["release_cleanup"].set()
        final = await terminal(client, job_id)
        assert final["status"] == "cancelled" and final["result"] is None
        with ResearchStore(context["db"]) as research:
            assert research.get_run(final["run_id"])["status"] == "cancelled"
            assert research.get_report(final["run_id"]) is None
        assert (await client.get(f"/api/jobs/{job_id}/report")).status_code == 409
        assert context["cleanup_completed"] == [True]


async def test_service_close_after_commit_waits_for_cleanup_without_cancelling(controlled_live):
    async with controlled_live() as context:
        client = context["client"]
        job_id, _ = await submit(client)
        await asyncio.wait_for(context["closing"].wait(), 3)
        manager = context["app"].state.manager
        shutdown = asyncio.create_task(manager.close())
        await asyncio.sleep(0.01)
        assert not shutdown.done() and context["cleanup_cancelled"] == []
        assert manager.get(job_id)["cancel_requested"] is False
        context["release_cleanup"].set()
        await asyncio.wait_for(shutdown, 3)
        final = manager.get(job_id)
        assert final["status"] == "completed"
        assert final["result"]["memory_write"]["status"] == "saved"
        assert context["cleanup_completed"] == [True]
