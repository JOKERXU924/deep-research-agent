"""No external dispatch after cancellation requested by synchronous auditing."""

import asyncio
import json

import pytest

from deep_research.dag_demo import DagDemoRetriever
from deep_research.dag_journal import DagJournal, DagJournalError
from deep_research.dag_research import run_dag_pipeline
from deep_research.gateway import ModelGateway
from deep_research.research_store import ResearchStore
from deep_research.review_demo import ReviewDemoProvider
from deep_research.review_journal import ReviewJournal
from deep_research.review_models import ReviewOptions


class RecordingProvider(ReviewDemoProvider):
    def __init__(self):
        super().__init__("passed")
        self.entered = []

    async def complete(self, request):
        context = json.loads(request.messages[-1].content)
        stage = (
            "review.red.0"
            if "claim_inventory" in context
            else "report"
            if "evidence" in context
            else "plan"
        )
        self.entered.append(stage)
        return await super().complete(request)


@pytest.mark.parametrize("cancel_stage", ["plan", "report", "review.red.0"])
@pytest.mark.parametrize("finish_fails", [False, True])
async def test_audit_requested_cancel_prevents_dispatch_and_preserves_cancel_on_cleanup_failure(
    tmp_path, monkeypatch, cancel_stage, finish_fails
):
    db = tmp_path / "cancel.sqlite"
    output = tmp_path / "reports"
    provider = RecordingProvider()
    call_stages = {}
    recorded_run_ids = []
    with ResearchStore(db) as store, DagJournal(db) as dag, ReviewJournal(db) as review:
        original_start = dag.start_model_call
        original_finish = dag.finish_model_call

        def cancel_after_start(run_id, stage):
            call_id = original_start(run_id, stage)
            call_stages[call_id] = stage
            recorded_run_ids.append(run_id)
            if stage == cancel_stage:
                asyncio.current_task().cancel("cancel requested by synchronous audit hook")
            return call_id

        def finish(run_id, call_id, status, details):
            if finish_fails and call_stages[call_id] == cancel_stage:
                raise DagJournalError("dag_audit_io", "Fixture cleanup audit failure.")
            return original_finish(run_id, call_id, status, details)

        monkeypatch.setattr(dag, "start_model_call", cancel_after_start)
        monkeypatch.setattr(dag, "finish_model_call", finish)
        task = asyncio.create_task(
            run_dag_pipeline(
                "Offline cancellation boundary",
                gateway=ModelGateway(provider),
                retriever=DagDemoRetriever(),
                store=store,
                journal=dag,
                output_dir=output,
                source_mode="offline_demo",
                review_options=ReviewOptions(),
                review_journal=review,
            )
        )
        with pytest.raises(asyncio.CancelledError):
            await task
        assert task.cancelled()
        assert cancel_stage not in provider.entered
        assert (
            provider.entered
            == {
                "plan": [],
                "report": ["plan"],
                "review.red.0": ["plan", "report"],
            }[cancel_stage]
        )
        assert len(set(recorded_run_ids)) == 1
        run_id = recorded_run_ids[0]
        assert store.get_run(run_id)["status"] == "cancelled"
        assert store.get_report(run_id) is None
        assert not output.exists()
        snapshot = dag.snapshot(run_id)
        last_call = snapshot["model_calls"][-1]
        assert last_call["stage"] == cancel_stage
        assert snapshot["summary"]["stop_reason"] == "cancelled"
        if finish_fails:
            # The unknown ledger terminal state remains visible, never fabricated.
            assert last_call["status"] == "in_flight"
        else:
            assert last_call["status"] == "cancelled"
            assert last_call["details"]["error_code"] == "cancelled"
        events = review.snapshot(run_id)
        assert [item["event"]["kind"] for item in events] == (
            ["initial"] if cancel_stage == "review.red.0" else []
        )
