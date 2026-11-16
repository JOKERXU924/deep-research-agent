"""Real parsing/journals with deterministic responses; no model or network access."""

import json
from uuid import uuid4

import httpx
import pytest
from test_compression_research import QUESTION
from test_review_research import ScriptedProvider, execute
from test_review_research import storage as storage

from deep_research.dag_journal import DagJournal
from deep_research.model_budget import ModelBudgetLimits
from deep_research.research import ResearchRunError
from deep_research.review_journal import ReviewJournal
from deep_research.web_app import create_app
from deep_research.web_jobs import WebJobStore
from deep_research.web_models import WebRequest

PRIVATE_KEY = "PRIVATE_MODEL_EXTRA_KEY"
PRIVATE_VALUE = "PRIVATE_MODEL_RESPONSE_VALUE_DO_NOT_REPLAY"


class InvalidReviewProvider(ScriptedProvider):
    """Violate actual cross-field validators, not just throw a mocked exception."""

    def __init__(self, *, red_uses_repair: bool):
        super().__init__("low" if red_uses_repair else "repair")
        self.red_uses_repair = red_uses_repair
        self.invalid_outputs: list[str] = []

    async def complete(self, request):
        response = await super().complete(request)
        stage = self.calls[-1][0]
        data = json.loads(response.content)
        if stage == "red" and self.red_calls == 1 and self.red_uses_repair:
            # category=evidence cannot omit claim targets. JSON syntax stays valid.
            data["findings"][0]["claim_ids"] = []
            data["findings"][0]["description"] = PRIVATE_VALUE
        elif stage == "blue" and self.blue_calls == 1:
            # MODIFY accepts target_id and claim, but forbids section_index.
            data["operations"][0]["section_index"] = 0
            data["operations"][0]["rationale"] = PRIVATE_VALUE
        else:
            return response
        data[PRIVATE_KEY] = PRIVATE_VALUE
        content = json.dumps(data)
        self.invalid_outputs.append(content)
        return response.model_copy(update={"content": content})


def assert_diagnostic(call, schema, path, kind):
    assert call["status"] == "failed"
    assert call["details"]["error_code"] == "schema_mismatch"
    assert call["details"]["usage"] == {
        "input_tokens": 7,
        "output_tokens": 3,
        "total_tokens": 10,
    }
    diagnostic = call["details"]["validation"]
    assert diagnostic["schema"] == schema
    assert diagnostic["truncated"] is False
    assert {"path": path, "kind": kind} in diagnostic["issues"]
    assert {"path": "<unknown>", "kind": "extra_forbidden"} in diagnostic["issues"]
    assert PRIVATE_KEY not in json.dumps(call)
    assert PRIVATE_VALUE not in json.dumps(call)
    return diagnostic


def assert_accounting(snapshot, *, expected_calls, budgeted):
    calls = snapshot["model_calls"]
    assert len(calls) == expected_calls
    assert sum(call["details"]["usage"]["total_tokens"] for call in calls) == 10 * expected_calls
    if budgeted:
        budget = snapshot["model_budget"]
        assert budget["reserved"]["calls"] == expected_calls
        assert budget["known_usage"] == {
            "input_tokens": 7 * expected_calls,
            "output_tokens": 3 * expected_calls,
            "total_tokens": 10 * expected_calls,
        }
        assert budget["usage_known_calls"] == expected_calls
        assert budget["usage_unknown_calls"] == 0
        assert budget["usage_complete"] is True
    else:
        assert "model_budget" not in snapshot


def assert_raw_responses_not_replayed(provider):
    for _, request in provider.calls:
        encoded = request.model_dump_json()
        assert PRIVATE_KEY not in encoded
        assert PRIVATE_VALUE not in encoded
        assert all(raw not in encoded for raw in provider.invalid_outputs)


def assert_retry_feedback(request, diagnostic):
    instructions = "\n".join(message.content for message in request.messages)
    marker = "Safe validation diagnostics (* denotes an array item): "
    assert marker in instructions
    feedback, _ = json.JSONDecoder().raw_decode(instructions.split(marker, 1)[1])
    assert feedback == diagnostic


@pytest.mark.parametrize("budgeted", [False, True], ids=["legacy", "budgeted"])
async def test_blue_invalid_combination_repairs_with_safe_feedback_and_preserved_usage(
    tmp_path, storage, budgeted
):
    provider = InvalidReviewProvider(red_uses_repair=False)
    result = await execute(
        tmp_path,
        storage,
        provider=provider,
        model_budget=ModelBudgetLimits() if budgeted else None,
    )
    run_id = result["run_id"]
    snapshot = storage[1].snapshot(run_id)
    calls = snapshot["model_calls"]
    assert [call["stage"] for call in calls] == [
        "plan",
        "report",
        "review.red.0",
        "review.blue.1",
        "review.blue.1.format",
        "review.red.1",
    ]
    diagnostic = assert_diagnostic(calls[3], "BlueRepairs", "operations.*", "operation_fields")
    assert [call["status"] for call in calls] == [
        "completed",
        "completed",
        "completed",
        "failed",
        "completed",
        "completed",
    ]
    retry = provider.calls[4][1]
    assert json.loads(provider.calls[3][1].messages[-1].content) == json.loads(
        retry.messages[-1].content
    )
    assert_retry_feedback(retry, diagnostic)
    assert_raw_responses_not_replayed(provider)
    assert result["status"] == "completed"
    assert result["review"]["status"] == "passed"
    assert result["review"]["format_repairs"] == 1
    assert result["review"]["best_round"] == 1
    assert result["review_model_calls"] == 4
    assert result["model_call_count"] == 6
    assert_accounting(snapshot, expected_calls=6, budgeted=budgeted)
    bundle = storage[0].get_report(run_id)
    assert bundle["scheduler_audit"]["model_calls"][3]["details"]["validation"] == diagnostic
    assert [item["event"]["kind"] for item in bundle["review_events"]] == [
        "initial",
        "red",
        "blue",
        "red",
        "stop",
    ]
    with DagJournal(tmp_path / "review.sqlite") as reopened:
        assert reopened.snapshot(run_id)["model_calls"] == calls


async def assert_failed_audit_available_through_api(tmp_path, run_id, expected_calls):
    """Expose an already-finished fixture run, without submitting or dispatching work."""
    db = tmp_path / "review.sqlite"
    with WebJobStore(db) as web:
        job, created = web.submit(
            WebRequest(mode="live", question=QUESTION, confirm_live=True), str(uuid4()), 8
        )
        assert created
        assert web.begin_next()["id"] == job["id"]
        web.bind_run(job["id"], run_id)
        web.finish(job["id"], "failed", error_code="schema_mismatch")

    async def forbidden_runner(*_args):
        pytest.fail("read-only audit checks must not dispatch any research")

    app = create_app(db=db, output_dir=tmp_path / "reports", runner=forbidden_runner)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1:8765"
        ) as client:
            response = await client.get(f"/api/jobs/{job['id']}/model-audit")
            assert response.status_code == 200
            exposed = response.json()["calls"]
            assert len(exposed) == len(expected_calls)
            for actual, expected in zip(exposed, expected_calls, strict=True):
                assert actual["stage"] == expected["stage"]
                assert actual["details"] == expected["details"]
            assert PRIVATE_KEY not in response.text
            assert PRIVATE_VALUE not in response.text
            assert (await client.get(f"/api/jobs/{job['id']}/report")).status_code == 409


@pytest.mark.parametrize("budgeted", [False, True], ids=["legacy", "budgeted"])
async def test_red_spent_repair_then_blue_failure_keeps_diagnostics_and_no_extra_calls(
    tmp_path, storage, budgeted
):
    provider = InvalidReviewProvider(red_uses_repair=True)
    with pytest.raises(ResearchRunError) as caught:
        await execute(
            tmp_path,
            storage,
            provider=provider,
            model_budget=ModelBudgetLimits() if budgeted else None,
        )
    assert caught.value.code == "schema_mismatch"
    run_id = caught.value.run_id
    snapshot = storage[1].snapshot(run_id)
    calls = snapshot["model_calls"]
    assert [call["stage"] for call in calls] == [
        "plan",
        "report",
        "review.red.0",
        "review.red.0.format",
        "review.blue.1",
    ]
    assert len(provider.calls) == 5
    red_diagnostic = assert_diagnostic(
        calls[2], "RedReview", "findings.*", "finding_claim_required"
    )
    assert_diagnostic(calls[4], "BlueRepairs", "operations.*", "operation_fields")
    assert calls[3]["status"] == "completed"
    assert_retry_feedback(provider.calls[3][1], red_diagnostic)
    assert_raw_responses_not_replayed(provider)
    assert_accounting(snapshot, expected_calls=5, budgeted=budgeted)
    assert storage[0].get_run(run_id)["status"] == "failed"
    assert storage[0].get_report(run_id) is None
    assert not (tmp_path / "reports").exists()
    events = storage[2].snapshot(run_id)
    assert [item["event"]["kind"] for item in events] == ["initial", "red"]
    assert events[0]["event"]["report"]["title"] == "Initial report"
    assert events[0]["event"]["report_hash"] == events[1]["event"]["report_hash"]
    with DagJournal(tmp_path / "review.sqlite") as reopened:
        assert reopened.snapshot(run_id)["model_calls"] == calls
    with ReviewJournal(tmp_path / "review.sqlite") as reopened:
        assert reopened.snapshot(run_id) == events
    await assert_failed_audit_available_through_api(tmp_path, run_id, calls)
    assert not (tmp_path / "reports").exists()
