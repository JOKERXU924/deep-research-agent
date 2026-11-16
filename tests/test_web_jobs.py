"""M6 queue tests use no API, HTTP listener, model download, or existing DB."""

import asyncio
import json
import sqlite3
import subprocess
import sys
from uuid import uuid4

import pytest
from pydantic import ValidationError

from deep_research.demo import DEMO_QUESTION
from deep_research.errors import ResearchError
from deep_research.research_store import ResearchStore
from deep_research.web_jobs import MAX_EVENTS, MAX_PROGRESS_EVENTS, WebJobStore, WebManager
from deep_research.web_models import WebRequest


def key():
    return str(uuid4())


async def wait_for_status(manager, job_id, status):
    async with asyncio.timeout(3):
        while manager.get(job_id)["status"] not in status:  # noqa: ASYNC110
            await asyncio.sleep(0)
    return manager.get(job_id)


async def complete(request, progress, created):
    run_id = key()
    created(run_id)
    progress("证据已保存，正在审查")
    return {"run_id": run_id, "status": "completed", "degraded": False}


@pytest.fixture
def store(tmp_path):
    with WebJobStore(tmp_path / "data" / "research.sqlite") as value:
        yield value


def test_request_defaults_and_live_confirmation():
    request = WebRequest()
    assert request.question == DEMO_QUESTION
    assert request.mode == "demo"
    assert request.scenario == "repair"
    assert request.review and not request.confirm_live
    assert WebRequest(mode="live", confirm_live=True, question="  你好  ").question == "你好"


@pytest.mark.parametrize(
    "payload",
    [
        {"mode": "live"},
        {"confirm_live": True},
        {"memory_scope": "team"},
        {"mode": "live", "confirm_live": True, "scenario": "slow"},
        {"mode": "live", "confirm_live": True, "memory_scope": "../secret"},
        {"question": ""},
        {"question": " "},
        {"question": "x" * 1001},
        {"question": "arbitrary fixed demo claim"},
        {"question": "\ud800"},
        {"review": "false"},
        {"compression": 1},
        {"api_key": "secret"},
        {"output_dir": "elsewhere"},
        {"provider": "other"},
        {"review": False, "scenario": "format"},
        {"scenario": "unknown"},
    ],
)
def test_request_rejects_unsafe_or_ambiguous_inputs(payload):
    with pytest.raises(ValidationError):
        WebRequest(**payload)


def test_request_strict_scope_and_optional_features():
    assert WebRequest(mode="live", confirm_live=True, memory_scope="my-team_1").memory_scope
    assert WebRequest(review=False, compression=True).compression
    with pytest.raises(ValidationError):
        WebRequest(mode="live", confirm_live=True, memory_scope="x" * 65)


def test_store_submission_is_idempotent_and_conflicting_key_is_rejected(store):
    idem = key()
    first, made = store.submit(WebRequest(), idem, 8)
    assert made and first["status"] == "queued"
    same, made = store.submit(WebRequest(), idem.upper(), 8)
    assert not made and same == first
    with pytest.raises(ResearchError, match="冲突") as caught:
        store.submit(WebRequest(review=False), idem, 8)
    assert caught.value.code == "web_conflict"
    assert len(store.events(first["id"])) == 1
    assert len(store.list()) == 1


@pytest.mark.parametrize("bad", [None, "", "../abc", "1", "a" * 32, "{" + str(uuid4()) + "}"])
def test_store_idempotency_keys_must_be_canonical_uuid(store, bad):
    with pytest.raises(ResearchError) as caught:
        store.submit(WebRequest(), bad, 8)
    assert caught.value.code == "web_validation"
    assert store.list() == []


def test_store_revalidates_model_construct(store):
    with pytest.raises(ResearchError):
        store.submit(WebRequest.model_construct(mode="live", confirm_live=False), key(), 8)


def test_capacity_replay_is_allowed_even_when_full(store):
    idem = key()
    job, _ = store.submit(WebRequest(), idem, 1)
    with pytest.raises(ResearchError) as caught:
        store.submit(WebRequest(), key(), 1)
    assert caught.value.code == "web_capacity"
    assert store.submit(WebRequest(), idem, 1)[0]["id"] == job["id"]
    assert not store.submit(WebRequest(), idem, 1)[1]
    store.cancel(job["id"])
    assert store.submit(WebRequest(), key(), 1)[1]


def test_events_and_job_survive_reopening_and_are_detached(store):
    job, _ = store.submit(WebRequest(), key(), 8)
    store.begin_next()
    run_id = key()
    store.bind_run(job["id"], run_id)
    store.progress(job["id"], "处理中文😀")
    store.finish(job["id"], "completed", result={"run_id": run_id, "status": "completed"})
    snapshot = store.events(job["id"])
    assert [item["seq"] for item in snapshot] == list(range(1, 6))
    assert [item["kind"] for item in snapshot] == [
        "queued",
        "running",
        "run_created",
        "progress",
        "completed",
    ]
    with WebJobStore(store.path) as reopened:
        assert reopened.events(job["id"]) == snapshot
        assert reopened.get(job["id"])["status"] == "completed"
    snapshot[3]["data"]["message"] = "changed"
    assert store.events(job["id"], after=3)[0]["data"]["message"] == "处理中文😀"
    assert store.events(job["id"], after=5) == []


@pytest.mark.parametrize("value", [-1, 513, True, "2"])
def test_event_cursor_validation(store, value):
    job, _ = store.submit(WebRequest(), key(), 8)
    with pytest.raises(ResearchError) as caught:
        store.events(job["id"], value)
    assert caught.value.code == "web_validation"


@pytest.mark.parametrize("operation", ["get", "cancel", "events"])
def test_unknown_job_is_404(store, operation):
    with pytest.raises(ResearchError) as caught:
        getattr(store, operation)("not-an-existing-job")
    assert caught.value.code == "web_not_found"
    assert caught.value.status_code == 404


def test_progress_cap_preserves_terminal_event_and_drop_count(store):
    job, _ = store.submit(WebRequest(), key(), 8)
    store.begin_next()
    run_id = key()
    store.bind_run(job["id"], run_id)
    for _ in range(MAX_PROGRESS_EVENTS + 10):
        store.progress(job["id"], "工作进度")
    store.cancel(job["id"])
    store.finish(job["id"], "cancelled")
    events = store.events(job["id"])
    assert len(events) <= MAX_EVENTS
    assert sum(item["kind"] == "progress" for item in events) == MAX_PROGRESS_EVENTS
    assert events[-1]["kind"] == "cancelled"
    assert store.get(job["id"])["dropped_events"] == 10


@pytest.mark.parametrize(
    "message",
    [
        "provider sk-secret-value",
        "Bearer abcd",
        "api_key=secret",
        "token:secret",
        "saved C:\\Users\\Admin\\data",
        "saved /home/me/output/report.md",
        "\\\\server\\share",
    ],
)
def test_progress_hides_secret_and_path_details(store, message):
    job, _ = store.submit(WebRequest(), key(), 8)
    store.begin_next()
    store.progress(job["id"], message)
    assert store.events(job["id"])[-1]["data"]["message"] == "研究进度已更新（敏感细节已隐藏）。"


def test_progress_truncates_on_valid_utf8_boundary(store):
    job, _ = store.submit(WebRequest(), key(), 8)
    store.begin_next()
    store.progress(job["id"], "😀" * 1000)
    value = store.events(job["id"])[-1]["data"]["message"]
    assert len(value.encode("utf-8")) == 1000
    assert "�" not in value


def test_result_uses_allowlist_not_arbitrary_model_or_local_content(store):
    job, _ = store.submit(WebRequest(), key(), 8)
    store.begin_next()
    run_id = key()
    store.bind_run(job["id"], run_id)
    result = {
        "status": "completed",
        "run_id": run_id,
        "output_dir": "C:/secret",
        "api_key": "sk-secret",
        "reply": "private output",
        "degraded": True,
        "review": {
            "status": "unresolved",
            "stop_reason": "max_rounds",
            "calls": 5,
            "prompt": "sk-secret",
            "history": ["private"],
            "final_scores": {
                "evidence_support": 50,
                "logical_consistency": 80,
                "citation_quality": 80,
            },
        },
    }
    stored = store.finish(job["id"], "completed", result=result)
    assert stored["result"]["degraded"] is True
    raw = json.dumps(stored)
    assert "private" not in raw and "secret" not in raw and "output_dir" not in raw
    assert stored["result"]["review"]["calls"] == 5


@pytest.mark.parametrize(
    "result",
    [
        None,
        [],
        {},
        {"status": "failed"},
        {"status": "completed", "degraded": "yes"},
        {"status": "completed", "review": {"stop_reason": "C:/secret"}},
        {"status": "completed", "review": {"status": "whatever"}},
        {"status": "completed", "review": {"calls": True}},
        {"status": "completed", "review": {"final_scores": {}}},
    ],
)
def test_invalid_result_does_not_commit_terminal_state(store, result):
    job, _ = store.submit(WebRequest(), key(), 8)
    store.begin_next()
    run_id = key()
    store.bind_run(job["id"], run_id)
    if isinstance(result, dict):
        result = {"run_id": run_id, **result}
    before = store.events(job["id"])
    with pytest.raises(ResearchError):
        store.finish(job["id"], "completed", result=result)
    assert store.get(job["id"])["status"] == "running"
    assert store.events(job["id"]) == before


def test_state_and_event_transaction_is_atomic(store, monkeypatch):
    def fail(*_):
        raise sqlite3.OperationalError("secret path")

    monkeypatch.setattr(store, "_event", fail)
    with pytest.raises(ResearchError) as caught:
        store.submit(WebRequest(), key(), 8)
    assert caught.value.code == "web_io"
    assert "secret" not in str(caught.value)
    assert store.list() == []


def test_terminal_write_failure_rolls_back_status(store, monkeypatch):
    job, _ = store.submit(WebRequest(), key(), 8)
    store.begin_next()
    run_id = key()
    store.bind_run(job["id"], run_id)

    def fail(*_):
        raise sqlite3.OperationalError("disk full")

    monkeypatch.setattr(store, "_event", fail)
    with pytest.raises(ResearchError):
        store.finish(job["id"], "completed", result={"status": "completed", "run_id": run_id})
    assert store.get(job["id"])["status"] == "running"


def test_cancel_is_idempotent_and_never_rewrites_terminal_result(store):
    job, _ = store.submit(WebRequest(), key(), 8)
    first = store.cancel(job["id"])
    assert first["status"] == "cancelled" and first["cancel_requested"]
    assert first == store.cancel(job["id"])
    assert len(store.events(job["id"])) == 2
    job, _ = store.submit(WebRequest(), key(), 8)
    store.begin_next()
    run_id = key()
    store.bind_run(job["id"], run_id)
    done = store.finish(job["id"], "completed", result={"run_id": run_id, "status": "completed"})
    assert store.cancel(job["id"]) == done


def test_cancel_wins_over_late_completion_and_cleanup_error(store):
    job, _ = store.submit(WebRequest(), key(), 8)
    store.begin_next()
    store.cancel(job["id"])
    result = store.finish(job["id"], "failed", error_code="cleanup_failure")
    assert result["status"] == "cancelled"
    assert result["error_code"] is None and result["result"] is None


def test_recovery_only_interrupts_web_jobs_and_their_bound_running_research(store):
    with ResearchStore(store.path) as research:
        unrelated = research.create_run("CLI job remains untouched")
        bound = research.create_run("Web run")
        job, _ = store.submit(WebRequest(), key(), 8)
        store.begin_next()
        store.bind_run(job["id"], bound)
        queued, _ = store.submit(WebRequest(), key(), 8)
        terminal, _ = store.submit(WebRequest(), key(), 8)
        store.cancel(terminal["id"])
        assert store.recover() == 2
        assert store.recover() == 0
        assert research.get_run(unrelated)["status"] == "running"
        assert research.get_run(bound)["status"] == "failed"
        assert research.get_run(bound)["error_code"] == "web_interrupted"
        for target in (job, queued):
            assert store.get(target["id"])["status"] == "interrupted"
            assert store.events(target["id"])[-1]["kind"] == "interrupted"
        assert store.get(terminal["id"])["status"] == "cancelled"


async def test_manager_serial_execution_and_ordered_events(store):
    running = 0
    max_running = 0
    order = []

    async def runner(request, progress, created):
        nonlocal running, max_running
        running += 1
        max_running = max(max_running, running)
        order.append(len(order))
        await asyncio.sleep(0)
        result = await complete(request, progress, created)
        running -= 1
        return result

    manager = WebManager(store, runner)
    await manager.start()
    try:
        jobs = [manager.submit(WebRequest(), key())[0] for _ in range(4)]
        for job in jobs:
            await wait_for_status(manager, job["id"], {"completed"})
        assert max_running == 1
        assert len(order) == 4
        for job in jobs:
            assert manager.events(job["id"])[-1]["kind"] == "completed"
    finally:
        await manager.close()


async def test_queue_capacity_is_separate_from_one_running_slot(store):
    gate = asyncio.Event()
    entered = asyncio.Event()

    async def runner(request, progress, created):
        entered.set()
        await gate.wait()
        return await complete(request, progress, created)

    manager = WebManager(store, runner, max_queued=1)
    await manager.start()
    try:
        first, _ = manager.submit(WebRequest(), key())
        await entered.wait()
        second, _ = manager.submit(WebRequest(), key())
        with pytest.raises(ResearchError) as caught:
            manager.submit(WebRequest(), key())
        assert caught.value.code == "web_capacity"
        assert manager.get(first["id"])["status"] == "running"
        assert manager.get(second["id"])["status"] == "queued"
    finally:
        await manager.close()


async def test_queued_cancellation_never_invokes_runner(store):
    invoked = []

    async def runner(*args):
        invoked.append(True)
        return await complete(*args)

    manager = WebManager(store, runner)
    await manager.start()
    job, _ = manager.submit(WebRequest(), key())
    assert manager.cancel(job["id"])["status"] == "cancelled"
    await asyncio.sleep(0)
    assert invoked == []
    await manager.close()


async def test_running_cancel_waits_for_cleanup_and_repeated_cancel_does_not_interrupt_it(store):
    entered, cleaning, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
    cleaned = False

    async def runner(request, progress, created):
        nonlocal cleaned
        created(key())
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            cleaning.set()
            await release.wait()
            cleaned = True

    manager = WebManager(store, runner)
    await manager.start()
    job, _ = manager.submit(WebRequest(), key())
    await entered.wait()
    assert manager.cancel(job["id"])["status"] == "cancelling"
    await cleaning.wait()
    manager.cancel(job["id"])
    closing = asyncio.create_task(manager.close())
    await asyncio.sleep(0)
    assert not closing.done() and not cleaned
    release.set()
    await closing
    assert cleaned
    assert manager.get(job["id"])["status"] == "cancelled"
    assert [item["kind"] for item in manager.events(job["id"])].count("cancelling") == 1


async def test_cancellation_is_not_replaced_by_cleanup_exception(store):
    entered = asyncio.Event()

    async def runner(request, progress, created):
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            raise RuntimeError("cleanup contains sk-secret")

    manager = WebManager(store, runner)
    await manager.start()
    try:
        job, _ = manager.submit(WebRequest(), key())
        await entered.wait()
        manager.cancel(job["id"])
        result = await wait_for_status(manager, job["id"], {"cancelled"})
        assert result["error_code"] is None
        assert "secret" not in json.dumps(result)
    finally:
        await manager.close()


async def test_runner_exception_is_sanitized_and_next_job_can_run(store):
    called = 0

    async def runner(*args):
        nonlocal called
        called += 1
        if called == 1:
            raise RuntimeError("api_key=secret C:/private")
        return await complete(*args)

    manager = WebManager(store, runner)
    await manager.start()
    try:
        first, _ = manager.submit(WebRequest(), key())
        failed = await wait_for_status(manager, first["id"], {"failed"})
        assert failed["error_code"] == "web_execution"
        assert "secret" not in json.dumps(failed)
        second, _ = manager.submit(WebRequest(), key())
        await wait_for_status(manager, second["id"], {"completed"})
    finally:
        await manager.close()


async def test_invalid_runner_result_is_failed_not_stuck_running(store):
    async def runner(*_):
        return {"status": "completed"}

    manager = WebManager(store, runner)
    await manager.start()
    try:
        job, _ = manager.submit(WebRequest(), key())
        result = await wait_for_status(manager, job["id"], {"failed"})
        assert result["error_code"] == "web_execution"
        assert manager._healthy
    finally:
        await manager.close()


async def test_audit_failure_cancels_even_if_runner_swallows_callback_error(store, monkeypatch):
    after_cancel_point = []

    def fail(*_):
        raise ResearchError("web_io", "safe")

    monkeypatch.setattr(store, "progress", fail)

    async def runner(request, progress, created):
        created(key())
        try:
            progress("保存审计")
        except Exception:
            pass  # Existing CLI progress intentionally tolerates observer errors.
        await asyncio.sleep(0)
        after_cancel_point.append("would call model")
        return {}

    manager = WebManager(store, runner)
    await manager.start()
    try:
        job, _ = manager.submit(WebRequest(), key())
        result = await wait_for_status(manager, job["id"], {"failed"})
        assert result["error_code"] == "web_io"
        assert after_cancel_point == []
        with pytest.raises(ResearchError) as caught:
            manager.submit(WebRequest(), key())
        assert caught.value.code == "web_unavailable"
    finally:
        await manager.close()


async def test_start_recovers_without_resubmitting_old_jobs(store):
    stale, _ = store.submit(WebRequest(), key(), 8)
    called = []

    async def runner(*args):
        called.append(True)
        return await complete(*args)

    manager = WebManager(store, runner)
    await manager.start()
    try:
        await asyncio.sleep(0)
        assert manager.get(stale["id"])["status"] == "interrupted"
        assert called == []
    finally:
        await manager.close()


async def test_single_instance_lock_and_release(store):
    first = WebManager(store, complete)
    await first.start()
    with WebJobStore(store.path) as other_store:
        second = WebManager(other_store, complete)
        with pytest.raises(ResearchError) as caught:
            await second.start()
        assert caught.value.code == "web_locked"
        await second.close()
        await first.close()
        third = WebManager(other_store, complete)
        await third.start()
        await third.close()


async def test_os_lock_also_blocks_a_second_process(store):
    program = """
import asyncio
import sys
from pathlib import Path
from deep_research.errors import ResearchError
from deep_research.web_jobs import WebJobStore, WebManager
async def runner(*args):
    return {}
async def check():
    with WebJobStore(Path(sys.argv[1])) as store:
        manager = WebManager(store, runner)
        try:
            await manager.start()
        except ResearchError as error:
            print(error.code)
        else:
            await manager.close()
            print('unexpectedly-started')
asyncio.run(check())
"""
    manager = WebManager(store, complete)
    await manager.start()
    try:
        process = await asyncio.to_thread(
            subprocess.run,
            [sys.executable, "-c", program, str(store.path)],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        assert process.returncode == 0
        assert process.stdout.strip() == "web_locked"
    finally:
        await manager.close()


async def test_closed_or_not_started_manager_cannot_accept_work(store):
    manager = WebManager(store, complete)
    with pytest.raises(ResearchError):
        manager.submit(WebRequest(), key())
    await manager.start()
    await manager.close()
    await manager.close()
    with pytest.raises(ResearchError):
        manager.submit(WebRequest(), key())


async def test_read_storage_failure_disables_submission(store):
    manager = WebManager(store, complete)
    await manager.start()
    store.close()
    with pytest.raises(ResearchError):
        manager.list()
    with pytest.raises(ResearchError) as caught:
        manager.submit(WebRequest(), key())
    assert caught.value.code == "web_unavailable"
    await manager.close()


def test_store_close_is_idempotent_and_errors_are_sanitized(store):
    store.close()
    store.close()
    with pytest.raises(ResearchError) as caught:
        store.list()
    assert caught.value.code == "web_closed"


@pytest.mark.parametrize("limit", [0, -1, 101, True, "8"])
def test_queue_and_listing_limits_are_strict(store, limit):
    for operation in (
        lambda: WebManager(store, complete, max_queued=limit),
        lambda: store.submit(WebRequest(), key(), limit),
        lambda: store.list(limit),
    ):
        with pytest.raises(ResearchError) as caught:
            operation()
        assert caught.value.code == "web_validation"


def test_separate_research_failure_does_not_overwrite_completed_research_during_recovery(store):
    with ResearchStore(store.path) as research:
        run_id = research.create_run("Bound already-failed research")
        research.fail_run(run_id, "source_failed")
        job, _ = store.submit(WebRequest(), key(), 8)
        store.begin_next()
        store.bind_run(job["id"], run_id)
        store.cancel(job["id"])
        assert store.recover() == 1
        assert research.get_run(run_id)["error_code"] == "source_failed"
        assert store.get(job["id"])["status"] == "interrupted"


def test_recovery_is_atomic_including_bound_research(store, monkeypatch):
    with ResearchStore(store.path) as research:
        run_id = research.create_run("Bound research")
        job, _ = store.submit(WebRequest(), key(), 8)
        store.begin_next()
        store.bind_run(job["id"], run_id)

        def fail(*_):
            raise sqlite3.OperationalError("cannot save recovery event")

        monkeypatch.setattr(store, "_event", fail)
        with pytest.raises(ResearchError):
            store.recover()
        assert store.get(job["id"])["status"] == "running"
        assert research.get_run(run_id)["status"] == "running"


def test_job_run_binding_cannot_be_changed_or_reused(store):
    first, _ = store.submit(WebRequest(), key(), 8)
    store.begin_next()
    run_id = key()
    store.bind_run(first["id"], run_id)
    with pytest.raises(ResearchError) as caught:
        store.bind_run(first["id"], key())
    assert caught.value.code == "web_conflict"
    second, _ = store.submit(WebRequest(), key(), 8)
    store.begin_next()
    with pytest.raises(ResearchError):
        store.bind_run(second["id"], run_id)
    assert store.get(second["id"])["run_id"] is None


@pytest.mark.parametrize("status", ["saved", "skipped", "failed"])
def test_memory_publication_result_remains_visible_in_safe_summary(store, status):
    job, _ = store.submit(WebRequest(), key(), 8)
    store.begin_next()
    run_id = key()
    store.bind_run(job["id"], run_id)
    memory = {
        "status": status,
        "code": "memory_io",
        "reason": "degraded_run",
        "added_items": 1,
        "added_provenance": 2,
        "deduplicated": 3,
        "skipped_recalled": 4,
        "created_conflicts": 5,
        "created_candidates": 6,
        "scope": "private-project",
        "output_dir": "C:/secret",
        "arbitrary": "sk-secret",
    }
    result = {"run_id": run_id, "status": "completed", "memory_write": memory}
    job = store.finish(job["id"], "completed", result=result)
    public = job["result"]["memory_write"]
    assert public["status"] == status and public["code"] == "memory_io"
    assert public["added_items"] == 1 and public["created_candidates"] == 6
    assert "private" not in json.dumps(public) and "secret" not in json.dumps(public)


def test_unsafe_memory_error_detail_is_mapped_to_fixed_code(store):
    job, _ = store.submit(WebRequest(), key(), 8)
    store.begin_next()
    run_id = key()
    store.bind_run(job["id"], run_id)
    result = {
        "run_id": run_id,
        "status": "completed",
        "memory_write": {"status": "failed", "code": "C:/secret provider response"},
    }
    job = store.finish(job["id"], "completed", result=result)
    assert job["result"]["memory_write"]["code"] == "memory_unexpected"


def test_useful_execution_and_compression_summary_is_preserved(store):
    job, _ = store.submit(WebRequest(), key(), 8)
    store.begin_next()
    run_id = key()
    store.bind_run(job["id"], run_id)
    result = {
        "status": "completed",
        "run_id": run_id,
        "engine": "dag",
        "source_count": 2,
        "evidence_count": 4,
        "model_call_count": 5,
        "replan_count": 0,
        "reasons": ["review_unresolved"],
        "report_kind": "model_report",
        "counters": {"model": 5, "search": 2, "fetch": 2},
        "compression": {
            "status": "completed",
            "encoder_kind": "semantic",
            "report_evidence_bytes": 2048,
            "model_path": "C:/secret",
        },
    }
    stored = store.finish(job["id"], "completed", result=result)["result"]
    assert stored["counters"] == result["counters"]
    assert stored["reasons"] == result["reasons"]
    assert stored["model_call_count"] == 5
    assert stored["compression"] == {
        "status": "completed",
        "encoder_kind": "semantic",
        "report_evidence_bytes": 2048,
    }


@pytest.mark.parametrize(
    "bad",
    [
        {"source_count": True},
        {"engine": "C:/secret"},
        {"report_kind": "anything"},
        {"reasons": ["secret path"]},
        {"counters": {"api_key": "secret"}},
        {"counters": {"model": -1, "search": 1, "fetch": 1}},
        {"compression": {"status": "completed", "encoder_kind": "secret model path"}},
        {"compression": {"status": "completed", "report_evidence_bytes": "secret"}},
        {"memory_write": {"status": "failed", "added_items": -1}},
    ],
)
def test_summary_fields_cannot_smuggle_unbounded_details(store, bad):
    job, _ = store.submit(WebRequest(), key(), 8)
    store.begin_next()
    run_id = key()
    store.bind_run(job["id"], run_id)
    with pytest.raises(ResearchError):
        store.finish(
            job["id"], "completed", result={"status": "completed", "run_id": run_id, **bad}
        )
    assert store.get(job["id"])["status"] == "running"


async def test_pending_shutdown_does_not_depend_on_recent_terminal_list(store):
    manager = WebManager(store, complete)
    await manager.start()
    queued, _ = manager.submit(WebRequest(), key())
    # Many newer terminal records must not hide the old queued item at shutdown.
    for _ in range(105):
        other, _ = store.submit(WebRequest(), key(), 8)
        store.cancel(other["id"])
    await manager.close()
    assert store.get(queued["id"])["status"] == "cancelled"


async def test_failed_recovery_releases_instance_lock(store, monkeypatch):
    original = store.recover

    def fail():
        raise ResearchError("web_io", "safe")

    monkeypatch.setattr(store, "recover", fail)
    broken = WebManager(store, complete)
    with pytest.raises(ResearchError):
        await broken.start()
    monkeypatch.setattr(store, "recover", original)
    replacement = WebManager(store, complete)
    await replacement.start()
    await replacement.close()


async def test_failed_terminal_audit_stops_next_job_and_preserves_uncertain_state(
    store, monkeypatch
):
    original_event = store._event
    called = []

    def event(db, job_id, kind, data, now):
        if kind == "completed":
            raise sqlite3.OperationalError("disk full")
        original_event(db, job_id, kind, data, now)

    monkeypatch.setattr(store, "_event", event)

    async def runner(*args):
        called.append(True)
        return await complete(*args)

    manager = WebManager(store, runner)
    await manager.start()
    first, _ = manager.submit(WebRequest(), key())
    second, _ = manager.submit(WebRequest(), key())
    await manager._worker
    assert len(called) == 1
    assert manager.get(first["id"])["status"] == "running"
    assert manager.get(second["id"])["status"] == "queued"
    with pytest.raises(ResearchError):
        manager.submit(WebRequest(), key())
    await manager.close()


async def test_binding_audit_failure_prevents_runner_from_reaching_first_await(store, monkeypatch):
    reached = []

    def fail(*_):
        raise ResearchError("web_io", "safe")

    monkeypatch.setattr(store, "bind_run", fail)

    async def runner(request, progress, created):
        created(key())
        reached.append("would call model")
        await asyncio.sleep(0)

    manager = WebManager(store, runner)
    await manager.start()
    try:
        job, _ = manager.submit(WebRequest(), key())
        result = await wait_for_status(manager, job["id"], {"failed"})
        assert result["error_code"] == "web_io"
        assert reached == []
        assert not manager._healthy
    finally:
        await manager.close()


def test_corrupt_job_or_events_fail_closed_instead_of_exposing_raw_payload(store):
    job, _ = store.submit(WebRequest(), key(), 8)
    store._db().execute(
        "UPDATE web_jobs SET request_json='secret invalid' WHERE id=?", (job["id"],)
    )
    with pytest.raises(ResearchError) as caught:
        store.get(job["id"])
    assert caught.value.code == "web_io" and "secret" not in str(caught.value)
    store._db().execute(
        "UPDATE web_jobs SET request_json=? WHERE id=?", (WebRequest().model_dump_json(), job["id"])
    )
    store._db().execute(
        "UPDATE web_events SET data_json='secret invalid' WHERE job_id=?", (job["id"],)
    )
    with pytest.raises(ResearchError) as caught:
        store.events(job["id"])
    assert caught.value.code == "web_io" and "secret" not in str(caught.value)


@pytest.mark.parametrize(
    "bad",
    [
        {"status": []},
        {"review": {"status": {}}},
        {"report_kind": []},
        {"compression": {"status": []}},
        {"compression": {"status": "completed", "encoder_kind": []}},
        {"memory_write": {"status": []}},
    ],
)
async def test_malformed_result_does_not_kill_worker(store, bad):
    async def runner(request, progress, created):
        run_id = key()
        created(run_id)
        return {"status": "completed", "run_id": run_id, **bad}

    manager = WebManager(store, runner)
    await manager.start()
    try:
        job, _ = manager.submit(WebRequest(), key())
        failed = await wait_for_status(manager, job["id"], {"failed"})
        assert failed["error_code"] == "web_execution"
        assert manager._healthy and not manager._worker.done()
    finally:
        await manager.close()


async def test_wrong_synchronous_adapter_fails_job_not_entire_worker(store):
    def runner(*_):
        raise RuntimeError("implementation error")

    manager = WebManager(store, runner)
    await manager.start()
    try:
        job, _ = manager.submit(WebRequest(), key())
        await wait_for_status(manager, job["id"], {"failed"})
        assert manager._healthy and not manager._worker.done()
    finally:
        await manager.close()
