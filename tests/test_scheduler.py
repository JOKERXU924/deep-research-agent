"""Deterministic scheduler checks, using events rather than speed comparisons."""

import asyncio
import json
from dataclasses import replace

import pytest

from deep_research.errors import ModelError, ResearchError
from deep_research.scheduler import (
    TERMINAL_STATES,
    Scheduler,
    SchedulerAuditError,
    SchedulerValidationError,
    TaskContext,
    TaskRetryableError,
    TaskSpec,
    TaskState,
)


async def value(context):
    return context.attempt


async def wait_for(event: asyncio.Event):
    async with asyncio.timeout(2):
        await event.wait()


def states(result, task_id):
    return [event["state"] for event in result.events if event["task_id"] == task_id]


async def test_diamond_dependency_results_and_monotonic_audit_events():
    seen = []

    async def branch(context):
        assert context.results == {"root": 1}
        assert context.states == {"root": TaskState.SUCCEEDED}
        with pytest.raises(TypeError):
            context.results["root"] = 10
        with pytest.raises(TypeError):
            context.states["root"] = TaskState.FAILED
        return 2

    async def join(context):
        assert context.results == {"left": 2, "right": 2}
        assert "root" not in context.states
        return sum(context.results.values())

    scheduler = Scheduler(on_event=seen.append)
    result = await scheduler.run(
        [
            TaskSpec("join", join, ("left", "right")),
            TaskSpec("right", branch, ("root",)),
            TaskSpec("left", branch, ("root",)),
            TaskSpec("root", value),
        ]
    )
    assert result.results == {"root": 1, "right": 2, "left": 2, "join": 4}
    assert result.events == seen
    assert [item["seq"] for item in seen] == list(range(1, len(seen) + 1))
    assert [item["state"] for item in seen[:4]] == ["pending"] * 4
    for record in result.records.values():
        assert record.state is TaskState.SUCCEEDED
        assert record.attempts == 1
        assert record.started_at is not None
        assert record.finished_at is not None
        assert states(result, record.id) == ["pending", "ready", "running", "succeeded"]
        json.dumps(record.to_dict())
    assert scheduler.last_result is result
    assert not result.timed_out


async def test_semaphore_limits_actual_execution_and_waiting_does_not_time_out():
    first_started = asyncio.Event()
    first_release = asyncio.Event()
    second_started = asyncio.Event()
    third_started = asyncio.Event()
    active = 0
    peak = 0

    async def first(context):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        first_started.set()
        try:
            await first_release.wait()
        finally:
            active -= 1
        return "first"

    async def second(context):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        second_started.set()
        active -= 1
        return "second"

    async def third(context):
        third_started.set()
        return "third"

    scheduler = Scheduler(concurrency=1)
    task = asyncio.create_task(
        scheduler.run(
            [
                TaskSpec("first", first),
                TaskSpec("second", second, timeout_seconds=0.001),
                TaskSpec("third", third),
            ]
        )
    )
    await wait_for(first_started)
    # Waiting outside the slot must exceed second's timeout without consuming it.
    await asyncio.sleep(0.01)
    assert not second_started.is_set()
    assert not third_started.is_set()
    first_release.set()
    result = await task
    assert peak == 1
    assert active == 0
    assert second_started.is_set() and third_started.is_set()
    assert all(record.state is TaskState.SUCCEEDED for record in result.records.values())


async def test_independent_tasks_really_overlap():
    entered = [asyncio.Event(), asyncio.Event()]
    release = asyncio.Event()
    third_entered = asyncio.Event()

    async def left(context):
        entered[0].set()
        await release.wait()

    async def right(context):
        entered[1].set()
        await release.wait()

    async def third(context):
        third_entered.set()

    task = asyncio.create_task(
        Scheduler(concurrency=2).run(
            [TaskSpec("left", left), TaskSpec("right", right), TaskSpec("third", third)]
        )
    )
    await wait_for(entered[0])
    await wait_for(entered[1])
    assert not third_entered.is_set()
    release.set()
    result = await task
    assert third_entered.is_set()
    assert len(result.results) == 3


async def test_failed_dependency_skips_descendants_but_all_done_can_summarize():
    async def fail(context):
        raise ValueError("secret raw payload")

    async def all_done(context):
        assert context.states == {"bad": TaskState.FAILED, "good": TaskState.SUCCEEDED}
        assert context.results == {"good": 1}
        return "partial"

    result = await Scheduler().run(
        [
            TaskSpec("last", value, ("blocked",)),
            TaskSpec("blocked", value, ("bad",)),
            TaskSpec("summary", all_done, ("bad", "good"), allow_failed_dependencies=True),
            TaskSpec("bad", fail),
            TaskSpec("good", value),
        ]
    )
    assert result.records["bad"].error_code == "unexpected"
    assert result.records["blocked"].state is TaskState.SKIPPED
    assert result.records["last"].state is TaskState.SKIPPED
    assert result.records["last"].attempts == 0
    assert result.results == {"good": 1, "summary": "partial"}
    assert "secret" not in json.dumps(result.events)


@pytest.mark.parametrize(
    "error",
    [
        TaskRetryableError(),
        ModelError("rate_limit", "private"),
        ModelError("network", "private"),
        ModelError("upstream", "private"),
        ResearchError("retrieval_network", "private"),
        ResearchError("retrieval_timeout", "private"),
    ],
)
async def test_retry_is_explicit_bounded_and_attempt_visible(error):
    attempts = []

    async def flaky(context):
        attempts.append(context.attempt)
        if context.attempt == 1:
            raise error
        return "recovered"

    result = await Scheduler().run([TaskSpec("flaky", flaky, max_attempts=2)])
    assert attempts == [1, 2]
    assert result.results == {"flaky": "recovered"}
    assert result.records["flaky"].attempts == 2
    assert result.records["flaky"].error_code is None
    assert states(result, "flaky") == [
        "pending",
        "ready",
        "running",
        "retry_wait",
        "ready",
        "running",
        "succeeded",
    ]


@pytest.mark.parametrize(
    "error",
    [
        ModelError("authentication", "private key"),
        ModelError("billing", "private account"),
        ModelError("invalid_schema", "raw payload"),
        ValueError("programmer bug"),
        ResearchError("unexpected-secret-code", "private"),
    ],
)
async def test_permanent_errors_never_retry(error):
    async def fail(context):
        raise error

    result = await Scheduler().run([TaskSpec("bad", fail, max_attempts=3)])
    assert result.records["bad"].attempts == 1
    assert result.records["bad"].state is TaskState.FAILED
    assert "retry_wait" not in states(result, "bad")
    assert "secret" not in json.dumps(result.events)


async def test_retry_exhaustion_and_default_no_retry():
    async def fail(context):
        raise TaskRetryableError()

    result = await Scheduler().run(
        [TaskSpec("default", fail), TaskSpec("bounded", fail, max_attempts=3)]
    )
    assert result.records["default"].attempts == 1
    assert result.records["bounded"].attempts == 3
    assert all(record.state is TaskState.FAILED for record in result.records.values())


async def test_retry_delay_releases_concurrency_slot():
    retry_announced = asyncio.Event()
    other_finished = asyncio.Event()

    def audit(event):
        if event["state"] == "retry_wait":
            retry_announced.set()

    async def retry(context):
        if context.attempt == 1:
            raise TaskRetryableError()
        assert other_finished.is_set()

    async def other(context):
        assert retry_announced.is_set()
        other_finished.set()

    result = await Scheduler(concurrency=1, on_event=audit).run(
        [TaskSpec("retry", retry, max_attempts=2), TaskSpec("other", other)]
    )
    assert all(record.state is TaskState.SUCCEEDED for record in result.records.values())


async def test_task_timeout_can_retry_but_is_terminal_after_limit():
    cleaned = []

    async def hang(context):
        try:
            await asyncio.Event().wait()
        finally:
            cleaned.append(context.attempt)

    result = await Scheduler().run([TaskSpec("hang", hang, timeout_seconds=0.01, max_attempts=2)])
    assert cleaned == [1, 2]
    assert result.records["hang"].state is TaskState.TIMED_OUT
    assert result.records["hang"].error_code == "timeout"
    assert result.records["hang"].attempts == 2
    assert not result.timed_out


async def test_global_deadline_keeps_successes_and_drains_children():
    started = asyncio.Event()
    cleaned = asyncio.Event()

    async def hang(context):
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cleaned.set()

    result = await Scheduler(concurrency=1).run(
        [TaskSpec("good", value), TaskSpec("hang", hang), TaskSpec("queued", value)],
        timeout_seconds=0.03,
    )
    assert started.is_set() and cleaned.is_set()
    assert result.timed_out
    assert result.results == {"good": 1}
    assert result.records["hang"].state is TaskState.CANCELLED
    assert result.records["hang"].error_code == "global_timeout"
    assert result.records["queued"].attempts == 0
    assert all(record.state in TERMINAL_STATES for record in result.records.values())


@pytest.mark.parametrize("audit_state", ["pending", "ready", "running"])
async def test_audit_crossing_global_deadline_never_starts_task(monkeypatch, audit_state):
    loop = asyncio.get_running_loop()
    clock = [loop.time()]
    calls = []

    def audit(event):
        if event["state"] == audit_state:
            # Advance the monotonic clock without wall-clock sleeps or timing races.
            clock[0] += 10

    async def work(context):
        calls.append(True)

    with monkeypatch.context() as patch:
        patch.setattr(loop, "time", lambda: clock[0])
        result = await Scheduler(on_event=audit).run([TaskSpec("only", work)], timeout_seconds=1)
    assert calls == []
    assert result.timed_out
    assert result.records["only"].state is TaskState.CANCELLED
    assert result.records["only"].error_code == "global_timeout"


async def test_external_cancellation_is_preserved_and_last_result_is_available():
    started = asyncio.Event()
    cleaned = asyncio.Event()

    async def hang(context):
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cleaned.set()

    scheduler = Scheduler(concurrency=1)
    task = asyncio.create_task(scheduler.run([TaskSpec("hang", hang), TaskSpec("queued", value)]))
    await wait_for(started)
    task.cancel("user requested stop")
    with pytest.raises(asyncio.CancelledError, match="user requested stop"):
        await task
    assert cleaned.is_set()
    assert scheduler.last_result is not None
    assert not scheduler.last_result.timed_out
    assert all(
        record.state is TaskState.CANCELLED for record in scheduler.last_result.records.values()
    )


async def test_repeated_parent_cancellation_does_not_orphan_cleanup():
    started = asyncio.Event()
    cleaning = asyncio.Event()
    may_finish = asyncio.Event()
    cleaned = asyncio.Event()

    async def hang(context):
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cleaning.set()
            await may_finish.wait()
            cleaned.set()

    scheduler = Scheduler()
    task = asyncio.create_task(scheduler.run([TaskSpec("hang", hang)]))
    await wait_for(started)
    task.cancel("first cancellation")
    await wait_for(cleaning)
    task.cancel("second cancellation")
    await asyncio.sleep(0)
    assert not task.done()
    may_finish.set()
    with pytest.raises(asyncio.CancelledError, match="first cancellation"):
        await task
    assert cleaned.is_set()


async def test_child_cancellation_is_not_parent_cancellation():
    async def cancel_self(context):
        raise asyncio.CancelledError("child chose to stop")

    result = await Scheduler().run(
        [
            TaskSpec("self", cancel_self),
            TaskSpec("dependent", value, ("self",)),
            TaskSpec("other", value),
        ]
    )
    assert result.records["self"].state is TaskState.CANCELLED
    assert result.records["dependent"].state is TaskState.SKIPPED
    assert result.results == {"other": 1}


@pytest.mark.parametrize("state", ["pending", "ready", "running", "succeeded"])
async def test_audit_failure_stops_schedule_and_never_exposes_callback_error(state):
    called = []

    def fail_audit(event):
        if event["state"] == state:
            raise OSError("SECRET database connection")

    async def work(context):
        called.append(True)
        return 1

    scheduler = Scheduler(on_event=fail_audit)
    with pytest.raises(SchedulerAuditError) as captured:
        await scheduler.run([TaskSpec("task", work)])
    assert "SECRET" not in str(captured.value)
    assert scheduler.last_result is not None
    assert all(record.state in TERMINAL_STATES for record in scheduler.last_result.records.values())
    assert bool(called) == (state == "succeeded")


async def test_cancellation_audit_failure_does_not_replace_cancellation():
    started = asyncio.Event()

    def audit(event):
        if event["state"] == "cancelled":
            raise OSError("SECRET")

    async def hang(context):
        started.set()
        await asyncio.Event().wait()

    scheduler = Scheduler(on_event=audit)
    task = asyncio.create_task(scheduler.run([TaskSpec("hang", hang)]))
    await wait_for(started)
    task.cancel("external cancellation")
    with pytest.raises(asyncio.CancelledError, match="external cancellation"):
        await task
    assert scheduler.last_result.records["hang"].state is TaskState.CANCELLED


async def test_deadline_cleanup_audit_failure_is_not_reported_as_successful_timeout():
    cleaned = asyncio.Event()

    def audit(event):
        if event["state"] == "cancelled":
            raise OSError("audit unavailable")

    async def hang(context):
        try:
            await asyncio.Event().wait()
        finally:
            cleaned.set()

    scheduler = Scheduler(on_event=audit)
    with pytest.raises(SchedulerAuditError):
        await scheduler.run([TaskSpec("hang", hang)], timeout_seconds=0.01)
    assert cleaned.is_set()
    assert scheduler.last_result.timed_out
    assert scheduler.last_result.records["hang"].state is TaskState.CANCELLED


async def test_external_cancel_during_retry_wait_cannot_start_another_attempt():
    retrying = asyncio.Event()
    calls = []

    def audit(event):
        if event["state"] == "retry_wait":
            retrying.set()

    async def fail(context):
        calls.append(context.attempt)
        raise TaskRetryableError()

    scheduler = Scheduler(on_event=audit)
    task = asyncio.create_task(
        scheduler.run([TaskSpec("retry", fail, max_attempts=3, retry_delay_seconds=10)])
    )
    await wait_for(retrying)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert calls == [1]
    assert scheduler.last_result.records["retry"].state is TaskState.CANCELLED
    assert states(scheduler.last_result, "retry")[-2:] == ["retry_wait", "cancelled"]


async def test_audit_failure_blocks_queued_tasks_and_drains_started_sibling():
    started = asyncio.Event()
    cleaned = asyncio.Event()
    called = []

    def audit(event):
        if event["task_id"] == "bad" and event["state"] == "running":
            raise OSError("audit unavailable")

    async def hang(context):
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cleaned.set()

    async def unexpected_work(context):
        called.append(True)

    scheduler = Scheduler(concurrency=3, on_event=audit)
    with pytest.raises(SchedulerAuditError):
        await scheduler.run(
            [
                TaskSpec("started", hang),
                TaskSpec("bad", unexpected_work),
                TaskSpec("queued", unexpected_work),
            ]
        )
    assert started.is_set() and cleaned.is_set()
    assert called == []
    assert all(record.state in TERMINAL_STATES for record in scheduler.last_result.records.values())


@pytest.mark.parametrize("code", ["store_write", "dag_invalid_plan", "retrieval_http"])
async def test_application_error_categories_are_preserved_without_payload(code):
    async def fail(context):
        raise ResearchError(code, "PRIVATE raw response")

    result = await Scheduler().run([TaskSpec("bad", fail)])
    assert result.records["bad"].error_code == code
    assert "PRIVATE" not in json.dumps(result.events)


async def test_async_audit_callback_is_rejected_without_unawaited_warning():
    async def audit(event):
        pass

    with pytest.raises(SchedulerAuditError):
        await Scheduler(on_event=audit).run([TaskSpec("a", value)])


async def test_audit_callback_cannot_mutate_recorded_event():
    def audit(event):
        event["state"] = "tampered"

    result = await Scheduler(on_event=audit).run([TaskSpec("a", value)])
    assert states(result, "a") == ["pending", "ready", "running", "succeeded"]


async def test_scheduler_rejects_concurrent_reuse_but_allows_sequential_runs():
    started = asyncio.Event()
    release = asyncio.Event()

    async def hang(context):
        started.set()
        await release.wait()

    scheduler = Scheduler()
    first = asyncio.create_task(scheduler.run([TaskSpec("first", hang)]))
    await wait_for(started)
    with pytest.raises(SchedulerValidationError, match="active run"):
        await scheduler.run([TaskSpec("second", value)])
    release.set()
    result1 = await first
    result2 = await scheduler.run([TaskSpec("second", value)])
    assert set(result1.records) == {"first"}
    assert set(result2.records) == {"second"}
    assert result2.events[0]["seq"] == 1


@pytest.mark.parametrize("concurrency", [0, -1, True, False, 1.5, "2", 129, None])
def test_invalid_concurrency_rejected(concurrency):
    with pytest.raises(SchedulerValidationError):
        Scheduler(concurrency=concurrency)


@pytest.mark.parametrize("invalid_timeout", [0, -1, True, float("nan"), float("inf"), "2", None])
async def test_invalid_global_timeout_does_not_execute_or_audit(invalid_timeout):
    seen = []
    scheduler = Scheduler(on_event=seen.append)
    with pytest.raises(SchedulerValidationError):
        await scheduler.run([TaskSpec("a", value)], timeout_seconds=invalid_timeout)
    assert seen == []
    assert scheduler.last_result is None


@pytest.mark.parametrize(
    "change",
    [
        {"id": ""},
        {"id": " "},
        {"id": "a" * 101},
        {"id": 123},
        {"run": None},
        {"dependencies": ["a"]},
        {"dependencies": ("x", "x")},
        {"dependencies": (123,)},
        {"timeout_seconds": 0},
        {"timeout_seconds": True},
        {"timeout_seconds": float("nan")},
        {"timeout_seconds": float("inf")},
        {"max_attempts": 0},
        {"max_attempts": 11},
        {"max_attempts": True},
        {"max_attempts": 1.5},
        {"retry_delay_seconds": -1},
        {"retry_delay_seconds": True},
        {"retry_delay_seconds": float("nan")},
        {"retry_delay_seconds": float("inf")},
        {"allow_failed_dependencies": 1},
    ],
)
async def test_invalid_task_is_rejected_before_any_work(change):
    seen = []
    with pytest.raises(SchedulerValidationError):
        await Scheduler(on_event=seen.append).run(
            [TaskSpec("good", value), replace(TaskSpec("bad", value), **change)]
        )
    assert seen == []


@pytest.mark.parametrize(
    "tasks",
    [
        [],
        [None],
        [TaskSpec("same", value), TaskSpec("same", value)],
        [TaskSpec("a", value, ("missing",))],
        [TaskSpec("a", value, ("a",))],
        [TaskSpec("a", value, ("b",)), TaskSpec("b", value, ("a",))],
        [TaskSpec(str(index), value) for index in range(129)],
    ],
)
async def test_invalid_graph_never_executes_or_audits(tasks):
    seen = []
    with pytest.raises(SchedulerValidationError):
        await Scheduler(on_event=seen.append).run(tasks)
    assert seen == []


async def test_exact_graph_size_limit_is_accepted():
    result = await Scheduler().run([TaskSpec(str(index), value) for index in range(128)])
    assert len(result.results) == 128


def test_exactly_nine_states_and_context_is_frozen():
    assert len(TaskState) == 9
    context = TaskContext(results={}, states={}, attempt=1)
    with pytest.raises(AttributeError):
        context.attempt = 2
