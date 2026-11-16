"""Bounded, auditable DAG execution for cooperative asynchronous tasks.

Timeouts request cancellation; they cannot kill blocking code or a coroutine that
swallows cancellation. Task functions must release resources in ``finally`` and
propagate ``CancelledError``. Callbacks are synchronous durable audit writes, not
best-effort UI notifications. Their failure stops the entire schedule.
"""

from __future__ import annotations

import asyncio
import inspect
import math
import re
from collections import deque
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from enum import StrEnum
from types import MappingProxyType
from typing import Any

from deep_research.errors import ResearchError


class TaskState(StrEnum):
    PENDING = "pending"
    READY = "ready"
    RUNNING = "running"
    RETRY_WAIT = "retry_wait"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    TIMED_OUT = "timed_out"
    SKIPPED = "skipped"
    CANCELLED = "cancelled"


TERMINAL_STATES = frozenset(
    {
        TaskState.SUCCEEDED,
        TaskState.FAILED,
        TaskState.TIMED_OUT,
        TaskState.SKIPPED,
        TaskState.CANCELLED,
    }
)
_TRANSITIONS = {
    TaskState.PENDING: {TaskState.READY, TaskState.SKIPPED, TaskState.CANCELLED},
    TaskState.READY: {TaskState.RUNNING, TaskState.CANCELLED},
    TaskState.RUNNING: {
        TaskState.SUCCEEDED,
        TaskState.FAILED,
        TaskState.TIMED_OUT,
        TaskState.RETRY_WAIT,
        TaskState.CANCELLED,
    },
    TaskState.RETRY_WAIT: {TaskState.READY, TaskState.CANCELLED},
}
_RETRYABLE_CODES = frozenset(
    {"network", "upstream", "rate_limit", "retrieval_network", "retrieval_timeout"}
)
_ERROR_CODE = re.compile(r"[a-z][a-z0-9_]{0,63}\Z")


class SchedulerValidationError(ValueError):
    """The complete graph is rejected before any task or audit callback runs."""


class SchedulerAuditError(RuntimeError):
    """Durable state recording failed; raw callback errors are not exposed."""

    def __init__(self) -> None:
        super().__init__("Scheduler audit recording failed; execution was stopped.")


class TaskRetryableError(Exception):
    """A task explicitly declares a transient failure, without public error text."""

    def __init__(self, code: str = "retryable_error") -> None:
        super().__init__("A retryable task failure occurred.")
        self.code = code if code in _RETRYABLE_CODES else "retryable_error"


@dataclass(frozen=True)
class TaskContext:
    results: Mapping[str, Any]
    states: Mapping[str, TaskState]
    attempt: int


@dataclass(frozen=True)
class TaskSpec:
    id: str
    run: Callable[[TaskContext], Awaitable[Any]]
    dependencies: tuple[str, ...] = ()
    timeout_seconds: float = 15
    max_attempts: int = 1
    retry_delay_seconds: float = 0
    allow_failed_dependencies: bool = False


@dataclass
class TaskRecord:
    id: str
    state: TaskState = TaskState.PENDING
    attempts: int = 0
    error_code: str | None = None
    started_at: str | None = None
    finished_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "state": self.state.value,
            "attempts": self.attempts,
            "error_code": self.error_code,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
        }


@dataclass(frozen=True)
class ScheduleResult:
    records: dict[str, TaskRecord]
    results: dict[str, Any]
    events: list[dict[str, Any]]
    timed_out: bool = False


def _number(value: object, *, positive: bool) -> bool:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return False
    try:
        return math.isfinite(value) and (value > 0 if positive else value >= 0)
    except OverflowError:
        return False


def _validate(tasks: list[TaskSpec], timeout_seconds: float) -> list[TaskSpec]:
    if not _number(timeout_seconds, positive=True):
        raise SchedulerValidationError("The global timeout must be finite and positive.")
    if not isinstance(tasks, list) or not 1 <= len(tasks) <= 128:
        raise SchedulerValidationError("A schedule must contain between 1 and 128 tasks.")
    specs = list(tasks)
    by_id: dict[str, TaskSpec] = {}
    for spec in specs:
        if not isinstance(spec, TaskSpec):
            raise SchedulerValidationError("Every task must be a TaskSpec.")
        if not isinstance(spec.id, str) or not spec.id.strip() or len(spec.id) > 100:
            raise SchedulerValidationError("Task IDs must contain 1 to 100 characters.")
        if spec.id in by_id:
            raise SchedulerValidationError("Duplicate task IDs are not allowed.")
        if not callable(spec.run):
            raise SchedulerValidationError("A task must provide an asynchronous callable.")
        if (
            not isinstance(spec.dependencies, tuple)
            or any(not isinstance(item, str) for item in spec.dependencies)
            or len(set(spec.dependencies)) != len(spec.dependencies)
        ):
            raise SchedulerValidationError("Dependencies must be a tuple of distinct task IDs.")
        if not _number(spec.timeout_seconds, positive=True):
            raise SchedulerValidationError("Task timeouts must be finite and positive.")
        if not _number(spec.retry_delay_seconds, positive=False):
            raise SchedulerValidationError("Retry delays must be finite and nonnegative.")
        if (
            isinstance(spec.max_attempts, bool)
            or not isinstance(spec.max_attempts, int)
            or not 1 <= spec.max_attempts <= 10
        ):
            raise SchedulerValidationError("Task attempts must be an integer between 1 and 10.")
        if not isinstance(spec.allow_failed_dependencies, bool):
            raise SchedulerValidationError("The dependency policy must be a boolean.")
        by_id[spec.id] = spec

    children: dict[str, list[str]] = {spec.id: [] for spec in specs}
    degrees = {spec.id: len(spec.dependencies) for spec in specs}
    for spec in specs:
        for dependency in spec.dependencies:
            if dependency not in by_id:
                raise SchedulerValidationError("A dependency references an unknown task.")
            children[dependency].append(spec.id)
    ready = deque(task_id for task_id, degree in degrees.items() if degree == 0)
    visited = 0
    while ready:
        visited += 1
        for child in children[ready.popleft()]:
            degrees[child] -= 1
            if degrees[child] == 0:
                ready.append(child)
    if visited != len(specs):
        raise SchedulerValidationError("Task dependencies contain a cycle.")
    return specs


class Scheduler:
    def __init__(
        self,
        concurrency: int = 3,
        on_event: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        if isinstance(concurrency, bool) or not isinstance(concurrency, int):
            raise SchedulerValidationError("Concurrency must be an integer between 1 and 128.")
        if not 1 <= concurrency <= 128:
            raise SchedulerValidationError("Concurrency must be an integer between 1 and 128.")
        if on_event is not None and not callable(on_event):
            raise SchedulerValidationError("The audit callback must be callable.")
        self.concurrency = concurrency
        self.on_event = on_event
        self.last_result: ScheduleResult | None = None
        self._running = False
        self._records: dict[str, TaskRecord] = {}
        self._results: dict[str, Any] = {}
        self._events: list[dict[str, Any]] = []
        self._active: dict[str, asyncio.Task[None]] = {}
        self._audit_failed = False
        self._stop_reason = "cancelled"

    def _emit(self, record: TaskRecord, timestamp: str) -> None:
        event = {
            "seq": len(self._events) + 1,
            "task_id": record.id,
            "state": record.state.value,
            "attempt": record.attempts,
            "error_code": record.error_code,
            "timestamp": timestamp,
        }
        self._events.append(event)
        if self.on_event is not None and not self._audit_failed:
            try:
                result = self.on_event(dict(event))
                if inspect.isawaitable(result):
                    if inspect.iscoroutine(result):
                        result.close()
                    raise TypeError("The audit callback must be synchronous.")
            except (Exception, asyncio.CancelledError):
                self._audit_failed = True
                raise SchedulerAuditError() from None

    def _transition(self, task_id: str, state: TaskState, error_code: str | None = None) -> None:
        record = self._records[task_id]
        if state not in _TRANSITIONS.get(record.state, set()):
            raise RuntimeError("Invalid scheduler state transition.")
        timestamp = datetime.now(UTC).isoformat()
        record.state = state
        record.error_code = error_code
        if state is TaskState.RUNNING:
            record.attempts += 1
            record.started_at = record.started_at or timestamp
        if state in TERMINAL_STATES:
            record.finished_at = timestamp
        self._emit(record, timestamp)

    async def _execute(self, spec: TaskSpec, semaphore: asyncio.Semaphore, deadline: float) -> None:
        try:
            while True:
                # Queueing time does not consume a task's execution timeout.
                async with semaphore:
                    if self._audit_failed:
                        raise SchedulerAuditError()
                    if asyncio.get_running_loop().time() >= deadline:
                        raise TimeoutError
                    self._transition(spec.id, TaskState.RUNNING)
                    context = TaskContext(
                        results=MappingProxyType(
                            {
                                key: self._results[key]
                                for key in spec.dependencies
                                if key in self._results
                            }
                        ),
                        states=MappingProxyType(
                            {key: self._records[key].state for key in spec.dependencies}
                        ),
                        attempt=self._records[spec.id].attempts,
                    )
                    retryable = False
                    failure_state = TaskState.FAILED
                    # A synchronous durable audit write may have consumed the
                    # remaining global budget before asyncio's timer can run.
                    if asyncio.get_running_loop().time() >= deadline:
                        raise TimeoutError
                    try:
                        async with asyncio.timeout(spec.timeout_seconds):
                            value = await spec.run(context)
                    except TimeoutError:
                        error_code = "timeout"
                        failure_state = TaskState.TIMED_OUT
                        retryable = True
                    except TaskRetryableError as exc:
                        error_code = exc.code
                        retryable = True
                    except ResearchError as exc:
                        error_code = (
                            exc.code
                            if isinstance(exc.code, str) and _ERROR_CODE.fullmatch(exc.code)
                            else "unexpected"
                        )
                        retryable = exc.code in _RETRYABLE_CODES
                    except Exception:
                        error_code = "unexpected"
                    else:
                        self._results[spec.id] = value
                        self._transition(spec.id, TaskState.SUCCEEDED)
                        return
                    if retryable and self._records[spec.id].attempts < spec.max_attempts:
                        self._transition(spec.id, TaskState.RETRY_WAIT, error_code)
                    else:
                        self._transition(spec.id, failure_state, error_code)
                        return
                # Backoff does not monopolize a concurrency slot.
                await asyncio.sleep(spec.retry_delay_seconds)
                self._transition(spec.id, TaskState.READY)
        except asyncio.CancelledError:
            if self._records[spec.id].state not in TERMINAL_STATES:
                try:
                    self._transition(spec.id, TaskState.CANCELLED, self._stop_reason)
                except SchedulerAuditError:
                    pass  # A failed audit must not replace the original cancellation.
            raise

    async def _drive(self, specs: list[TaskSpec], deadline: float) -> None:
        semaphore = asyncio.Semaphore(self.concurrency)
        while any(record.state not in TERMINAL_STATES for record in self._records.values()):
            if asyncio.get_running_loop().time() >= deadline:
                raise TimeoutError
            for spec in specs:
                record = self._records[spec.id]
                if record.state is not TaskState.PENDING:
                    continue
                states = [self._records[key].state for key in spec.dependencies]
                if not all(state in TERMINAL_STATES for state in states):
                    continue
                if not spec.allow_failed_dependencies and any(
                    state is not TaskState.SUCCEEDED for state in states
                ):
                    self._transition(spec.id, TaskState.SKIPPED, "dependency_failed")
                else:
                    self._transition(spec.id, TaskState.READY)
                    self._active[spec.id] = asyncio.create_task(
                        self._execute(spec, semaphore, deadline), name=f"research:{spec.id}"
                    )
            if not self._active:
                # A reverse-ordered chain may need another scan to propagate skips.
                continue
            done, _ = await asyncio.wait(self._active.values(), return_when=asyncio.FIRST_COMPLETED)
            for task_id, task in list(self._active.items()):
                if task not in done:
                    continue
                del self._active[task_id]
                if not task.cancelled() and (error := task.exception()) is not None:
                    raise error
            if self._audit_failed:
                raise SchedulerAuditError()

    async def _stop(self, reason: str) -> bool:
        """Cancel and drain every child; a repeated parent cancel cannot orphan them."""
        self._stop_reason = reason
        for task in self._active.values():
            if not task.done():
                task.cancel()
        received_cancel = False
        drain = asyncio.gather(*self._active.values(), return_exceptions=True)
        while not drain.done():
            try:
                await asyncio.shield(drain)
            except asyncio.CancelledError:
                received_cancel = True
        drain.result()
        self._active.clear()
        for task_id, record in self._records.items():
            if record.state not in TERMINAL_STATES:
                try:
                    self._transition(task_id, TaskState.CANCELLED, reason)
                except SchedulerAuditError:
                    pass
        return received_cancel

    async def run(self, tasks: list[TaskSpec], timeout_seconds: float = 60) -> ScheduleResult:
        if self._running:
            raise SchedulerValidationError("This scheduler already has an active run.")
        specs = _validate(tasks, timeout_seconds)
        self._running = True
        self._records = {spec.id: TaskRecord(spec.id) for spec in specs}
        self._results = {}
        self._events = []
        self._active = {}
        self._audit_failed = False
        self._stop_reason = "cancelled"
        timed_out = False
        deadline = asyncio.get_running_loop().time() + timeout_seconds
        try:
            for record in self._records.values():
                self._emit(record, datetime.now(UTC).isoformat())
            async with asyncio.timeout_at(deadline):
                await self._drive(specs, deadline)
        except TimeoutError:
            timed_out = True
            if await self._stop("global_timeout"):
                raise asyncio.CancelledError from None
            if self._audit_failed:
                raise SchedulerAuditError() from None
        except asyncio.CancelledError:
            await self._stop("cancelled")
            raise
        except BaseException:
            if await self._stop("audit_error" if self._audit_failed else "scheduler_error"):
                raise asyncio.CancelledError from None
            raise
        finally:
            self.last_result = ScheduleResult(
                records={key: replace(record) for key, record in self._records.items()},
                results=dict(self._results),
                events=[dict(event) for event in self._events],
                timed_out=timed_out,
            )
            self._running = False
        return self.last_result
