"""Loopback-only, single-user HTTP transport for the bounded research pipeline.

The browser never controls filesystem paths, credentials, or provider settings.
Disconnecting its SSE subscription does not own/cancel the durable job.
"""

import asyncio
import json
import re
import secrets
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from uuid import UUID

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError
from starlette.exceptions import HTTPException
from starlette.responses import FileResponse, JSONResponse, Response, StreamingResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from deep_research.dag_journal import DagJournal
from deep_research.errors import ResearchError
from deep_research.model_budget import validate_budget_summary, validate_profile_snapshot
from deep_research.model_profiles import ProfileRegistry
from deep_research.research import render_markdown
from deep_research.research_store import ResearchStore
from deep_research.web_jobs import WebJobStore, WebManager, WebRequest

MAX_BODY_BYTES = 8192
MAX_SSE_CONNECTIONS = 16
SSE_POLL_SECONDS = 0.25
SSE_HEARTBEAT_POLLS = 40
TERMINAL = frozenset({"completed", "failed", "cancelled", "interrupted"})
STATIC_DIR = Path(__file__).with_name("static")
_SECURITY_HEADERS = {
    "content-security-policy": "default-src 'self'; script-src 'self'; style-src 'self'; "
    "img-src 'self'; connect-src 'self'; object-src 'none'; base-uri 'none'; "
    "frame-ancestors 'none'; form-action 'self'",
    "x-content-type-options": "nosniff",
    "referrer-policy": "no-referrer",
    "cache-control": "no-store",
    "x-frame-options": "DENY",
}
_ERROR_MESSAGES = {
    "web_not_found": (404, "任务或资源不存在。"),
    "web_conflict": (409, "请求与已有任务不一致，或报告尚不可用。"),
    "web_capacity": (429, "任务队列或连接数量已达上限，请稍后重试。"),
    "web_unavailable": (503, "服务暂不可用。"),
    "web_io": (503, "本地任务存储暂不可用。"),
    "web_closed": (503, "服务正在关闭。"),
    "web_validation": (422, "请求字段无效，请检查输入。"),
    "web_origin": (403, "仅接受本机同源请求。"),
    "web_csrf": (403, "请求校验失败，请刷新页面后重试。"),
    "web_body_size": (413, "请求体超过 8 KiB 上限。"),
    "web_media_type": (415, "请使用 application/json。"),
    "web_live_disabled": (403, "真实研究尚未启用，或未确认外发与费用。"),
    "web_profile": (422, "所选模型配置不可用，请刷新配置后重试。"),
    "web_cursor": (400, "事件游标无效。"),
    "web_unexpected": (500, "请求处理失败，请检查本地服务状态。"),
}


def _error(code: str) -> JSONResponse:
    status, message = _ERROR_MESSAGES.get(code, _ERROR_MESSAGES["web_unexpected"])
    if code not in _ERROR_MESSAGES:
        code = "web_unexpected"
    return JSONResponse({"error": {"code": code, "message": message}}, status_code=status)


def _public(value: Any) -> Any:
    """Remove server-owned path bookkeeping, without rewriting user/source prose."""
    if isinstance(value, dict):
        return {
            key: _public(item)
            for key, item in value.items()
            if key not in {"output_dir", "env_file", "db", "path"} and not key.endswith("_path")
        }
    if isinstance(value, list):
        return [_public(item) for item in value]
    return value


class LocalSecurityMiddleware:
    """Pure ASGI guard: cap streamed bytes before JSON parsing, including chunked bodies."""

    def __init__(self, app: ASGIApp, *, port: int, csrf_token: str) -> None:
        self.app, self.port, self.csrf_token = app, port, csrf_token

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        async def secure_send(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = [
                    (key, value)
                    for key, value in message.get("headers", [])
                    if key.decode("latin-1").lower() not in _SECURITY_HEADERS
                ]
                headers.extend(
                    (key.encode("ascii"), value.encode("ascii"))
                    for key, value in _SECURITY_HEADERS.items()
                )
                message = {**message, "headers": headers}
            await send(message)

        async def reject(code: str) -> None:
            await _error(code)(scope, receive, secure_send)

        raw = scope.get("headers", [])

        def values(name: bytes) -> list[str]:
            return [v.decode("latin-1") for k, v in raw if k.lower() == name]

        hosts, origins = values(b"host"), values(b"origin")
        allowed_hosts = {f"127.0.0.1:{self.port}", f"localhost:{self.port}"}
        if len(hosts) != 1 or hosts[0].lower() not in allowed_hosts:
            await reject("web_origin")
            return
        if origins and (len(origins) != 1 or origins[0] != f"http://{hosts[0].lower()}"):
            await reject("web_origin")
            return
        sites = values(b"sec-fetch-site")
        if sites and (len(sites) != 1 or sites[0] not in {"same-origin", "none"}):
            await reject("web_origin")
            return
        if scope["method"] not in {"GET", "HEAD", "OPTIONS"}:
            tokens = values(b"x-csrf-token")
            if len(tokens) != 1 or not secrets.compare_digest(
                tokens[0].encode("utf-8"), self.csrf_token.encode("ascii")
            ):
                await reject("web_csrf")
                return
        lengths = values(b"content-length")
        if lengths and (
            len(lengths) != 1
            or not re.fullmatch(r"[0-9]{1,10}", lengths[0])
            or int(lengths[0]) > MAX_BODY_BYTES
        ):
            await reject("web_body_size")
            return
        chunks: list[bytes] = []
        size = 0
        try:
            async with asyncio.timeout(10):
                while True:
                    message = await receive()
                    if message["type"] == "http.disconnect":
                        return
                    if message["type"] != "http.request":
                        continue
                    chunk = message.get("body", b"")
                    size += len(chunk)
                    if size > MAX_BODY_BYTES:
                        await reject("web_body_size")
                        return
                    chunks.append(chunk)
                    if not message.get("more_body", False):
                        break
        except TimeoutError:
            await reject("web_validation")
            return
        sent = False

        async def replay_receive() -> Message:
            nonlocal sent
            if not sent:
                sent = True
                return {"type": "http.request", "body": b"".join(chunks), "more_body": False}
            return await receive()

        await self.app(scope, replay_receive, secure_send)


def _unique_object(pairs: list[tuple[str, Any]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _invalid_constant(_: str) -> None:
    raise ValueError("non-finite JSON number")


def _uuid(value: str) -> str:
    try:
        if str(UUID(value)) != value:
            raise ValueError
    except (ValueError, AttributeError):
        raise ResearchError("web_validation", "任务编号无效。") from None
    return value


def create_app(
    *,
    db: Path,
    output_dir: Path,
    env_file: Path = Path(".env"),
    allow_live: bool = False,
    runner: Any = None,
    port: int = 8765,
    embedding_backend: str = "lexical",
    embedding_model_path: Path | None = None,
    profiles: ProfileRegistry | None = None,
    evidence_selection: str = "coverage",
) -> FastAPI:
    """Construct without loading credentials/models or starting any work."""
    if type(port) is not int or not 1 <= port <= 65535:
        raise ValueError("invalid local port")
    if evidence_selection not in ("coverage", "multilingual", "contextual"):
        raise ValueError("invalid evidence selection")
    csrf_token = secrets.token_hex(32)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        nonlocal runner
        if runner is None:
            from deep_research.web_runner import make_runner

            runner = make_runner(
                db=db,
                output_dir=output_dir,
                env_file=env_file,
                allow_live=allow_live,
                embedding_backend=embedding_backend,
                embedding_model_path=embedding_model_path,
                profiles=profiles,
                evidence_selection=evidence_selection,
            )
        with WebJobStore(db) as store:
            manager = WebManager(store, runner)
            app.state.manager = manager
            await manager.start()
            try:
                yield
            finally:
                await manager.close()

    app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
    app.state.sse_connections = 0
    app.add_middleware(LocalSecurityMiddleware, port=port, csrf_token=csrf_token)

    @app.exception_handler(ResearchError)
    async def research_error(_: Request, error: ResearchError) -> JSONResponse:
        return _error(error.code)

    @app.exception_handler(RequestValidationError)
    async def validation_error(_: Request, __: RequestValidationError) -> JSONResponse:
        # FastAPI's default response includes submitted values; do not echo them.
        return _error("web_validation")

    @app.exception_handler(HTTPException)
    async def http_error(_: Request, error: HTTPException) -> JSONResponse:
        if error.status_code == 404:
            return _error("web_not_found")
        return JSONResponse(
            {"error": {"code": "web_http", "message": "请求方法或资源无效。"}},
            status_code=error.status_code,
        )

    @app.exception_handler(Exception)
    async def unexpected(_: Request, __: Exception) -> JSONResponse:
        response = _error("web_unexpected")
        response.headers.update(_SECURITY_HEADERS)
        return response

    def manager() -> WebManager:
        result = getattr(app.state, "manager", None)
        if result is None:
            raise ResearchError("web_unavailable", "服务尚未启动。")
        return result

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html", media_type="text/html")

    @app.get("/app.js")
    async def script() -> FileResponse:
        return FileResponse(STATIC_DIR / "app.js", media_type="text/javascript")

    @app.get("/harness")
    async def harness_page() -> FileResponse:
        return FileResponse(STATIC_DIR / "harness.html", media_type="text/html")

    @app.get("/harness.js")
    async def harness_script() -> FileResponse:
        return FileResponse(STATIC_DIR / "harness.js", media_type="text/javascript")

    @app.get("/api/harness")
    async def harness_runs() -> dict:
        from deep_research.harness_store import HarnessStore

        path = db.parent / "harness-research.harness.sqlite3"
        return {"runs": HarnessStore(path).list_runs() if path.is_file() else []}

    @app.get("/api/harness/{harness_id}")
    async def harness_trace(harness_id: str) -> dict:
        from deep_research.harness_store import HarnessStore

        _uuid(harness_id)
        path = db.parent / "harness-research.harness.sqlite3"
        if not path.is_file():
            raise ResearchError("web_not_found", "未找到此运行记录。", status_code=404)
        return HarnessStore(path).trace(harness_id)

    @app.get("/styles.css")
    async def stylesheet() -> FileResponse:
        return FileResponse(STATIC_DIR / "styles.css", media_type="text/css")

    @app.get("/api/config")
    async def config() -> dict:
        return {
            "csrf_token": csrf_token,
            "allow_live": allow_live,
            "supports_robust_contracts": True,
            "supports_evidence_gate": True,
            "supports_research_loop": True,
            "embedding_backend": embedding_backend,
            "max_active": 1,
            "max_queued": 8,
            "demo_question": WebRequest().question,
            "model_profiles": profiles.public_catalog() if profiles is not None else [],
            "default_profile": profiles.default_profile if profiles is not None else None,
            "demo_scenarios": [
                "passed",
                "repair",
                "oscillation",
                "stalled",
                "max-rounds",
                "format",
                "deadline",
                "slow",
            ],
        }

    @app.post("/api/jobs")
    async def submit(request: Request) -> JSONResponse:
        content_types = request.headers.getlist("content-type")
        if (
            len(content_types) != 1
            or content_types[0].split(";", 1)[0].strip().lower() != "application/json"
        ):
            return _error("web_media_type")
        keys = request.headers.getlist("idempotency-key")
        if len(keys) != 1:
            return _error("web_validation")
        key = _uuid(keys[0])
        try:
            payload = json.loads(
                (await request.body()).decode("utf-8"),
                object_pairs_hook=_unique_object,
                parse_constant=_invalid_constant,
            )
            data = WebRequest.model_validate(payload)
        except (ValueError, UnicodeError, RecursionError, ValidationError):
            return _error("web_validation")
        if data.mode == "live" and (not allow_live or not data.confirm_live):
            return _error("web_live_disabled")
        if data.mode == "live":
            if profiles is not None:
                try:
                    selected = profiles.get(data.profile_id)
                except ResearchError:
                    return _error("web_profile")
                data = WebRequest.model_validate({**data.model_dump(), "profile_id": selected.id})
            elif data.profile_id is not None:
                return _error("web_profile")
        job, created = manager().submit(data, key)
        return JSONResponse(_public(job), status_code=202 if created else 200)

    @app.get("/api/jobs")
    async def list_jobs() -> dict:
        return {"jobs": _public(manager().list())}

    @app.get("/api/jobs/{job_id}")
    async def get_job(job_id: str) -> dict:
        return _public(manager().get(_uuid(job_id)))

    @app.post("/api/jobs/{job_id}/cancel")
    async def cancel_job(job_id: str) -> dict:
        return _public(manager().cancel(_uuid(job_id)))

    @app.get("/api/jobs/{job_id}/model-audit")
    async def model_audit(job_id: str) -> dict:
        job = manager().get(_uuid(job_id))
        if job["run_id"] is None:
            return {"run_id": None, "model_profile": None, "model_budget": None, "calls": []}
        with DagJournal(db) as journal:
            audit = journal.snapshot(job["run_id"])
        if audit is None:
            return {
                "run_id": job["run_id"],
                "model_profile": None,
                "model_budget": None,
                "calls": [],
            }
        profile = audit["config"].get("model_profile")
        budget = audit.get("model_budget")
        return {
            "run_id": job["run_id"],
            "model_profile": validate_profile_snapshot(profile) if profile is not None else None,
            "model_budget": validate_budget_summary(budget) if budget is not None else None,
            "calls": [
                {key: call[key] for key in ("call_id", "stage", "status", "details") if key in call}
                for call in audit["model_calls"]
            ],
        }

    @app.get("/api/jobs/{job_id}/events")
    async def events(job_id: str, request: Request) -> Response:
        job_id = _uuid(job_id)
        job = manager().get(job_id)
        headers = request.headers.getlist("last-event-id")
        queries = request.query_params.getlist("after")
        if len(headers) > 1 or len(queries) > 1:
            return _error("web_cursor")
        # Browser reconnect supplies Last-Event-ID; it supersedes the original URL cursor.
        raw_cursor = headers[0] if headers else (queries[0] if queries else "0")
        if not re.fullmatch(r"[0-9]{1,10}", raw_cursor):
            return _error("web_cursor")
        cursor = int(raw_cursor)
        snapshot = manager().events(job_id)
        if cursor > (snapshot[-1]["seq"] if snapshot else 0):
            return _error("web_cursor")
        if app.state.sse_connections >= MAX_SSE_CONNECTIONS:
            return _error("web_capacity")
        # Reserve before returning StreamingResponse so concurrent requests cannot overbook.
        app.state.sse_connections += 1

        async def stream():
            nonlocal cursor, job
            polls = 0
            try:
                yield "retry: 1500\n\n"
                while True:
                    if await request.is_disconnected():
                        return
                    batch = manager().events(job_id, after=cursor)
                    # Read state before yielding: a job can finish while a slow
                    # client consumes this batch, so checking afterwards could
                    # close the stream without delivering its terminal event.
                    job = manager().get(job_id)
                    for item in batch:
                        cursor = item["seq"]
                        name = "progress" if item["kind"] == "progress" else "status"
                        serialized = json.dumps(_public(item), ensure_ascii=False, allow_nan=False)
                        yield f"id: {cursor}\nevent: {name}\ndata: {serialized}\n\n"
                    if job["status"] in TERMINAL:
                        return
                    polls += 1
                    if polls >= SSE_HEARTBEAT_POLLS:
                        yield ": heartbeat\n\n"
                        polls = 0
                    await asyncio.sleep(SSE_POLL_SECONDS)
            except ResearchError:
                # A stream already sent 200; signal unavailable without exposing internals.
                yield 'event: unavailable\ndata: {"code":"web_unavailable"}\n\n'

        class ReservedStream(StreamingResponse):
            async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
                try:
                    await super().__call__(scope, receive, send)
                finally:
                    app.state.sse_connections -= 1

        return ReservedStream(
            stream(),
            media_type="text/event-stream",
            headers={"X-Accel-Buffering": "no"},
        )

    def report_bundle(job_id: str) -> dict:
        job = manager().get(_uuid(job_id))
        if job["status"] != "completed" or not job.get("run_id"):
            raise ResearchError("web_conflict", "任务尚无可下载的完成报告。")
        with ResearchStore(db) as store:
            run = store.get_run(job["run_id"])
            bundle = store.get_report(job["run_id"])
        if (
            run is None
            or run["status"] != "completed"
            or not isinstance(bundle, dict)
            or bundle.get("run_id") != job["run_id"]
        ):
            raise ResearchError("web_conflict", "任务报告绑定无效。")
        return _public(bundle)

    @app.get("/api/jobs/{job_id}/report")
    async def report(job_id: str) -> dict:
        return report_bundle(job_id)

    @app.get("/api/jobs/{job_id}/download/{format}")
    async def download(job_id: str, format: str) -> Response:
        if format not in {"markdown", "json"}:
            return _error("web_not_found")
        bundle = report_bundle(job_id)
        if format == "markdown":
            content, media, extension = render_markdown(bundle), "text/markdown", "md"
        else:
            content = json.dumps(bundle, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
            media, extension = "application/json", "json"
        return Response(
            content,
            media_type=media,
            headers={
                "Content-Disposition": f'attachment; filename="research-{job_id}.{extension}"'
            },
        )

    return app
