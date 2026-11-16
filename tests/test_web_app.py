"""Offline HTTP contract, request guards, durable SSE and export acceptance."""

import asyncio
import hashlib
import json
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import httpx
import pytest

from deep_research.research_store import ResearchStore
from deep_research.web_app import MAX_BODY_BYTES, create_app

BASE = "http://127.0.0.1:8765"


def complete_report(db: Path, run_id: str) -> None:
    text = "离线资料只用于测试，不代表真实研究。"
    bundle = {
        "schema_version": 2,
        "run_id": run_id,
        "question": "测试问题",
        "retrieval_mode": "offline_demo",
        "report": {
            "title": "离线报告 <script>alert(1)</script>",
            "sections": [{"heading": "结果", "claims": [{"text": text, "citations": ["E1"]}]}],
            "limitations": ["仅测试。"],
        },
        "sources": [
            {
                "source_id": "S1",
                "url": "https://example.com/source",
                "title": "测试资料",
                "text": text,
                "sha256": hashlib.sha256(text.encode()).hexdigest(),
                "retrieved_at": datetime.now(UTC).isoformat(),
                "truncated": False,
            }
        ],
        "evidence": [
            {"evidence_id": "E1", "source_id": "S1", "start": 0, "end": len(text), "quote": text}
        ],
        "system_limitations": ["本测试没有真实模型调用。"],
        "warnings": [],
        "execution": {"degraded": False},
    }
    with ResearchStore(db) as store:
        store.save_sources(run_id, bundle["sources"], bundle["evidence"])
        store.complete_run(run_id, bundle, [], "D:/private/outputs")


@pytest.fixture
def app_factory(tmp_path):
    @asynccontextmanager
    async def factory(*, runner=None, allow_live=False, port=8765):
        db = tmp_path / f"{uuid4()}.sqlite3"

        async def normal(request, on_progress, on_run_created):
            with ResearchStore(db) as store:
                run_id = store.create_run(request.question)
            on_run_created(run_id)
            on_progress("离线步骤已完成")
            await asyncio.sleep(0)
            complete_report(db, run_id)
            return {
                "status": "completed",
                "run_id": run_id,
                "report_path": "D:/private/outputs/report.md",
                "evidence_path": "D:/private/outputs/evidence.json",
                "model_call_count": 0,
                "degraded": False,
            }

        app = create_app(
            db=db,
            output_dir=tmp_path / "reports",
            runner=runner or normal,
            allow_live=allow_live,
            port=port,
        )
        async with app.router.lifespan_context(app):
            transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
            async with httpx.AsyncClient(
                transport=transport, base_url=f"http://127.0.0.1:{port}"
            ) as client:
                token = (await client.get("/api/config")).json()["csrf_token"]
                headers = {"X-CSRF-Token": token, "Idempotency-Key": str(uuid4())}
                yield app, client, headers, db

    return factory


async def wait_terminal(client, job_id):
    for _ in range(100):
        job = (await client.get(f"/api/jobs/{job_id}")).json()
        if job["status"] in {"completed", "cancelled", "failed", "interrupted"}:
            return job
        await asyncio.sleep(0.005)
    raise AssertionError("offline job did not finish")


async def test_config_is_offline_default_and_does_not_expose_paths_or_keys(app_factory):
    async with app_factory() as (_, client, _, _):
        result = await client.get("/api/config")
        config = result.json()
        assert config["allow_live"] is False
        assert config["max_active"] == 1 and config["max_queued"] == 8
        assert len(config["csrf_token"]) == 64
        assert config["embedding_backend"] == "lexical"
        assert config["demo_question"]
        assert "env_file" not in result.text and "api_key" not in result.text


async def test_submit_get_list_replay_and_safe_completion(app_factory):
    async with app_factory() as (_, client, headers, _):
        response = await client.post("/api/jobs", json={}, headers=headers)
        assert response.status_code == 202
        job_id = response.json()["id"]
        final = await wait_terminal(client, job_id)
        assert final["status"] == "completed"
        assert final["run_id"]
        assert "D:/private" not in json.dumps(final)
        replay = await client.post("/api/jobs", json={}, headers=headers)
        assert replay.status_code == 200 and replay.json()["id"] == job_id
        jobs = (await client.get("/api/jobs")).json()["jobs"]
        assert len(jobs) == 1 and jobs[0]["id"] == job_id
        conflict = await client.post("/api/jobs", json={"scenario": "passed"}, headers=headers)
        assert conflict.status_code == 409


@pytest.mark.parametrize(
    "host", ["evil.test:8765", "127.0.0.1:3000", "127.0.0.1", "localhost", "127.0.0.1.evil:8765"]
)
async def test_host_must_match_configured_loopback_port(app_factory, host):
    async with app_factory() as (_, client, _, _):
        response = await client.get("/api/config", headers={"Host": host})
        assert response.status_code == 403
        assert "csrf_token" not in response.text


@pytest.mark.parametrize(
    "origin",
    [
        "https://evil.example",
        "null",
        "http://localhost:8765",
        "http://127.0.0.1:3000",
        "https://127.0.0.1:8765",
    ],
)
async def test_foreign_origin_is_rejected_even_for_reads(app_factory, origin):
    async with app_factory() as (_, client, _, _):
        response = await client.get("/api/jobs", headers={"Origin": origin})
        assert response.status_code == 403


@pytest.mark.parametrize("site", ["cross-site", "same-site", "unknown"])
async def test_fetch_site_restriction(app_factory, site):
    async with app_factory() as (_, client, _, _):
        assert (
            await client.get("/api/config", headers={"Sec-Fetch-Site": site})
        ).status_code == 403


async def test_localhost_same_origin_and_missing_browser_headers_supported(app_factory):
    async with app_factory(port=8766) as (_, client, _, _):
        response = await client.get(
            "/api/config",
            headers={
                "Host": "localhost:8766",
                "Origin": "http://localhost:8766",
                "Sec-Fetch-Site": "same-origin",
            },
        )
        assert response.status_code == 200
        assert (await client.get("/api/config")).status_code == 200


@pytest.mark.parametrize("header", [None, "", "wrong"])
async def test_mutations_require_csrf(app_factory, header):
    async with app_factory() as (_, client, _, _):
        headers = {} if header is None else {"X-CSRF-Token": header}
        response = await client.post("/api/jobs", json={}, headers=headers)
        assert response.status_code == 403
        response = await client.post(f"/api/jobs/{uuid4()}/cancel", headers=headers)
        assert response.status_code == 403


async def test_csrf_rotates_with_new_app_instance(app_factory):
    async with app_factory() as (_, _, first, _):
        async with app_factory() as (_, client, second, _):
            assert first["X-CSRF-Token"] != second["X-CSRF-Token"]
            assert (await client.post("/api/jobs", json={}, headers=first)).status_code == 403


@pytest.mark.parametrize(
    "body",
    [
        b"{",
        b"[]",
        b"null",
        b'{"mode":"demo","mode":"live"}',
        b'{"unknown":1}',
        b'{"review":NaN}',
        b"\xff",
        b'{"question":123}',
        b'{"review":"true"}',
        b'{"memory_scope":"../../secret"}',
    ],
)
async def test_invalid_json_or_schema_does_not_echo_input(app_factory, body):
    async with app_factory() as (_, client, headers, _):
        response = await client.post(
            "/api/jobs", content=body, headers={**headers, "Content-Type": "application/json"}
        )
        assert response.status_code == 422
        assert "input" not in response.text and "traceback" not in response.text.lower()


@pytest.mark.parametrize("key", [None, "", "not-a-uuid", "../secret"])
async def test_idempotency_key_required_and_validated(app_factory, key):
    async with app_factory() as (_, client, headers, _):
        headers.pop("Idempotency-Key")
        if key is not None:
            headers["Idempotency-Key"] = key
        assert (await client.post("/api/jobs", json={}, headers=headers)).status_code == 422


@pytest.mark.parametrize("content_type", [None, "text/plain", "application/x-www-form-urlencoded"])
async def test_json_content_type_required(app_factory, content_type):
    async with app_factory() as (_, client, headers, _):
        if content_type is not None:
            headers["Content-Type"] = content_type
        assert (await client.post("/api/jobs", content=b"{}", headers=headers)).status_code == 415


async def test_body_hard_limit_with_and_without_content_length(app_factory):
    async with app_factory() as (_, client, headers, _):
        headers["Content-Type"] = "application/json"
        response = await client.post(
            "/api/jobs", content=b"x" * (MAX_BODY_BYTES + 1), headers=headers
        )
        assert response.status_code == 413

        async def chunks():
            yield b"x" * 4000
            yield b"x" * 4193

        response = await client.post("/api/jobs", content=chunks(), headers=headers)
        assert response.status_code == 413
        assert not (await client.get("/api/jobs")).json()["jobs"]


@pytest.mark.parametrize(
    "allow_live,confirm,status",
    [(False, False, 422), (False, True, 403), (True, False, 422), (True, True, 202)],
)
async def test_live_requires_server_optin_and_per_request_confirmation(
    app_factory, allow_live, confirm, status
):
    async with app_factory(allow_live=allow_live) as (_, client, headers, _):
        response = await client.post(
            "/api/jobs",
            headers=headers,
            json={"mode": "live", "question": "光合作用是什么？", "confirm_live": confirm},
        )
        assert response.status_code == status


async def test_report_and_exports_from_bound_database_not_paths(app_factory):
    async with app_factory() as (_, client, headers, _):
        job_id = (await client.post("/api/jobs", json={}, headers=headers)).json()["id"]
        job = await wait_terminal(client, job_id)
        report = await client.get(f"/api/jobs/{job_id}/report")
        assert report.status_code == 200 and report.json()["run_id"] == job["run_id"]
        assert report.json()["report"]["title"].endswith("</script>")
        assert "D:/private" not in report.text
        markdown = await client.get(f"/api/jobs/{job_id}/download/markdown")
        assert markdown.status_code == 200
        assert "attachment" in markdown.headers["content-disposition"]
        assert "&lt;script&gt;" in markdown.text and "<script>" not in markdown.text
        export = await client.get(f"/api/jobs/{job_id}/download/json")
        assert export.json() == report.json()
        assert "attachment" in export.headers["content-disposition"]
        assert (await client.get(f"/api/jobs/{job['run_id']}/report")).status_code == 404


async def test_running_job_cannot_export_and_cancel_is_idempotent(app_factory):
    cancelled = asyncio.Event()

    async def blocked(request, progress, created):
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    async with app_factory(runner=blocked) as (_, client, headers, _):
        job_id = (await client.post("/api/jobs", json={}, headers=headers)).json()["id"]
        await asyncio.sleep(0.01)
        assert (await client.get(f"/api/jobs/{job_id}/report")).status_code == 409
        assert (await client.get(f"/api/jobs/{job_id}/download/json")).status_code == 409
        response = await client.post(f"/api/jobs/{job_id}/cancel", headers=headers)
        assert response.status_code == 200
        assert (await wait_terminal(client, job_id))["status"] == "cancelled"
        assert cancelled.is_set()
        assert (await client.post(f"/api/jobs/{job_id}/cancel", headers=headers)).status_code == 200


async def test_sse_replays_durable_ids_then_closes_terminal_stream(app_factory):
    async with app_factory() as (app, client, headers, _):
        job_id = (await client.post("/api/jobs", json={}, headers=headers)).json()["id"]
        await wait_terminal(client, job_id)
        response = await client.get(f"/api/jobs/{job_id}/events")
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        assert "event: progress" in response.text and "event: status" in response.text
        records = [
            json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")
        ]
        assert [item["seq"] for item in records] == sorted(item["seq"] for item in records)
        assert records[-1]["kind"] == "completed"
        assert app.state.sse_connections == 0
        resume = await client.get(
            f"/api/jobs/{job_id}/events?after=0", headers={"Last-Event-ID": str(records[-2]["seq"])}
        )
        assert resume.text.count("data: ") == 1
        final = await client.get(f"/api/jobs/{job_id}/events?after={records[-1]['seq']}")
        assert "data: " not in final.text


@pytest.mark.parametrize("cursor", ["-1", "1.2", "NaN", "10000000000000000", "99999"])
async def test_sse_cursor_rejects_invalid_and_future(app_factory, cursor):
    async with app_factory() as (_, client, headers, _):
        job_id = (await client.post("/api/jobs", json={}, headers=headers)).json()["id"]
        await wait_terminal(client, job_id)
        assert (await client.get(f"/api/jobs/{job_id}/events?after={cursor}")).status_code == 400


async def test_sse_connection_cap(app_factory):
    async with app_factory() as (app, client, headers, _):
        job_id = (await client.post("/api/jobs", json={}, headers=headers)).json()["id"]
        app.state.sse_connections = 16
        assert (await client.get(f"/api/jobs/{job_id}/events")).status_code == 429
        app.state.sse_connections = 0


async def test_sse_disconnect_does_not_cancel_job_and_releases_slot(app_factory):
    started = asyncio.Event()

    async def blocked(request, progress, created):
        started.set()
        await asyncio.Event().wait()

    async with app_factory(runner=blocked) as (app, client, headers, _):
        job_id = (await client.post("/api/jobs", json={}, headers=headers)).json()["id"]
        await started.wait()
        disconnected = asyncio.Event()
        first = True
        messages = []

        async def receive():
            nonlocal first
            if first:
                first = False
                return {"type": "http.request", "body": b"", "more_body": False}
            await disconnected.wait()
            return {"type": "http.disconnect"}

        async def send(message):
            messages.append(message)
            if b"event: status" in message.get("body", b""):
                disconnected.set()

        scope = {
            "type": "http",
            "asgi": {"version": "3.0", "spec_version": "2.3"},
            "http_version": "1.1",
            "method": "GET",
            "scheme": "http",
            "path": f"/api/jobs/{job_id}/events",
            "query_string": b"",
            "root_path": "",
            "headers": [(b"host", b"127.0.0.1:8765")],
            "client": ("127.0.0.1", 12345),
            "server": ("127.0.0.1", 8765),
        }
        async with asyncio.timeout(2):
            await app(scope, receive, send)
        assert app.state.sse_connections == 0
        job = (await client.get(f"/api/jobs/{job_id}")).json()
        assert job["status"] == "running" and not job["cancel_requested"]
        assert any(b"event: status" in item.get("body", b"") for item in messages)


def stream_scope(job_id):
    return {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": "GET",
        "scheme": "http",
        "path": f"/api/jobs/{job_id}/events",
        "query_string": b"",
        "root_path": "",
        "headers": [(b"host", b"127.0.0.1:8765")],
        "client": ("127.0.0.1", 12345),
        "server": ("127.0.0.1", 8765),
    }


async def test_sse_finishing_while_client_reads_does_not_lose_terminal_event(app_factory):
    ready, release = asyncio.Event(), asyncio.Event()

    async def controlled(request, progress, created):
        run_id = str(uuid4())
        created(run_id)
        ready.set()
        await release.wait()
        return {"status": "completed", "run_id": run_id}

    async with app_factory(runner=controlled) as (app, client, headers, _):
        job_id = (await client.post("/api/jobs", json={}, headers=headers)).json()["id"]
        await ready.wait()
        first = True
        bodies = []

        async def receive():
            nonlocal first
            if first:
                first = False
                return {"type": "http.request", "body": b"", "more_body": False}
            await asyncio.Event().wait()

        async def send(message):
            body = message.get("body", b"")
            bodies.append(body)
            if b"event: status" in body and not release.is_set():
                release.set()
                await wait_terminal(client, job_id)

        async with asyncio.timeout(2):
            await app(stream_scope(job_id), receive, send)
        assert b'"kind": "completed"' in b"".join(bodies)
        assert app.state.sse_connections == 0


async def test_sse_reservation_released_when_cancelled_before_first_body(app_factory):
    async with app_factory() as (app, client, headers, _):
        job_id = (await client.post("/api/jobs", json={}, headers=headers)).json()["id"]
        await wait_terminal(client, job_id)
        first = True

        async def receive():
            nonlocal first
            if first:
                first = False
                return {"type": "http.request", "body": b"", "more_body": False}
            await asyncio.Event().wait()

        async def send(message):
            if message["type"] == "http.response.start":
                raise asyncio.CancelledError

        try:
            await app(stream_scope(job_id), receive, send)
        except asyncio.CancelledError:
            pass
        assert app.state.sse_connections == 0


async def test_static_assets_are_fixed_local_files_and_security_headers_apply(app_factory):
    async with app_factory() as (_, client, _, _):
        for path, media in [
            ("/", "text/html"),
            ("/app.js", "text/javascript"),
            ("/styles.css", "text/css"),
        ]:
            response = await client.get(path)
            assert response.status_code == 200
            assert response.headers["content-type"].startswith(media)
            assert "frame-ancestors 'none'" in response.headers["content-security-policy"]


async def test_unexpected_http_exception_is_sanitized_with_security_headers(
    app_factory, monkeypatch
):
    async with app_factory() as (app, client, _, _):

        def broken():
            raise RuntimeError("secret-key=D:/private/.env")

        monkeypatch.setattr(app.state.manager, "list", broken)
        response = await client.get("/api/jobs")
        assert response.status_code == 500
        assert "secret-key" not in response.text and "private" not in response.text
        assert response.headers["cache-control"] == "no-store"
        assert response.headers["x-content-type-options"] == "nosniff"


async def test_duplicate_security_headers_are_rejected(app_factory):
    async with app_factory() as (_, client, headers, _):
        response = await client.get(
            "/api/config", headers=[("Host", "127.0.0.1:8765"), ("Host", "evil.test")]
        )
        assert response.status_code == 403
        response = await client.post(
            "/api/jobs",
            json={},
            headers=[
                ("X-CSRF-Token", headers["X-CSRF-Token"]),
                ("X-CSRF-Token", headers["X-CSRF-Token"]),
                ("Idempotency-Key", headers["Idempotency-Key"]),
            ],
        )
        assert response.status_code == 403


async def test_non_ascii_csrf_fails_closed_without_exception(app_factory):
    async with app_factory() as (_, client, _, _):
        response = await client.post("/api/jobs", json={}, headers=[(b"X-CSRF-Token", b"\xff")])
        assert response.status_code == 403


@pytest.mark.parametrize(
    "path",
    [
        "/.env",
        "/docs",
        "/openapi.json",
        "/redoc",
        "/static/.env",
        "/../.env",
        "/api/jobs/not-a-uuid",
        "/api/jobs/../../.env",
    ],
)
async def test_no_arbitrary_static_files_or_openapi(app_factory, path):
    async with app_factory() as (_, client, _, _):
        response = await client.get(path)
        assert response.status_code in {404, 422}
        assert "traceback" not in response.text.lower()


@pytest.mark.parametrize("path", ["/api/config", "/does-not-exist", "/api/jobs/not-a-uuid"])
async def test_security_headers_on_success_and_error(app_factory, path):
    async with app_factory() as (_, client, _, _):
        response = await client.get(path)
        assert response.headers["cache-control"] == "no-store"
        assert response.headers["x-content-type-options"] == "nosniff"
        assert response.headers["referrer-policy"] == "no-referrer"
        assert "frame-ancestors 'none'" in response.headers["content-security-policy"]
        assert "unsafe-inline" not in response.headers["content-security-policy"]
        assert "access-control-allow-origin" not in response.headers


async def test_errors_never_expose_runner_exception(app_factory):
    async def failure(request, progress, created):
        raise RuntimeError("secret-key=D:/private/.env")

    async with app_factory(runner=failure) as (_, client, headers, _):
        job_id = (await client.post("/api/jobs", json={}, headers=headers)).json()["id"]
        job = await wait_terminal(client, job_id)
        assert job["status"] == "failed"
        assert "secret-key" not in json.dumps(job)


def test_constructing_app_does_not_create_db_or_load_configuration(tmp_path):
    db = tmp_path / "missing" / "jobs.db"
    app = create_app(db=db, output_dir=tmp_path / "out", env_file=tmp_path / "missing.env")
    assert app is not None and not db.exists()


@pytest.mark.parametrize("port", [0, -1, 65536, True, "8765"])
def test_port_validation(tmp_path, port):
    with pytest.raises(ValueError):
        create_app(db=tmp_path / "db", output_dir=tmp_path / "out", port=port)
