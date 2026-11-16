"""M7 public entrypoints use only fake models/retrieval and temporary stores."""

import asyncio
import json
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import uuid4

import httpx
import pytest

from deep_research import cli, web_runner
from deep_research.budget_demo import run_budget_demo
from deep_research.dag_demo import DagDemoRetriever
from deep_research.errors import ResearchError
from deep_research.model_budget import ModelBudgetLimits
from deep_research.model_profiles import ModelProfile, ProfileRegistry
from deep_research.providers import chat_compatible
from deep_research.research_store import ResearchStore
from deep_research.review_demo import ReviewDemoProvider
from deep_research.web_app import create_app
from deep_research.web_models import WebRequest


def registry(max_calls=9):
    return ProfileRegistry(
        version=1,
        default_profile="local-a",
        profiles=tuple(
            ModelProfile(
                id=f"local-{name}",
                label=f"本地 {name}",
                provider="vllm",
                model=f"fixture-{name}",
                base_url="http://127.0.0.1:8000/v1",
                budget=ModelBudgetLimits(max_calls=max_calls),
            )
            for name in ("a", "b")
        ),
    )


@pytest.fixture
def fake_live(monkeypatch):
    selected = []

    class Provider(ReviewDemoProvider):
        def __init__(self, settings):
            selected.append(settings)
            super().__init__("repair")

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
            pass

    monkeypatch.setattr(chat_compatible, "ChatCompatibleProvider", Provider)
    monkeypatch.setattr(web_runner, "WikipediaRetriever", Retriever)
    monkeypatch.setattr(cli, "WikipediaRetriever", Retriever)
    monkeypatch.setattr(web_runner, "load_settings", lambda *_: pytest.fail("legacy secret read"))
    monkeypatch.setattr(cli, "load_settings", lambda *_: pytest.fail("legacy secret read"))
    return selected


@asynccontextmanager
async def web(tmp_path, *, profiles=None, allow_live=True):
    app = create_app(
        db=tmp_path / "web.db",
        output_dir=tmp_path / "out",
        profiles=profiles,
        allow_live=allow_live,
    )
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1:8765"
        ) as client:
            config = (await client.get("/api/config")).json()
            yield client, config


async def submit(client, config, **changes):
    # Live entrypoints apply the same topic gate even when retrieval is a fake.
    # Keep the fixture question aligned with both DagDemoRetriever documents.
    data = {"mode": "live", "question": "异步等待与并发限制", "confirm_live": True, **changes}
    return await client.post(
        "/api/jobs",
        json=data,
        headers={"X-CSRF-Token": config["csrf_token"], "Idempotency-Key": str(uuid4())},
    )


async def terminal(client, job_id):
    async with asyncio.timeout(3):
        while True:
            job = (await client.get(f"/api/jobs/{job_id}")).json()
            if job["status"] in {"completed", "failed", "cancelled", "interrupted"}:
                return job
            await asyncio.sleep(0.005)


@pytest.mark.parametrize(
    "scenario,calls,status",
    [
        ("normal", 5, "completed"),
        ("calls", 2, "failed"),
        ("input", 0, "failed"),
        ("output", 0, "failed"),
        ("unknown", 5, "completed"),
    ],
)
async def test_budget_demos_are_durable_and_offline(tmp_path, monkeypatch, scenario, calls, status):
    monkeypatch.setattr(httpx, "AsyncClient", lambda *a, **kw: pytest.fail("network"))
    result = await run_budget_demo(db=tmp_path / "x.db", output_dir=tmp_path, scenario=scenario)
    assert result["status"] == status
    assert result["model_budget"]["reserved"]["calls"] == calls
    assert result["model_profile"]["provider"] == "offline_demo"
    if scenario == "unknown":
        assert result["model_budget"]["usage_unknown_calls"] == 5
        assert result["model_budget"]["usage_complete"] is False
    elif scenario == "normal":
        assert result["model_budget"]["known_usage"]["total_tokens"] == 75
        assert result["model_budget"]["usage_complete"] is True
    with ResearchStore(tmp_path / "x.db") as store:
        assert store.get_run(result["run_id"])["status"] == status
        assert (store.get_report(result["run_id"]) is None) == (status == "failed")


async def test_profile_switch_is_per_new_job_and_audit_survives(tmp_path, fake_live):
    profiles = registry()
    async with web(tmp_path, profiles=profiles) as (client, config):
        assert config["default_profile"] == "local-a"
        serialized = json.dumps(config)
        assert (
            "base_url" not in serialized
            and "api_key" not in serialized
            and "8000" not in serialized
        )
        first = (await submit(client, config)).json()
        assert first["request"]["profile_id"] == "local-a"
        second = (await submit(client, config, profile_id="local-b")).json()
        for job, model in ((first, "fixture-a"), (second, "fixture-b")):
            result = await terminal(client, job["id"])
            assert result["status"] == "completed"
            assert result["result"]["model_profile"]["model"] == model
            audit = (await client.get(f"/api/jobs/{job['id']}/model-audit")).json()
            assert audit["model_profile"]["model"] == model
            assert audit["model_budget"]["reserved"]["calls"] == 5
            assert len(audit["calls"]) == 5
            report = (await client.get(f"/api/jobs/{job['id']}/report")).json()
            assert report["model_profile"]["model"] == model
        assert [item.model for item in fake_live] == ["fixture-a", "fixture-b"]
    async with web(tmp_path, profiles=profiles) as (client, _):
        audit = (await client.get(f"/api/jobs/{first['id']}/model-audit")).json()
        assert audit["model_budget"]["reserved"]["calls"] == 5


async def test_failed_budget_keeps_public_ledger_and_has_no_report(tmp_path, fake_live):
    async with web(tmp_path, profiles=registry(max_calls=2)) as (client, config):
        response = await submit(client, config)
        job = await terminal(client, response.json()["id"])
        assert job["status"] == "failed" and job["error_code"] == "model_budget_exceeded"
        audit = (await client.get(f"/api/jobs/{job['id']}/model-audit")).json()
        assert audit["model_budget"]["reserved"]["calls"] == 2
        assert len(audit["calls"]) == 2
        assert (await client.get(f"/api/jobs/{job['id']}/report")).status_code == 409


@pytest.mark.parametrize(
    "change",
    [
        {"profile_id": "missing"},
        {"base_url": "https://evil.test"},
        {"api_key": "secret"},
        {"budget": {"max_calls": 9}},
        {"profile_id": "../private"},
    ],
)
async def test_unregistered_configuration_rejected_before_submission(tmp_path, fake_live, change):
    async with web(tmp_path, profiles=registry()) as (client, config):
        assert (await submit(client, config, **change)).status_code == 422
        assert (await client.get("/api/jobs")).json()["jobs"] == []
        assert fake_live == []


async def test_profile_does_not_bypass_live_gate_or_demo_restrictions(tmp_path, fake_live):
    async with web(tmp_path, profiles=registry(), allow_live=False) as (client, config):
        assert (await submit(client, config, profile_id="local-a")).status_code == 403
        assert fake_live == []
    with pytest.raises(ValueError):
        WebRequest(profile_id="local-a")


def test_profiles_cli_is_safe_and_does_not_load_credentials(tmp_path, monkeypatch, capsys):
    import deep_research.model_profiles as module

    monkeypatch.setattr(module, "load_profiles", lambda _: registry())
    monkeypatch.setattr(ModelProfile, "load_settings", lambda *a: pytest.fail("secret read"))
    monkeypatch.setattr(sys, "argv", ["research", "profiles", "--profiles", "example.toml"])
    assert cli.main() == 0
    result = json.loads(capsys.readouterr().out)
    assert result["credentials_checked"] is False
    assert len(result["profiles"]) == 2


async def test_cli_selected_profile_and_tighter_budget(tmp_path, monkeypatch, fake_live):
    import deep_research.model_profiles as module

    monkeypatch.setattr(module, "load_profiles", lambda _: registry())
    result = await cli.run_dag_research(
        "异步等待与并发限制",
        env_file=tmp_path / "missing.env",
        language="zh",
        corpus=None,
        db=tmp_path / "x.db",
        output_dir=tmp_path,
        profiles_file=Path("not-read.toml"),
        profile_id="local-b",
        max_model_calls=2,
    )
    assert result["model_profile"]["id"] == "local-b"
    assert result["model_budget"]["effective_max_calls"] == 2
    assert result["model_budget"]["reserved"]["calls"] == 2


@pytest.mark.parametrize("limit", [0, 10, True])
async def test_cli_cannot_raise_profile_cap(tmp_path, monkeypatch, fake_live, limit):
    import deep_research.model_profiles as module

    monkeypatch.setattr(module, "load_profiles", lambda _: registry())
    with pytest.raises(ResearchError):
        await cli.run_dag_research(
            "问题",
            env_file=tmp_path / "missing.env",
            language="zh",
            corpus=None,
            db=tmp_path / "x.db",
            output_dir=tmp_path,
            profiles_file=Path("not-read"),
            max_model_calls=limit,
        )
    assert not (tmp_path / "x.db").exists() and fake_live == []
