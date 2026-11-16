"""M2 CLI contracts and offline acceptance scenarios; no live network or credentials."""

import asyncio
import json
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from deep_research import cli
from deep_research.errors import ResearchError
from deep_research.research import ResearchRunError
from deep_research.research_store import ResearchStore


def test_run_dag_dispatches_explicit_budgets_and_paths(monkeypatch, capsys):
    calls = []

    async def fake_run(question, **options):
        calls.append((question, options))
        return {"status": "completed", "engine": "dag", "run_id": "offline-run"}

    monkeypatch.setattr(cli, "run_dag_research", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "research",
            "run-dag",
            "Evidence question",
            "--env-file",
            "test.env",
            "--language",
            "en",
            "--corpus",
            "sources.json",
            "--db",
            "dag.sqlite",
            "--output-dir",
            "reports",
            "--concurrency",
            "2",
            "--task-timeout",
            "4",
            "--total-timeout",
            "30",
            "--max-replans",
            "0",
        ],
    )
    assert cli.main() == 0
    output = capsys.readouterr()
    assert json.loads(output.out)["engine"] == "dag"
    assert output.err == ""
    assert calls == [
        (
            "Evidence question",
            {
                "env_file": Path("test.env"),
                "language": "en",
                "corpus": Path("sources.json"),
                "db": Path("dag.sqlite"),
                "output_dir": Path("reports"),
                "concurrency": 2,
                "task_timeout": 4.0,
                "total_timeout": 30.0,
                "max_replans": 0,
            },
        )
    ]


def test_run_dag_defaults_are_bounded(monkeypatch, capsys):
    calls = []

    async def fake_run(question, **options):
        calls.append(options)
        return {"status": "completed"}

    monkeypatch.setattr(cli, "run_dag_research", fake_run)
    monkeypatch.setattr(sys, "argv", ["research", "run-dag", "问题"])
    assert cli.main() == 0
    capsys.readouterr()
    assert calls == [
        {
            "env_file": Path(".env"),
            "language": "zh",
            "corpus": None,
            "db": Path("data/research.sqlite3"),
            "output_dir": Path("outputs"),
            "concurrency": 3,
            "task_timeout": 15.0,
            "total_timeout": 240.0,
            "max_replans": 1,
        }
    ]


def test_demo_dag_dispatches_scenario_and_concurrency(monkeypatch, capsys):
    calls = []

    async def fake_demo(**options):
        calls.append(options)
        return {"status": "completed", "mode": "offline_demo"}

    monkeypatch.setattr(cli, "run_dag_demo", fake_demo)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "research",
            "demo-dag",
            "--scenario",
            "replan",
            "--concurrency",
            "1",
            "--db",
            "demo.sqlite",
            "--output-dir",
            "demo-reports",
        ],
    )
    assert cli.main() == 0
    assert json.loads(capsys.readouterr().out)["mode"] == "offline_demo"
    assert calls == [
        {
            "db": Path("demo.sqlite"),
            "output_dir": Path("demo-reports"),
            "scenario": "replan",
            "concurrency": 1,
        }
    ]


def test_show_dag_dispatches_id_and_database(monkeypatch, capsys):
    calls = []

    def fake_show(run_id, db):
        calls.append((run_id, db))
        return {"run": {"id": run_id}, "audit": {"batches": []}}

    monkeypatch.setattr(cli, "show_dag", fake_show)
    monkeypatch.setattr(sys, "argv", ["research", "show-dag", "saved-run", "--db", "saved.sqlite"])
    assert cli.main() == 0
    assert json.loads(capsys.readouterr().out)["run"]["id"] == "saved-run"
    assert calls == [("saved-run", Path("saved.sqlite"))]


@pytest.mark.parametrize("scenario", ["normal", "replan", "deadline"])
async def test_offline_dag_scenarios_are_labeled_and_need_no_live_provider(
    tmp_path, monkeypatch, scenario
):
    monkeypatch.setattr(
        cli, "load_settings", lambda *_: pytest.fail("Offline scenario must not load credentials")
    )
    monkeypatch.setattr(
        cli, "DeepSeekProvider", lambda *_: pytest.fail("Offline scenario must not create a model")
    )
    monkeypatch.setattr(
        cli, "WikipediaRetriever", lambda **_: pytest.fail("Offline scenario must not use HTTP")
    )
    database = tmp_path / "demo.sqlite"
    result = await cli.run_dag_demo(db=database, output_dir=tmp_path / "reports", scenario=scenario)
    assert result["status"] == "completed"
    assert result["mode"] == "offline_demo"
    assert result["scenario"] == scenario
    assert result["engine"] == "dag"
    assert result["counters"]["model"] <= 3
    assert result["counters"]["search"] <= 6
    assert result["counters"]["fetch"] <= 8
    if scenario == "normal":
        assert result["degraded"] is False
        assert result["replan_count"] == 0
    elif scenario == "replan":
        assert result["degraded"] is True
        assert result["replan_count"] == 1
        assert result["model_call_count"] == 3
    else:
        assert result["degraded"] is True
        assert result["report_kind"] == "evidence_digest"
        assert result["model_call_count"] == 1
    with ResearchStore(database) as store:
        bundle = store.get_report(result["run_id"])
        assert bundle["retrieval_mode"] == "offline_demo"
        assert any("离线" in item for item in bundle["system_limitations"])
        assert all(call["provider"] == "offline_fixture" for call in bundle["model_calls"])
    detail = cli.show_dag(result["run_id"], database)
    assert detail["run"]["status"] == "completed"
    assert detail["audit"]["summary"]["report_kind"] == result["report_kind"]
    report_text = await asyncio.to_thread(Path(result["report_path"]).read_text, encoding="utf-8")
    assert r"offline\_demo" in report_text


@pytest.mark.parametrize(
    "invalid",
    [
        {"concurrency": 0},
        {"concurrency": 9},
        {"task_timeout": 0.0},
        {"total_timeout": float("inf")},
        {"max_replans": 2},
    ],
)
async def test_invalid_dag_budget_is_rejected_before_secrets_or_database(
    tmp_path, monkeypatch, invalid
):
    monkeypatch.setattr(
        cli, "load_settings", lambda *_: pytest.fail("Invalid budget must be rejected first")
    )
    database = tmp_path / "never-created.sqlite"
    with pytest.raises(ValidationError):
        await cli.run_dag_research(
            "Question",
            env_file=Path("never-read.env"),
            language="zh",
            corpus=None,
            db=database,
            output_dir=tmp_path / "reports",
            **invalid,
        )
    assert not database.exists()


def test_show_dag_missing_database_does_not_create_it(tmp_path):
    database = tmp_path / "missing.sqlite"
    with pytest.raises(ResearchError) as caught:
        cli.show_dag("missing-run", database)
    assert caught.value.code == "not_found"
    assert not database.exists()


def test_dag_error_output_distinguishes_audit_and_status_persistence(monkeypatch, capsys):
    async def fake_run(*args, **kwargs):
        error = ResearchRunError(
            ResearchError("dag_audit_io", "Audit unavailable."), "failed-run", status_saved=True
        )
        error.audit_saved = False
        raise error

    monkeypatch.setattr(cli, "run_dag_research", fake_run)
    monkeypatch.setattr(sys, "argv", ["research", "run-dag", "Question"])
    assert cli.main() == 1
    output = capsys.readouterr()
    assert output.out == ""
    detail = json.loads(output.err)
    assert detail["code"] == "dag_audit_io"
    assert detail["run_id"] == "failed-run"
    assert detail["status_saved"] is True
    assert detail["audit_saved"] is False
