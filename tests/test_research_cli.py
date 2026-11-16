"""M1 CLI dispatch and the explicitly offline demo, without live credentials."""

import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from deep_research import cli
from deep_research.errors import ResearchError
from deep_research.research import ResearchRunError
from deep_research.research_store import ResearchStore


def test_help_uses_utf8_even_when_process_defaults_to_legacy_encoding(tmp_path):
    environment = {
        key: value for key, value in os.environ.items() if not key.startswith("DEEPSEEK_")
    }
    environment["PYTHONIOENCODING"] = "cp936"
    result = subprocess.run(
        [sys.executable, "-m", "deep_research.cli", "--help"],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        check=True,
        timeout=10,
    )
    assert "证据与引用报告" in result.stdout.decode("utf-8")
    assert "完全离线演示" in result.stdout.decode("utf-8")
    assert result.stderr == b""


def test_run_command_passes_paths_language_and_question_without_calling_smoke(monkeypatch, capsys):
    calls = []
    expected = {"status": "completed", "run_id": "offline-run", "report_path": "report.md"}

    async def fake_run(question, **options):
        calls.append((question, options))
        return expected

    monkeypatch.setattr(cli, "run_research", fake_run)
    monkeypatch.setattr(
        cli, "run_smoke", lambda *_: pytest.fail("The run command must not call smoke")
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "research",
            "run",
            "How does evidence work?",
            "--language",
            "en",
            "--env-file",
            "test.env",
            "--corpus",
            "corpus.json",
            "--db",
            "runs.sqlite",
            "--output-dir",
            "reports",
        ],
    )

    assert cli.main() == 0
    output = capsys.readouterr()
    assert json.loads(output.out) == expected
    assert output.err == ""
    assert calls == [
        (
            "How does evidence work?",
            {
                "env_file": Path("test.env"),
                "language": "en",
                "corpus": Path("corpus.json"),
                "db": Path("runs.sqlite"),
                "output_dir": Path("reports"),
            },
        )
    ]


def test_run_command_defaults_to_wikipedia_without_a_corpus(monkeypatch, capsys):
    calls = []

    async def fake_run(question, **options):
        calls.append((question, options))
        return {"status": "completed"}

    monkeypatch.setattr(cli, "run_research", fake_run)
    monkeypatch.setattr(sys, "argv", ["research", "run", "研究证据"])
    assert cli.main() == 0
    assert json.loads(capsys.readouterr().out)["status"] == "completed"
    assert calls == [
        (
            "研究证据",
            {
                "env_file": Path(".env"),
                "language": "zh",
                "corpus": None,
                "db": Path("data/research.sqlite3"),
                "output_dir": Path("outputs"),
            },
        )
    ]


def test_demo_command_dispatches_without_entering_live_research(monkeypatch, capsys):
    calls = []

    async def fake_demo(**options):
        calls.append(options)
        return {"status": "completed", "mode": "offline_demo"}

    monkeypatch.setattr(cli, "run_demo", fake_demo)
    monkeypatch.setattr(
        cli, "run_research", lambda *_: pytest.fail("Demo must not enter live research")
    )
    monkeypatch.setattr(
        sys, "argv", ["research", "demo", "--db", "demo.sqlite", "--output-dir", "demo-output"]
    )
    assert cli.main() == 0
    assert json.loads(capsys.readouterr().out)["mode"] == "offline_demo"
    assert calls == [{"db": Path("demo.sqlite"), "output_dir": Path("demo-output")}]


def test_show_command_passes_id_and_database(monkeypatch, capsys):
    calls = []

    def fake_show(run_id, db):
        calls.append((run_id, db))
        return {"id": run_id, "status": "completed"}

    monkeypatch.setattr(cli, "show_research", fake_show)
    monkeypatch.setattr(sys, "argv", ["research", "show", "saved-run", "--db", "saved.sqlite"])
    assert cli.main() == 0
    assert json.loads(capsys.readouterr().out) == {"id": "saved-run", "status": "completed"}
    assert calls == [("saved-run", Path("saved.sqlite"))]


def test_research_failure_prints_run_id_for_later_inspection(monkeypatch, capsys):
    async def fake_run(*args, **kwargs):
        raise ResearchRunError(ResearchError("no_evidence", "No original evidence."), "failed-run")

    monkeypatch.setattr(cli, "run_research", fake_run)
    monkeypatch.setattr(sys, "argv", ["research", "run", "Question"])
    assert cli.main() == 1
    output = capsys.readouterr()
    assert output.out == ""
    assert json.loads(output.err) == {
        "status": "error",
        "code": "no_evidence",
        "message": "No original evidence.",
        "http_status": None,
        "run_id": "failed-run",
        "status_saved": True,
    }


async def test_invalid_question_is_rejected_before_loading_credentials_or_creating_db(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(
        cli, "load_settings", lambda *_: pytest.fail("Invalid input must be rejected first")
    )
    database = tmp_path / "never-created.sqlite"
    with pytest.raises(ValidationError):
        await cli.run_research(
            "  ",
            env_file=Path("never-read.env"),
            language="zh",
            corpus=None,
            db=database,
            output_dir=tmp_path / "reports",
        )
    assert not database.exists()


async def test_real_offline_demo_never_loads_settings_and_saves_honest_fixture_metadata(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(
        cli, "load_settings", lambda *_: pytest.fail("Offline demo must not load credentials")
    )
    monkeypatch.setattr(
        cli, "DeepSeekProvider", lambda *_: pytest.fail("Offline demo must not create live model")
    )
    monkeypatch.setattr(
        cli, "WikipediaRetriever", lambda **_: pytest.fail("Demo must not create HTTP client")
    )
    database = tmp_path / "demo.sqlite"
    result = await cli.run_demo(db=database, output_dir=tmp_path / "reports")
    assert result["status"] == "completed"
    assert result["mode"] == "offline_demo"
    assert result["model_call_count"] == 2
    assert result["source_count"] == 2
    assert await asyncio.to_thread(Path(result["report_path"]).is_file)
    with ResearchStore(database) as store:
        bundle = store.get_report(result["run_id"])
        assert bundle["retrieval_mode"] == "offline_demo"
        assert all(call["provider"] == "offline_fixture" for call in bundle["model_calls"])
        assert all(call["usage"] is None for call in bundle["model_calls"])
        assert any("离线" in limitation for limitation in bundle["system_limitations"])


def test_show_missing_database_does_not_create_it(tmp_path):
    database = tmp_path / "missing.sqlite"
    with pytest.raises(ResearchError) as caught:
        cli.show_research("unknown", database)
    assert caught.value.code == "not_found"
    assert not database.exists()


def test_show_reads_saved_status_and_reports_unknown_run(tmp_path):
    database = tmp_path / "saved.sqlite"
    with ResearchStore(database) as store:
        run_id = store.create_run("Offline question")
        store.fail_run(run_id, "no_evidence")
    result = cli.show_research(run_id, database)
    assert result["id"] == run_id
    assert result["status"] == "failed"
    assert result["error_code"] == "no_evidence"
    with pytest.raises(ResearchError) as caught:
        cli.show_research("unknown", database)
    assert caught.value.code == "not_found"


def test_unexpected_error_returns_sanitized_json_instead_of_traceback(monkeypatch, capsys):
    async def fake_run(*args, **kwargs):
        raise RuntimeError("secret-token-and-private-upstream-payload")

    monkeypatch.setattr(cli, "run_research", fake_run)
    monkeypatch.setattr(sys, "argv", ["research", "run", "Question"])
    assert cli.main() == 1
    output = capsys.readouterr()
    assert output.out == ""
    detail = json.loads(output.err)
    assert set(detail) == {"status", "code", "message"}
    assert detail["status"] == "error"
    assert detail["code"] == "unexpected"
    assert isinstance(detail["message"], str) and detail["message"]
    assert "secret-token-and-private-upstream-payload" not in output.err
    assert "Traceback" not in output.err


def test_cli_exposes_status_save_failure_without_claiming_terminal_state(monkeypatch, capsys):
    async def fake_run(*args, **kwargs):
        raise ResearchRunError(
            ResearchError("rate_limit", "Model limited."), "uncertain-run", status_saved=False
        )

    monkeypatch.setattr(cli, "run_research", fake_run)
    monkeypatch.setattr(sys, "argv", ["research", "run", "Question"])
    assert cli.main() == 1
    output = capsys.readouterr()
    detail = json.loads(output.err)
    assert output.out == ""
    assert detail["code"] == "rate_limit"
    assert detail["run_id"] == "uncertain-run"
    assert detail["status_saved"] is False
    assert "无法确认" in detail["message"]
