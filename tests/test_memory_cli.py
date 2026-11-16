"""Explicit memory namespaces and local-only commands, exercised without API credentials."""

import json
import sys
from pathlib import Path

import httpx
import pytest

from deep_research import cli, memory_demo
from deep_research.errors import ResearchError
from deep_research.memory_store import MemoryStore


@pytest.mark.parametrize(
    ("arguments", "method", "positional", "keywords"),
    [
        (
            ["memory-import", "seed-run", "--scope", "study"],
            "ingest_run",
            ("seed-run",),
            {"scope": "study"},
        ),
        (
            [
                "memory-search",
                "证据追踪",
                "--scope",
                "study",
                "--source-mode",
                "local_corpus",
                "--limit",
                "2",
                "--max-age-days",
                "7",
            ],
            "search",
            ("证据追踪",),
            {"scope": "study", "source_mode": "local_corpus", "limit": 2, "max_age_days": 7},
        ),
        (
            [
                "memory-conflicts",
                "--scope",
                "study",
                "--source-mode",
                "offline_demo",
                "--status",
                "all",
            ],
            "list_conflicts",
            (),
            {"scope": "study", "source_mode": "offline_demo", "status": "all"},
        ),
        (
            [
                "memory-resolve",
                "conflict-1",
                "--scope",
                "study",
                "--source-mode",
                "local_corpus",
                "--policy",
                "keep_both",
                "--reason",
                "保留双方待核验",
            ],
            "resolve_conflict",
            ("conflict-1",),
            {
                "scope": "study",
                "source_mode": "local_corpus",
                "policy": "keep_both",
                "reason": "保留双方待核验",
            },
        ),
        (
            ["memory-stats", "--scope", "study", "--source-mode", "wikipedia"],
            "stats",
            (),
            {"scope": "study", "source_mode": "wikipedia"},
        ),
    ],
)
def test_local_commands_dispatch_exact_namespace_arguments_without_model_settings(
    tmp_path, monkeypatch, capsys, arguments, method, positional, keywords
):
    database = tmp_path / "exists.sqlite"
    database.touch()
    calls, lifecycle = [], []

    class FakeMemory:
        def __init__(self, path):
            assert path == database

        def __enter__(self):
            lifecycle.append("open")
            return self

        def __exit__(self, *_):
            lifecycle.append("close")

        def __getattr__(self, name):
            def operation(*args, **kwargs):
                calls.append((name, args, kwargs))
                return [] if name == "list_conflicts" else {"operation": name}

            return operation

    monkeypatch.setattr(cli, "MemoryStore", FakeMemory)
    monkeypatch.setattr(cli, "load_settings", lambda *_: pytest.fail("Local command read a key"))
    monkeypatch.setattr(
        httpx, "AsyncClient", lambda *a, **kw: pytest.fail("Local command attempted HTTP")
    )
    monkeypatch.setattr(sys, "argv", ["research", *arguments, "--db", str(database)])
    assert cli.main() == 0
    output = capsys.readouterr()
    assert output.err == ""
    assert isinstance(json.loads(output.out), dict)
    assert calls == [(method, positional, keywords)]
    assert lifecycle == ["open", "close"]


@pytest.mark.parametrize(
    "arguments",
    [
        ["memory-import", "seed"],
        ["memory-search", "query"],
        ["memory-conflicts"],
        ["memory-stats"],
        ["memory-resolve", "id", "--policy", "keep_both", "--reason", "reviewed"],
        ["memory-resolve", "id", "--scope", "study", "--policy", "keep_both"],
    ],
)
def test_namespace_and_resolution_reason_are_required_before_opening_storage(
    monkeypatch, capsys, arguments
):
    monkeypatch.setattr(cli, "MemoryStore", lambda *_: pytest.fail("Invalid arguments opened DB"))
    monkeypatch.setattr(sys, "argv", ["research", *arguments])
    with pytest.raises(SystemExit) as caught:
        cli.main()
    assert caught.value.code == 2
    assert capsys.readouterr().err


def test_missing_database_is_reported_without_creating_one(tmp_path, monkeypatch, capsys):
    database = tmp_path / "missing.sqlite"
    monkeypatch.setattr(
        sys, "argv", ["research", "memory-stats", "--scope", "study", "--db", str(database)]
    )
    assert cli.main() == 1
    assert json.loads(capsys.readouterr().err)["code"] == "not_found"
    assert not database.exists()


def test_run_dag_passes_explicit_memory_scope_and_age(monkeypatch, capsys):
    calls = []

    async def fake_run(question, **kwargs):
        calls.append((question, kwargs))
        return {"status": "completed"}

    monkeypatch.setattr(cli, "run_dag_research", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "research",
            "run-dag",
            "evidence question",
            "--memory-scope",
            "study",
            "--memory-max-age-days",
            "7",
        ],
    )
    assert cli.main() == 0
    assert json.loads(capsys.readouterr().out)["status"] == "completed"
    assert calls[0][0] == "evidence question"
    assert calls[0][1]["memory_scope"] == "study"
    assert calls[0][1]["memory_max_age_days"] == 7


def test_memory_age_without_scope_does_not_silently_enable_memory(monkeypatch, capsys):
    monkeypatch.setattr(
        cli, "run_dag_research", lambda *a, **kw: pytest.fail("Missing scope started research")
    )
    monkeypatch.setattr(
        sys, "argv", ["research", "run-dag", "Question", "--memory-max-age-days", "7"]
    )
    assert cli.main() == 1
    assert json.loads(capsys.readouterr().err)["code"] == "memory_configuration"


@pytest.mark.parametrize(
    "options",
    [
        {"memory_scope": "invalid scope"},
        {"memory_scope": "study", "memory_max_age_days": 0},
        {"memory_scope": "study", "memory_max_age_days": 3651},
    ],
)
async def test_invalid_memory_options_are_rejected_before_loading_keys(
    tmp_path, monkeypatch, options
):
    monkeypatch.setattr(cli, "load_settings", lambda *_: pytest.fail("Invalid options read a key"))
    with pytest.raises(ResearchError) as caught:
        await cli.run_dag_research(
            "Question",
            env_file=Path("never-read.env"),
            language="zh",
            corpus=None,
            db=tmp_path / "never.sqlite",
            output_dir=tmp_path / "outputs",
            **options,
        )
    assert caught.value.code == "memory_configuration"
    assert not (tmp_path / "never.sqlite").exists()


def test_demo_memory_dispatches_to_explicit_offline_scenario(monkeypatch, capsys):
    calls = []

    async def fake_demo(**kwargs):
        calls.append(kwargs)
        return {"status": "completed", "mode": "offline_demo"}

    monkeypatch.setattr(memory_demo, "run_memory_demo", fake_demo)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "research",
            "demo-memory",
            "--scenario",
            "conflict",
            "--db",
            "demo.sqlite",
            "--output-dir",
            "reports",
        ],
    )
    assert cli.main() == 0
    assert json.loads(capsys.readouterr().out)["mode"] == "offline_demo"
    assert calls == [
        {"db": Path("demo.sqlite"), "output_dir": Path("reports"), "scenario": "conflict"}
    ]


@pytest.mark.parametrize("scenario", ["reuse", "conflict"])
async def test_real_memory_demo_is_offline_and_namespace_isolated(tmp_path, monkeypatch, scenario):
    monkeypatch.setattr(cli, "load_settings", lambda *_: pytest.fail("Offline demo read a key"))
    monkeypatch.setattr(
        httpx, "AsyncClient", lambda *a, **kw: pytest.fail("Offline demo attempted HTTP")
    )
    database = tmp_path / "demo.sqlite"
    result = await memory_demo.run_memory_demo(
        db=database, output_dir=tmp_path / "outputs", scenario=scenario
    )
    assert result["status"] == "completed"
    assert result["mode"] == "offline_demo"
    assert result["scenario"] == scenario
    assert result["scope"].startswith("demo-")
    assert result["stats"]["source_mode"] == "offline_demo"
    with MemoryStore(database) as memory:
        assert memory.stats(scope=result["scope"], source_mode="wikipedia")["items"] == 0
        assert memory.stats(scope="another", source_mode="offline_demo")["items"] == 0
    if scenario == "reuse":
        assert result["seed_run"]["memory_write"]["status"] == "saved"
        assert result["reuse_run"]["memory"]["recalled_count"] > 0
        assert result["reuse_run"]["memory_write"]["skipped_recalled"] > 0
    else:
        assert result["conflicts_before"]
        assert result["search_before"]["hits"] == []
        assert result["search_before"]["excluded_conflicts"] >= 2
        assert all(item["resolution"]["policy"] == "keep_both" for item in result["resolutions"])
        assert result["search_after"]["hits"]
        assert all(item["conflict_notes"] for item in result["search_after"]["hits"])
        assert result["stats"]["open_conflicts"] == 0
        assert result["stats"]["resolved_conflicts"] > 0


def test_conflicts_all_status_works_against_real_store(tmp_path, monkeypatch, capsys):
    from deep_research.research_store import ResearchStore

    database = tmp_path / "empty.sqlite"
    with ResearchStore(database), MemoryStore(database):
        pass
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "research",
            "memory-conflicts",
            "--scope",
            "study",
            "--status",
            "all",
            "--db",
            str(database),
        ],
    )
    assert cli.main() == 0
    assert json.loads(capsys.readouterr().out)["conflicts"] == []
