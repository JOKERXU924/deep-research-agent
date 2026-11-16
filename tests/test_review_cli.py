"""Explicit M5 CLI selection and local inspection without credentials or network."""

import asyncio
import json
import sys
from pathlib import Path

import httpx
import pytest

from deep_research import cli, review_demo
from deep_research.errors import ResearchError
from deep_research.research_store import ResearchStore
from deep_research.review_journal import ReviewJournal, ReviewJournalError
from deep_research.review_models import ReviewOptions


def forbid_external_services(monkeypatch):
    monkeypatch.setattr(cli, "load_settings", lambda *_: pytest.fail("Unexpected key access"))
    monkeypatch.setattr(
        httpx, "AsyncClient", lambda *a, **kw: pytest.fail("Unexpected HTTP access")
    )


def test_run_dag_dispatches_explicit_review_and_composable_memory_compression(monkeypatch, capsys):
    calls = []

    async def fake_run(question, **options):
        calls.append((question, options))
        return {"status": "completed"}

    monkeypatch.setattr(cli, "run_dag_research", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "research",
            "run-dag",
            "Evidence question",
            "--review",
            "red-blue",
            "--review-rounds",
            "1",
            "--review-threshold",
            "85",
            "--review-format-repairs",
            "0",
            "--compression",
            "hierarchical",
            "--context-bytes",
            "2048",
            "--memory-scope",
            "study",
        ],
    )
    assert cli.main() == 0
    assert json.loads(capsys.readouterr().out)["status"] == "completed"
    assert calls[0][0] == "Evidence question"
    options = calls[0][1]
    assert {
        key: options[key]
        for key in (
            "review",
            "review_rounds",
            "review_threshold",
            "review_format_repairs",
        )
    } == {
        "review": "red-blue",
        "review_rounds": 1,
        "review_threshold": 85,
        "review_format_repairs": 0,
    }
    assert options["compression"] == "hierarchical" and options["context_bytes"] == 2048
    assert options["memory_scope"] == "study"


def test_default_run_dag_does_not_pass_new_options_or_silently_enable_review(monkeypatch, capsys):
    calls = []

    async def fake_run(question, **options):
        calls.append(options)
        return {"status": "completed"}

    monkeypatch.setattr(cli, "run_dag_research", fake_run)
    monkeypatch.setattr(sys, "argv", ["research", "run-dag", "Question"])
    assert cli.main() == 0
    capsys.readouterr()
    assert not any(key.startswith("review") for key in calls[0])
    assert calls[0]["concurrency"] == 3 and calls[0]["max_replans"] == 1
    assert cli.build_review_options(None) is None


@pytest.mark.parametrize(
    "arguments",
    [
        ["--review-rounds", "1"],
        ["--review-threshold", "90"],
        ["--review-format-repairs", "0"],
    ],
)
def test_review_tuning_requires_explicit_opt_in_before_dispatch(monkeypatch, capsys, arguments):
    forbid_external_services(monkeypatch)
    monkeypatch.setattr(
        cli, "run_dag_research", lambda *a, **kw: pytest.fail("Missing review opt-in")
    )
    monkeypatch.setattr(sys, "argv", ["research", "run-dag", "Question", *arguments])
    assert cli.main() == 1
    assert json.loads(capsys.readouterr().err)["code"] == "review_configuration"


@pytest.mark.parametrize(
    "options",
    [
        {"review": "unknown"},
        {"review": "red-blue", "review_rounds": 3},
        {"review": "red-blue", "review_rounds": True},
        {"review": "red-blue", "review_threshold": -1},
        {"review": "red-blue", "review_threshold": 101},
        {"review": "red-blue", "review_format_repairs": 2},
    ],
)
async def test_invalid_review_configuration_precedes_credentials_database_and_embedding_work(
    tmp_path, monkeypatch, options
):
    forbid_external_services(monkeypatch)
    monkeypatch.setattr(
        cli, "build_compressor", lambda *a, **kw: pytest.fail("Premature encoder work")
    )
    database = tmp_path / "never-created.sqlite"
    with pytest.raises(ResearchError) as caught:
        await cli.run_dag_research(
            "Question",
            env_file=tmp_path / "never-read.env",
            language="zh",
            corpus=None,
            db=database,
            output_dir=tmp_path / "reports",
            **options,
        )
    assert caught.value.code == "review_configuration"
    assert not database.exists()


async def test_cli_assembles_a_real_review_journal_and_closes_it_after_the_pipeline(
    tmp_path, monkeypatch
):
    forbid_external_services(monkeypatch)
    settings, retriever = object(), object()
    monkeypatch.setattr(cli, "load_settings", lambda *_: settings)
    monkeypatch.setattr(cli.LocalCorpusRetriever, "from_file", lambda *_: retriever)
    recorded = []

    class FakeProvider:
        def __init__(self, supplied):
            assert supplied is settings

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            pass

    async def fake_pipeline(question, **options):
        recorded.append(options)
        assert isinstance(options["review_journal"], ReviewJournal)
        assert options["review_options"] == ReviewOptions(max_rounds=1, score_threshold=85)
        run_id = options["store"].create_run(question)
        options["review_journal"].append(run_id, {"kind": "initial"})
        return {"status": "completed", "run_id": run_id}

    monkeypatch.setattr(cli, "DeepSeekProvider", FakeProvider)
    monkeypatch.setattr(cli, "run_dag_pipeline", fake_pipeline)
    result = await cli.run_dag_research(
        "Question",
        env_file=tmp_path / "not-read.env",
        language="zh",
        corpus=tmp_path / "fake-corpus.json",
        db=tmp_path / "review.sqlite",
        output_dir=tmp_path / "reports",
        review="red-blue",
        review_rounds=1,
        review_threshold=85,
    )
    assert recorded[0]["retriever"] is retriever
    assert recorded[0]["source_mode"] == "local_corpus"
    with pytest.raises(ReviewJournalError) as caught:
        recorded[0]["review_journal"].snapshot(result["run_id"])
    assert caught.value.code == "review_audit_closed"


def test_show_review_dispatches_run_id_and_local_default_database(monkeypatch, capsys):
    forbid_external_services(monkeypatch)
    calls = []

    def fake_show(run_id, path):
        calls.append((run_id, path))
        return {"run": {"id": run_id}, "events": [], "model_calls": []}

    monkeypatch.setattr(cli, "show_review", fake_show)
    monkeypatch.setattr(sys, "argv", ["research", "show-review", "run-1"])
    assert cli.main() == 0
    assert json.loads(capsys.readouterr().out)["run"]["id"] == "run-1"
    assert calls == [("run-1", Path("data/review-demo.sqlite3"))]


def test_show_review_missing_database_does_not_create_it(tmp_path, monkeypatch, capsys):
    forbid_external_services(monkeypatch)
    path = tmp_path / "missing.sqlite"
    monkeypatch.setattr(sys, "argv", ["research", "show-review", "unknown", "--db", str(path)])
    assert cli.main() == 1
    assert json.loads(capsys.readouterr().err)["code"] == "not_found"
    assert not path.exists()


def test_demo_review_dispatches_only_explicit_local_arguments(monkeypatch, capsys):
    forbid_external_services(monkeypatch)
    calls = []

    async def fake_demo(**options):
        calls.append(options)
        return {"status": "completed", "mode": "offline_demo"}

    monkeypatch.setattr(review_demo, "run_review_demo", fake_demo)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "research",
            "demo-review",
            "--scenario",
            "oscillation",
            "--db",
            "demo.sqlite",
            "--output-dir",
            "reports",
        ],
    )
    assert cli.main() == 0
    assert json.loads(capsys.readouterr().out)["mode"] == "offline_demo"
    assert calls == [
        {"db": Path("demo.sqlite"), "output_dir": Path("reports"), "scenario": "oscillation"}
    ]


@pytest.mark.parametrize(
    ("scenario", "status", "stop_reason", "calls"),
    [
        ("passed", "passed", "passed", 1),
        ("repair", "passed", "passed", 3),
        ("oscillation", "unresolved", "oscillation", 4),
        ("stalled", "unresolved", "stalled", 2),
        ("max-rounds", "unresolved", "max_rounds", 5),
        ("format", "passed", "passed", 6),
        ("deadline", "interrupted", "global_timeout", 1),
    ],
)
async def test_all_seven_review_demos_are_offline_bounded_and_inspectable(
    tmp_path, monkeypatch, scenario, status, stop_reason, calls
):
    forbid_external_services(monkeypatch)
    database = tmp_path / "demo.sqlite"
    result = await review_demo.run_review_demo(
        db=database,
        output_dir=tmp_path / "reports",
        scenario=scenario,
    )
    assert result["status"] == "completed" and result["mode"] == "offline_demo"
    assert result["scenario"] == scenario
    assert result["review"]["status"] == status and result["review"]["stop_reason"] == stop_reason
    assert result["review_model_calls"] == calls
    assert result["model_call_count"] == 2 + calls <= 9
    assert result["degraded"] is (status != "passed")
    with ResearchStore(database) as store:
        bundle = store.get_report(result["run_id"])
        before = store.get_run(result["run_id"])
        shown = cli.show_review(result["run_id"], database)
        assert store.get_run(result["run_id"]) == before
        assert store.get_report(result["run_id"]) == bundle
    assert bundle["schema_version"] == 5 and bundle["retrieval_mode"] == "offline_demo"
    assert shown["events"] == bundle["review_events"]
    assert shown["model_calls"] == bundle["scheduler_audit"]["model_calls"]
    assert len(shown["events"]) <= 32
    assert all(item["status"] != "in_flight" for item in shown["model_calls"])
    saved = await asyncio.to_thread(Path(result["evidence_path"]).read_text, encoding="utf-8")
    assert json.loads(saved) == bundle
    if scenario == "deadline":
        assert result["report_kind"] == "evidence_digest"
        assert shown["model_calls"][-1]["status"] == "cancelled"
    elif scenario == "format":
        assert result["review"]["format_repairs"] == 1
        failed = [item for item in shown["model_calls"] if item["status"] == "failed"]
        assert len(failed) == 1 and failed[0]["details"]["error_code"] == "invalid_json"
