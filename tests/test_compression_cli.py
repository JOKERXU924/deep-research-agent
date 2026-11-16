"""M4 opt-in CLI configuration and offline demos; no downloads or real model calls."""

import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from deep_research import cli, compression_demo, embeddings
from deep_research.errors import ResearchError
from deep_research.research_store import ResearchStore


def forbid_external_services(monkeypatch):
    monkeypatch.setattr(cli, "load_settings", lambda *_: pytest.fail("Unexpected key access"))
    monkeypatch.setattr(
        httpx, "AsyncClient", lambda *a, **kw: pytest.fail("Unexpected HTTP client")
    )
    monkeypatch.setattr(
        embeddings, "prepare_local_e5", lambda *_: pytest.fail("Implicit model download")
    )


def test_run_dag_dispatches_compression_with_explicit_backend_budget_and_memory(
    monkeypatch, capsys
):
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
            "原文证据如何保留？",
            "--compression",
            "hierarchical",
            "--embedding-backend",
            "local-e5",
            "--embedding-model-path",
            "models/local-e5",
            "--context-bytes",
            "2048",
            "--memory-scope",
            "study",
            "--memory-max-age-days",
            "7",
        ],
    )
    assert cli.main() == 0
    assert json.loads(capsys.readouterr().out)["status"] == "completed"
    assert calls == [
        (
            "原文证据如何保留？",
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
                "memory_scope": "study",
                "memory_max_age_days": 7,
                "compression": "hierarchical",
                "embedding_backend": "local-e5",
                "embedding_model_path": Path("models/local-e5"),
                "context_bytes": 2048,
            },
        )
    ]


@pytest.mark.parametrize(
    "arguments",
    [
        ["--embedding-backend", "local-e5"],
        ["--embedding-model-path", "models/local-e5"],
        ["--context-bytes", "1024"],
    ],
)
def test_embedding_options_cannot_implicitly_enable_compression(monkeypatch, capsys, arguments):
    forbid_external_services(monkeypatch)
    monkeypatch.setattr(
        cli, "run_dag_research", lambda *a, **kw: pytest.fail("Missing opt-in started research")
    )
    monkeypatch.setattr(sys, "argv", ["research", "run-dag", "Question", *arguments])
    assert cli.main() == 1
    output = capsys.readouterr()
    assert output.out == ""
    assert json.loads(output.err)["code"] == "compression_configuration"


@pytest.mark.parametrize(
    ("options", "code"),
    [
        ({"compression": "hierarchical", "context_bytes": 511}, "compression_configuration"),
        ({"compression": "hierarchical", "context_bytes": 24577}, "compression_configuration"),
        ({"compression": "not-supported"}, "compression_configuration"),
        ({"context_bytes": 1024}, "compression_configuration"),
        ({"compression": "hierarchical", "embedding_backend": "local-e5"}, "embedding_validation"),
        (
            {
                "compression": "hierarchical",
                "embedding_backend": "lexical",
                "embedding_model_path": Path("models/never-read"),
            },
            "embedding_validation",
        ),
    ],
)
async def test_invalid_compression_configuration_fails_before_keys_database_and_paid_calls(
    tmp_path, monkeypatch, options, code
):
    forbid_external_services(monkeypatch)
    database = tmp_path / "not-created.sqlite"
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
    assert caught.value.code == code
    assert not database.exists()
    assert not (tmp_path / "reports").exists()


async def test_e5_preflight_failure_happens_before_loading_paid_provider_settings(
    tmp_path, monkeypatch
):
    forbid_external_services(monkeypatch)
    calls = []

    class PreflightEncoder:
        async def encode_queries(self, texts):
            calls.append(texts)
            raise ResearchError("embedding_tokens", "Local token limit exceeded.")

    monkeypatch.setattr(
        cli, "build_compressor", lambda *a, **kw: SimpleNamespace(encoder=PreflightEncoder())
    )
    with pytest.raises(ResearchError) as caught:
        await cli.run_dag_research(
            "Question",
            env_file=tmp_path / "never-read.env",
            language="zh",
            corpus=None,
            db=tmp_path / "not-created.sqlite",
            output_dir=tmp_path / "reports",
            compression="hierarchical",
            embedding_backend="local-e5",
            embedding_model_path=tmp_path / "fake-model",
        )
    assert caught.value.code == "embedding_tokens"
    assert calls == [["Question"]]
    assert not (tmp_path / "not-created.sqlite").exists()


async def test_run_dag_assembles_compressor_and_memory_without_changing_retrieval_contract(
    tmp_path, monkeypatch
):
    forbid_external_services(monkeypatch)
    settings, retriever = object(), object()
    lifecycle, calls = [], []
    monkeypatch.setattr(cli, "load_settings", lambda *_: settings)
    monkeypatch.setattr(cli.LocalCorpusRetriever, "from_file", lambda *_: retriever)

    class FakeProvider:
        def __init__(self, config):
            assert config is settings

        async def __aenter__(self):
            lifecycle.append("opened")
            return self

        async def __aexit__(self, *_):
            lifecycle.append("closed")

    async def fake_pipeline(question, **kwargs):
        calls.append((question, kwargs))
        assert kwargs["memory"].stats(scope="study", source_mode="local_corpus")["items"] == 0
        return {"status": "completed"}

    monkeypatch.setattr(cli, "DeepSeekProvider", FakeProvider)
    monkeypatch.setattr(cli, "run_dag_pipeline", fake_pipeline)
    result = await cli.run_dag_research(
        "Question",
        env_file=tmp_path / "not-read.env",
        language="en",
        corpus=tmp_path / "fake-corpus.json",
        db=tmp_path / "research.sqlite",
        output_dir=tmp_path / "reports",
        memory_scope="study",
        compression="hierarchical",
        context_bytes=1024,
    )
    assert result == {"status": "completed"}
    assert lifecycle == ["opened", "closed"]
    assert len(calls) == 1 and calls[0][0] == "Question"
    options = calls[0][1]
    assert options["retriever"] is retriever
    assert options["source_mode"] == "local_corpus" and options["language"] == "en"
    assert options["memory_scope"] == "study" and options["memory_max_age_days"] == 30
    assert options["compressor"].options.max_evidence_bytes == 1024
    assert options["compressor"].encoder.kind == "lexical"


def test_demo_compression_dispatches_paths_and_explicit_scenario(monkeypatch, capsys):
    calls = []

    async def fake_demo(**options):
        calls.append(options)
        return {"status": "completed", "mode": "offline_demo"}

    monkeypatch.setattr(compression_demo, "run_compression_demo", fake_demo)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "research",
            "demo-compression",
            "--scenario",
            "tight",
            "--db",
            "demo.sqlite",
            "--output-dir",
            "reports",
        ],
    )
    assert cli.main() == 0
    assert json.loads(capsys.readouterr().out)["mode"] == "offline_demo"
    assert calls == [
        {
            "db": Path("demo.sqlite"),
            "output_dir": Path("reports"),
            "scenario": "tight",
            "embedding_backend": "lexical",
            "embedding_model_path": None,
        }
    ]


@pytest.mark.parametrize("scenario", ["normal", "tight", "deadline"])
async def test_real_compression_demos_are_offline_auditable_and_bounded(
    tmp_path, monkeypatch, scenario
):
    forbid_external_services(monkeypatch)
    database = tmp_path / "demo.sqlite"
    result = await compression_demo.run_compression_demo(
        db=database,
        output_dir=tmp_path / "reports",
        scenario=scenario,
    )
    assert result["status"] == "completed"
    assert result["mode"] == "offline_demo" and result["scenario"] == scenario
    with ResearchStore(database) as store:
        bundle = store.get_report(result["run_id"])
        assert store.get_run(result["run_id"])["status"] == "completed"
        assert bundle["evidence"] == store.get_evidence(result["run_id"])
    assert bundle["schema_version"] == 4
    assert bundle["retrieval_mode"] == "offline_demo"
    assert await asyncio.to_thread(Path(result["report_path"]).is_file)
    exported = await asyncio.to_thread(Path(result["evidence_path"]).read_text, encoding="utf-8")
    assert json.loads(exported) == bundle
    source_text = {source["source_id"]: source["text"] for source in bundle["sources"]}
    for item in bundle["evidence"]:
        assert source_text[item["source_id"]][item["start"] : item["end"]] == item["quote"]
    audit = result["compression"]
    if scenario == "deadline":
        assert result["model_call_count"] == 1
        assert result["report_kind"] == "evidence_digest" and result["degraded"] is True
        assert audit["status"] == "skipped_deadline"
    else:
        assert result["model_call_count"] == 2 and result["report_kind"] == "model_report"
        assert audit["status"] == "completed" and audit["encoder"]["kind"] == "lexical"
        assert audit["report_evidence_bytes"] <= (512 if scenario == "tight" else 8192)
        assert audit["report_request_bytes"] <= 64 * 1024


@pytest.mark.parametrize(
    ("command", "module", "function_name"),
    [
        ("embedding-prepare", embeddings, "prepare_local_e5"),
        ("embedding-check", compression_demo, "check_local_embedding"),
    ],
)
def test_embedding_commands_dispatch_only_the_explicit_local_model_path(
    tmp_path, monkeypatch, capsys, command, module, function_name
):
    forbid_external_services(monkeypatch)
    calls = []

    async def fake_operation(path):
        calls.append(path)
        return {"status": "ok", "operation": command}

    monkeypatch.setattr(module, function_name, fake_operation)
    path = tmp_path / "prepared-model"
    monkeypatch.setattr(sys, "argv", ["research", command, "--model-path", str(path)])
    assert cli.main() == 0
    assert json.loads(capsys.readouterr().out) == {"status": "ok", "operation": command}
    assert calls == [path]
    assert not path.exists()


def test_embedding_check_reports_local_model_failure_without_retry_or_download(
    tmp_path, monkeypatch, capsys
):
    forbid_external_services(monkeypatch)
    calls = []

    async def missing_model(path):
        calls.append(path)
        raise ResearchError("embedding_model", "Local model unavailable.")

    monkeypatch.setattr(compression_demo, "check_local_embedding", missing_model)
    path = tmp_path / "missing-model"
    monkeypatch.setattr(sys, "argv", ["research", "embedding-check", "--model-path", str(path)])
    assert cli.main() == 1
    output = capsys.readouterr()
    assert output.out == ""
    assert json.loads(output.err)["code"] == "embedding_model"
    assert calls == [path] and not path.exists()
