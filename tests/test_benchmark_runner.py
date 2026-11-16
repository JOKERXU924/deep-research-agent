import asyncio
import json
from pathlib import Path

import pytest

from deep_research import benchmark_runner as runner
from deep_research.benchmark_cli import _write_new, judge_saved
from deep_research.benchmark_data import load_benchmark
from deep_research.errors import ResearchError
from deep_research.model_profiles import ModelProfile
from deep_research.research import ResearchRunError


@pytest.fixture
async def experiment(tmp_path):
    result = await runner.run_benchmark(
        output_dir=tmp_path, presets=("baseline", "full", "tight-budget"), limit=2
    )
    return Path(result["results_path"])


def artifact_from(root):
    return runner.read_json(next(root.glob("*/results.json")))


async def test_six_presets_use_real_engine_and_keep_failures(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("offline evaluation must not load any credential")

    monkeypatch.setattr(ModelProfile, "load_settings", forbidden)
    result = await runner.run_benchmark(output_dir=tmp_path, presets=tuple(runner.PRESETS), limit=2)
    artifact = runner.read_json(Path(result["results_path"]))
    assert len(artifact["records"]) == 12
    assert artifact["manifest"]["planned_call_ceiling"] == 56
    assert artifact["manifest"]["memory"] == "disabled"
    assert result["mode"] == "offline_fixture" and result["quality_claim"] is False
    for row in artifact["records"]:
        if row["preset"] == "tight-budget":
            assert row["status"] == "failed"
            assert row["error_code"] == "model_budget_exceeded"
            assert row["bundle_path"] is None
            assert row["model_budget"]["reserved"]["calls"] == 1
            assert row["metrics"]["citation_coverage"] == 0
        else:
            assert row["status"] == "completed"
            assert row["metrics"]["citation_validity"] == 1
            assert row["model_budget"]["usage_unknown_calls"] >= 2
    assert result["summary"]["tight-budget"]["attempted"] == 2
    assert result["summary"]["tight-budget"]["metrics"]["citation_coverage"]["scored_cases"] == 2
    assert "不是" not in Path(result["report_path"]).name


@pytest.mark.parametrize(
    "options",
    [
        {"limit": 0},
        {"limit": 51},
        {"limit": True},
        {"repeats": 0},
        {"repeats": 4},
        {"repeats": True},
        {"seed": -1},
        {"seed": True},
        {"seed": 2**32},
        {"presets": ()},
        {"presets": ("no-such",)},
        {"presets": ("baseline", "baseline")},
        {"max_total_model_calls": 0},
        {"max_total_model_calls": True},
        {"max_total_model_calls": 1},
        {"allow_live": True},
        {"profiles_file": Path("absent.toml")},
        {"profile_id": "some"},
        {"allow_live": 1},
    ],
)
async def test_preflight_rejects_before_output_and_calls(tmp_path, options, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("provider not allowed before validation")

    monkeypatch.setattr(runner, "BenchmarkFixtureProvider", forbidden)
    with pytest.raises(ResearchError):
        await runner.run_benchmark(output_dir=tmp_path / "out", **options)
    assert not (tmp_path / "out").exists()


async def test_relative_output_paths(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = await runner.run_benchmark(output_dir=Path("relative"), limit=1)
    assert Path(result["results_path"]).is_absolute()
    row = runner.read_json(Path(result["results_path"]))["records"][0]
    assert row["status"] == "completed"


async def test_gold_never_enters_generation_prompts(tmp_path, monkeypatch):
    dataset = load_benchmark().model_dump(mode="json")
    dataset["cases"] = dataset["cases"][:1]
    sentinel = "PRIVATE_GOLD_SENTINEL_DO_NOT_SEND_8712"
    dataset["cases"][0]["gold"]["answer_aliases"] = [sentinel]
    dataset["cases"][0]["gold"]["facts"][0]["statement"] = sentinel
    data_path = tmp_path / "dataset.json"
    data_path.write_text(json.dumps(dataset), encoding="utf-8")
    original = runner.BenchmarkFixtureProvider.complete
    seen = []

    async def checked(self, request):
        seen.append(request.model_dump_json())
        return await original(self, request)

    monkeypatch.setattr(runner.BenchmarkFixtureProvider, "complete", checked)
    await runner.run_benchmark(
        output_dir=tmp_path / "out", dataset_path=data_path, presets=("full",), limit=1
    )
    assert len(seen) == 3
    assert all(sentinel not in request for request in seen)


async def test_seed_reproduces_selection_order_not_claiming_model_determinism(tmp_path):
    first = await runner.run_benchmark(output_dir=tmp_path, limit=2)
    second = await runner.run_benchmark(output_dir=tmp_path, limit=2)
    left = runner.read_json(Path(first["results_path"]))["manifest"]
    right = runner.read_json(Path(second["results_path"]))["manifest"]
    assert left["case_ids"] == right["case_ids"]
    assert left["task_order"] == right["task_order"]
    assert left["experiment_id"] != right["experiment_id"]


async def test_live_gate_uses_selected_profile_and_same_engine(tmp_path, monkeypatch):
    profiles = tmp_path / "profiles.toml"
    profiles.write_text(
        'version=1\ndefault_profile="test"\n[[profiles]]\n'
        'id="test"\nlabel="Local fixture"\nprovider="vllm"\n'
        'model="test-model"\nbase_url="http://127.0.0.1:8000/v1"\n',
        encoding="utf-8",
    )
    instances = []

    class ContextFixture(runner.BenchmarkFixtureProvider):
        def __init__(self, settings):
            assert settings.model == "test-model"
            self.closed = False
            instances.append(self)

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            self.closed = True

    monkeypatch.setattr(runner, "ChatCompatibleProvider", ContextFixture)
    result = await runner.run_benchmark(
        output_dir=tmp_path / "out",
        allow_live=True,
        profiles_file=profiles,
        profile_id="test",
        limit=1,
    )
    assert result["mode"] == "live_model_frozen_corpus"
    assert len(instances) == 1 and instances[0].closed
    artifact = runner.read_json(Path(result["results_path"]))
    assert artifact["manifest"]["model_profile"]["id"] == "test"


@pytest.mark.parametrize("cleanup_fails", [False, True])
async def test_cancel_stops_matrix_and_preserves_original(tmp_path, monkeypatch, cleanup_fails):
    entered = asyncio.Event()
    count = 0

    async def blocked(self, request):
        nonlocal count
        count += 1
        entered.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(runner.BenchmarkFixtureProvider, "complete", blocked)
    original = runner.save_json

    def fail_cleanup(path, value):
        if cleanup_fails and path.name == "results.json" and value["records"]:
            raise OSError("disk unavailable")
        original(path, value)

    monkeypatch.setattr(runner, "save_json", fail_cleanup)
    task = asyncio.create_task(runner.run_benchmark(output_dir=tmp_path, limit=3))
    await asyncio.wait_for(entered.wait(), 2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert count == 1
    saved = artifact_from(tmp_path)
    if not cleanup_fails:
        assert saved["manifest"]["status"] == "cancelled"
        assert len(saved["records"]) == 1
        assert saved["records"][0]["model_budget"]["usage_unknown_calls"] == 1
    else:
        assert saved["manifest"]["status"] == "running"  # never invent a saved terminal state


async def test_audit_failure_stops_before_first_external_call(tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise ResearchError("dag_audit_io", "fixture")

    async def forbidden(*args, **kwargs):
        pytest.fail("broken audit must not dispatch a model")

    monkeypatch.setattr(runner.DagJournal, "start_model_call", fail)
    monkeypatch.setattr(runner.BenchmarkFixtureProvider, "complete", forbidden)
    with pytest.raises(ResearchError):
        await runner.run_benchmark(output_dir=tmp_path, limit=3)
    assert len(artifact_from(tmp_path)["records"]) == 1


async def test_error_without_dynamic_audit_attribute_stops_conservatively(tmp_path, monkeypatch):
    async def failure(*args, **kwargs):
        raise ResearchRunError(ResearchError("store_io", "fixture"), "missing-run")

    monkeypatch.setattr(runner, "run_dag_pipeline", failure)
    with pytest.raises(ResearchError) as error:
        await runner.run_benchmark(output_dir=tmp_path, limit=3)
    assert error.value.code == "benchmark_audit"
    row = artifact_from(tmp_path)["records"][0]
    assert row["error_code"] == "store_io"


async def test_repeat_means_before_case_bootstrap(experiment):
    result = runner.compare_results(experiment, candidate="tight-budget")
    assert result["pairs_used"] == 2
    assert result["mean_delta"] == -1
    assert result["cohen_dz"] is None
    assert result["quality_claim"] is False


@pytest.mark.parametrize("raw", ['{"a":1,"a":2}', '{"a":NaN}', '{"a":1e999}', "[]bad"])
def test_bounded_json_rejects_ambiguous_or_nonfinite(tmp_path, raw):
    path = tmp_path / "bad.json"
    path.write_text(raw, encoding="utf-8")
    with pytest.raises(ResearchError):
        runner.read_json(path)


def test_json_limit_and_new_output_never_overwrite(tmp_path):
    path = tmp_path / "data.json"
    path.write_text('{"a":1}', encoding="utf-8")
    with pytest.raises(ResearchError):
        runner.read_json(path, max_bytes=3)
    with pytest.raises(ResearchError):
        _write_new(path, {"changed": True})
    assert json.loads(path.read_text()) == {"a": 1}


async def test_saved_judge_binds_dataset_candidate_and_own_identity(experiment):
    artifact = runner.read_json(experiment)
    case_id = artifact["manifest"]["case_ids"][0]
    before = experiment.read_bytes()
    result = await judge_saved(experiment, case_id=case_id)
    assert result["method"] == "offline_fixture" and result["calibrated"] is False
    assert result["model_profile"]["model"] == "scripted-judge"
    assert result["model_budget"]["reserved"]["calls"] == 1
    assert all(item["label"] == "insufficient" for item in result["claims"])
    assert experiment.read_bytes() == before


async def test_judge_rejects_changed_report_before_calls(experiment):
    artifact = runner.read_json(experiment)
    row = next(row for row in artifact["records"] if row["preset"] == "baseline")
    path = experiment.parent / row["bundle_path"]
    bundle = runner.read_json(path)
    bundle["report"]["title"] = "modified"
    runner.save_json(path, bundle)
    with pytest.raises(ResearchError) as error:
        await judge_saved(experiment, case_id=row["case_id"])
    assert error.value.code == "benchmark_artifact"
    assert not (experiment.parent / "judgements").exists()


async def test_judge_never_accepts_report_path_escape(experiment, tmp_path):
    artifact = runner.read_json(experiment)
    row = next(row for row in artifact["records"] if row["preset"] == "baseline")
    row["bundle_path"] = str(tmp_path / "outside.json")
    runner.save_json(experiment, artifact)
    with pytest.raises(ResearchError):
        await judge_saved(experiment, case_id=row["case_id"])


@pytest.mark.parametrize(
    "options",
    [
        {"allow_live": True},
        {"profile_id": "x"},
        {"repeat": True},
        {"repeat": 3},
    ],
)
async def test_judge_gates_before_read(tmp_path, options):
    with pytest.raises(ResearchError):
        await judge_saved(tmp_path / "not-found.json", case_id="rb-001", **options)


async def test_late_cancel_after_commit_preserves_completed_row(tmp_path, monkeypatch):
    profiles = tmp_path / "profiles.toml"
    profiles.write_text(
        'version=1\ndefault_profile="test"\n[[profiles]]\n'
        'id="test"\nlabel="Local fixture"\nprovider="vllm"\n'
        'model="test-model"\nbase_url="http://127.0.0.1:8000/v1"\n',
        encoding="utf-8",
    )
    instances = []

    class LateCancel(runner.BenchmarkFixtureProvider):
        def __init__(self, settings):
            instances.append(self)

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            raise asyncio.CancelledError

    monkeypatch.setattr(runner, "ChatCompatibleProvider", LateCancel)
    with pytest.raises(asyncio.CancelledError):
        await runner.run_benchmark(
            output_dir=tmp_path / "out", allow_live=True, profiles_file=profiles, limit=3
        )
    saved = artifact_from(tmp_path / "out")
    assert saved["manifest"]["status"] == "cancelled"
    assert len(instances) == len(saved["records"]) == 1
    assert saved["records"][0]["status"] == "completed"
    path = next((tmp_path / "out").glob("*/results.json"))
    assert runner.load_verified_results(path)["records"][0]["metrics"]["citation_coverage"] == 1


async def test_snapshot_bug_cannot_replace_original_cancel(tmp_path, monkeypatch):
    entered = asyncio.Event()

    async def blocked(*args):
        entered.set()
        await asyncio.Event().wait()

    def broken_snapshot(*args):
        raise RuntimeError("unexpected cleanup fault")

    monkeypatch.setattr(runner.BenchmarkFixtureProvider, "complete", blocked)
    monkeypatch.setattr(runner.DagJournal, "snapshot", broken_snapshot)
    task = asyncio.create_task(runner.run_benchmark(output_dir=tmp_path, limit=3))
    await asyncio.wait_for(entered.wait(), 2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    record = artifact_from(tmp_path)["records"][0]
    assert record["status"] == "cancelled" and record["model_budget"] is None
    assert record["audit_error"] == "benchmark_audit"


async def test_recomputed_metrics_reject_self_consistent_modified_score(experiment):
    artifact = runner.read_json(experiment)
    row = next(r for r in artifact["records"] if r["preset"] == "baseline")
    row["metrics"]["counts"]["matched_gold_facts"] = 0
    row["metrics"]["gold_fact_coverage"] = 0.0
    artifact["summary"] = runner.summarize(artifact["records"], artifact["manifest"])
    runner.save_json(experiment, artifact)
    with pytest.raises(ResearchError) as error:
        runner.compare_results(experiment)
    assert error.value.code == "benchmark_artifact"


async def test_dataset_hash_change_is_rejected(experiment):
    data_path = experiment.parent / "dataset.json"
    dataset = runner.read_json(data_path)
    dataset["description"] += " changed"
    runner.save_json(data_path, dataset)
    with pytest.raises(ResearchError):
        runner.compare_results(experiment)


@pytest.mark.parametrize("cancelled", [True, False])
async def test_judge_failed_cleanup_preserves_original(experiment, monkeypatch, cancelled):
    from deep_research import benchmark_cli, benchmark_judge

    async def fail(**kwargs):
        if cancelled:
            raise asyncio.CancelledError("original cancel")
        raise ResearchError("original_model_error", "fixture")

    original = benchmark_cli.save_json

    def fail_final(path, value):
        if value["status"] != "running":
            raise OSError("cleanup failed")
        original(path, value)

    monkeypatch.setattr(benchmark_judge, "judge_report", fail)
    monkeypatch.setattr(benchmark_cli, "save_json", fail_final)
    case_id = runner.read_json(experiment)["manifest"]["case_ids"][0]
    expected = asyncio.CancelledError if cancelled else ResearchError
    with pytest.raises(expected) as error:
        await judge_saved(experiment, case_id=case_id)
    if not cancelled:
        assert error.value.code == "original_model_error"
