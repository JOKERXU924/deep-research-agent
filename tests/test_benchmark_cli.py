"""Exercise M8 command boundaries with local synthetic data and fake providers."""

import hashlib
import json
import sys
from pathlib import Path

import pytest

from deep_research import benchmark_cli as commands
from deep_research import benchmark_runner as runner
from deep_research import cli
from deep_research.benchmark_data import dataset_fingerprint, load_benchmark
from deep_research.errors import ResearchError
from deep_research.model_profiles import ModelProfile


def invoke(monkeypatch, capsys, *args):
    monkeypatch.setattr(sys, "argv", ["research", *map(str, args)])
    status = cli.main()
    captured = capsys.readouterr()
    return status, captured.out, captured.err


def assert_public_error(status, out, err, code):
    assert status == 1
    assert out == ""
    detail = json.loads(err)
    assert detail["code"] == code
    assert "Traceback" not in err
    assert "PRIVATE_SENTINEL_8712" not in err
    assert "input_value" not in err
    return detail


def no_credentials(*args, **kwargs):
    pytest.fail("This offline command must not load credentials or create a live provider.")


@pytest.fixture
def small_experiment(tmp_path, monkeypatch, capsys):
    status, out, _ = invoke(
        monkeypatch,
        capsys,
        "bench-run",
        "--output-dir",
        tmp_path / "experiments",
        "--limit",
        2,
        "--presets",
        "baseline",
        "tight-budget",
    )
    assert status == 0
    result = json.loads(out)
    return Path(result["results_path"])


def hotpot_rows():
    return [
        {
            "_id": "hp-first",
            "question": "How long is the river in Town A?",
            "answer": "ten",
            "supporting_facts": [["Town A", 0], ["River A", 1]],
            "context": [
                ["Town A", ["Town A is home to River A.", "This is local test data."]],
                ["River A", ["River A runs through Town A.", "Its length is ten."]],
            ],
            "type": "bridge",
            "level": "easy",
        },
        {
            "_id": "hp-second",
            "question": "How long is the river in Town B?",
            "answer": "twenty",
            "supporting_facts": [["Town B", 0], ["River B", 0]],
            "context": [
                ["Town B", ["Town B is home to River B."]],
                ["River B", ["River B has length twenty."]],
            ],
            "type": "bridge",
            "level": "medium",
        },
    ]


@pytest.fixture
def imported_hotpot(tmp_path, monkeypatch, capsys):
    raw = tmp_path / "synthetic-hotpot.json"
    raw.write_text(json.dumps(hotpot_rows(), indent=2), encoding="utf-8")
    destination = tmp_path / "imported.json"
    status, out, _ = invoke(
        monkeypatch,
        capsys,
        "bench-import-hotpot",
        raw,
        "--output",
        destination,
        "--limit",
        2,
    )
    assert status == 0
    assert json.loads(out)["downloaded"] is False
    return raw, destination


def predictions(rows=None):
    rows = hotpot_rows() if rows is None else rows
    return {
        "answer": {row["_id"]: row["answer"] for row in rows},
        "sp": {row["_id"]: row["supporting_facts"] for row in rows},
    }


def test_bench_info_builtin_has_expected_size_and_no_model_use(monkeypatch, capsys):
    monkeypatch.setattr(ModelProfile, "load_settings", no_credentials)
    monkeypatch.setattr(runner, "ChatCompatibleProvider", no_credentials)
    status, out, err = invoke(monkeypatch, capsys, "bench-info")
    assert status == 0 and err == ""
    result = json.loads(out)
    assert result["case_count"] == 35
    assert len(result["domains"]) == 11
    assert result["presets"] == runner.PRESETS
    assert result["fingerprint"] == dataset_fingerprint(load_benchmark())
    assert result["model_calls"] == 0
    assert result["human_calibrated"] is False


def test_help_lists_all_six_m8_commands_and_keeps_legacy_commands(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["research", "--help"])
    with pytest.raises(SystemExit) as caught:
        cli.main()
    assert caught.value.code == 0
    output = capsys.readouterr().out
    for name in (
        "bench-info",
        "bench-run",
        "bench-compare",
        "bench-import-hotpot",
        "bench-score-hotpot",
        "bench-judge",
        "smoke",
        "run-dag",
        "demo-budget",
        "serve",
    ):
        assert name in output


@pytest.mark.parametrize(
    "command",
    [
        "bench-info",
        "bench-run",
        "bench-compare",
        "bench-import-hotpot",
        "bench-score-hotpot",
        "bench-judge",
    ],
)
def test_each_command_help_does_not_read_credentials(command, monkeypatch, capsys):
    monkeypatch.setattr(ModelProfile, "load_settings", no_credentials)
    monkeypatch.setattr(sys, "argv", ["research", command, "--help"])
    with pytest.raises(SystemExit) as caught:
        cli.main()
    assert caught.value.code == 0
    assert command in capsys.readouterr().out


def test_info_malformed_dataset_has_safe_public_error(tmp_path, monkeypatch, capsys):
    path = tmp_path / "malformed.json"
    path.write_text('{"PRIVATE_SENTINEL_8712":"bad"}', encoding="utf-8")
    assert_public_error(
        *invoke(monkeypatch, capsys, "bench-info", "--dataset", path), "benchmark_data"
    )


def test_true_offline_cli_run_freezes_input_and_exposes_budget_without_credentials(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.setattr(ModelProfile, "load_settings", no_credentials)
    monkeypatch.setattr(runner, "ChatCompatibleProvider", no_credentials)
    data = load_benchmark().model_dump(mode="json")
    data["cases"] = data["cases"][:1]
    supplied = tmp_path / "input.json"
    supplied.write_text(json.dumps(data, ensure_ascii=False, indent=3), encoding="utf-8")
    env = tmp_path / "never-read.env"
    env.write_text("OPENAI_API_KEY=PRIVATE_SENTINEL_8712", encoding="utf-8")
    status, out, err = invoke(
        monkeypatch,
        capsys,
        "bench-run",
        "--dataset",
        supplied,
        "--output-dir",
        tmp_path / "outputs",
        "--limit",
        1,
        "--env-file",
        env,
    )
    assert status == 0
    assert "PRIVATE_SENTINEL_8712" not in out + err
    result = json.loads(out)
    assert result["mode"] == "offline_fixture"
    assert result["planned_call_ceiling"] == 3
    assert result["quality_claim"] is False
    saved = Path(result["results_path"])
    artifact = runner.load_verified_results(saved)
    assert artifact["manifest"]["dataset_fingerprint"] == dataset_fingerprint(
        load_benchmark(supplied)
    )
    assert artifact["manifest"]["memory"] == "disabled"
    assert artifact["manifest"]["retrieval"] == "frozen_local_lexical"
    assert (saved.parent / "dataset.json").read_bytes() == runner.canonical(data)
    row = artifact["records"][0]
    assert row["status"] == "completed"
    assert row["model_budget"]["reserved"]["calls"] == 2
    assert row["model_budget"]["usage_unknown_calls"] == 2
    assert Path(result["report_path"]).is_file()


@pytest.mark.parametrize(
    "flags,code",
    [
        (["--limit", "0"], "benchmark_configuration"),
        (["--limit", "51"], "benchmark_configuration"),
        (["--repeats", "0"], "benchmark_configuration"),
        (["--repeats", "4"], "benchmark_configuration"),
        (["--seed", "-1"], "benchmark_configuration"),
        (["--max-total-model-calls", "0"], "benchmark_configuration"),
        (["--max-total-model-calls", "1"], "benchmark_budget"),
        (["--allow-live"], "benchmark_configuration"),
        (["--profiles", "PRIVATE_SENTINEL_8712.toml"], "benchmark_configuration"),
        (["--profile", "PRIVATE_SENTINEL_8712"], "benchmark_configuration"),
        (["--presets", "baseline", "baseline"], "benchmark_configuration"),
    ],
)
def test_run_preflight_rejects_before_credentials_provider_or_output(
    tmp_path, monkeypatch, capsys, flags, code
):
    monkeypatch.setattr(ModelProfile, "load_settings", no_credentials)
    monkeypatch.setattr(runner, "BenchmarkFixtureProvider", no_credentials)
    monkeypatch.setattr(runner, "ChatCompatibleProvider", no_credentials)
    output = tmp_path / "outputs"
    response = invoke(monkeypatch, capsys, "bench-run", "--output-dir", output, *flags)
    assert_public_error(*response, code)
    assert not output.exists()


def test_run_command_forwards_all_arguments_without_implicit_live(monkeypatch, capsys, tmp_path):
    seen = []

    async def fake_run(**kwargs):
        seen.append(kwargs)
        return {"status": "fixture"}

    monkeypatch.setattr(commands, "run_benchmark", fake_run)
    status, out, _ = invoke(
        monkeypatch,
        capsys,
        "bench-run",
        "--dataset",
        tmp_path / "data.json",
        "--output-dir",
        tmp_path / "out",
        "--presets",
        "review",
        "serial",
        "--limit",
        7,
        "--repeats",
        2,
        "--seed",
        123,
        "--max-total-model-calls",
        180,
    )
    assert status == 0 and json.loads(out) == {"status": "fixture"}
    assert len(seen) == 1
    assert seen[0] == {
        "dataset_path": tmp_path / "data.json",
        "output_dir": tmp_path / "out",
        "presets": ("review", "serial"),
        "limit": 7,
        "repeats": 2,
        "seed": 123,
        "max_total_model_calls": 180,
        "allow_live": False,
        "profiles_file": None,
        "profile_id": None,
        "env_file": Path(".env"),
        "on_progress": cli._progress,
    }


def test_compare_cli_prints_and_saves_same_result(small_experiment, tmp_path, monkeypatch, capsys):
    destination = tmp_path / "comparison.json"
    status, out, err = invoke(
        monkeypatch,
        capsys,
        "bench-compare",
        small_experiment,
        "--candidate",
        "tight-budget",
        "--resamples",
        100,
        "--seed",
        9,
        "--output",
        destination,
    )
    assert status == 0 and err == ""
    result = json.loads(out)
    assert json.loads(destination.read_text(encoding="utf-8")) == result
    assert result["pairs_used"] == result["pairs_total"] == 2
    assert result["mean_delta"] == -1
    assert result["ci95"] == [-1, -1]
    assert result["cohen_dz"] is None
    assert result["quality_claim"] is False


def test_compare_cli_will_not_overwrite_output(small_experiment, tmp_path, monkeypatch, capsys):
    destination = tmp_path / "comparison.json"
    destination.write_bytes(b"KEEP ORIGINAL")
    response = invoke(
        monkeypatch,
        capsys,
        "bench-compare",
        small_experiment,
        "--candidate",
        "tight-budget",
        "--output",
        destination,
    )
    assert_public_error(*response, "benchmark_output")
    assert destination.read_bytes() == b"KEEP ORIGINAL"


def test_compare_unknown_artifact_has_safe_error(tmp_path, monkeypatch, capsys):
    path = tmp_path / "invalid.json"
    path.write_text('{"PRIVATE_SENTINEL_8712":"bad"}', encoding="utf-8")
    assert_public_error(*invoke(monkeypatch, capsys, "bench-compare", path), "benchmark_artifact")


def test_import_preserves_original_sentence_coordinates_and_raw_hash(imported_hotpot):
    source, imported = imported_hotpot
    dataset = load_benchmark(imported)
    assert dataset.source_sha256 == hashlib.sha256(source.read_bytes()).hexdigest()
    assert dataset.split == "local-unverified-subset"
    assert len(dataset.cases) == 2
    assert dataset.cases[0].documents[0].sentences == tuple(hotpot_rows()[0]["context"][0][1])
    assert [
        (item.title, item.sentence_index) for item in dataset.cases[0].gold.supporting_facts
    ] == [
        ("Town A", 0),
        ("River A", 1),
    ]


def test_import_limit_selects_rows_but_hashes_entire_raw_file(tmp_path, monkeypatch, capsys):
    raw = json.dumps(hotpot_rows(), indent=4).encode()
    source = tmp_path / "source.json"
    source.write_bytes(raw)
    destination = tmp_path / "selected.json"
    status, out, _ = invoke(
        monkeypatch, capsys, "bench-import-hotpot", source, "--limit", 1, "--output", destination
    )
    assert status == 0 and json.loads(out)["count"] == 1
    dataset = load_benchmark(destination)
    assert len(dataset.cases) == 1
    assert dataset.source_sha256 == hashlib.sha256(raw).hexdigest()
    assert dataset.cases[0].id == "hp-first"


def test_import_will_not_overwrite_destination(imported_hotpot, monkeypatch, capsys):
    source, destination = imported_hotpot
    original = destination.read_bytes()
    assert_public_error(
        *invoke(monkeypatch, capsys, "bench-import-hotpot", source, "--output", destination),
        "benchmark_output",
    )
    assert destination.read_bytes() == original


@pytest.mark.parametrize("complete", [True, False])
def test_hotpot_score_all_gold_denominator(
    imported_hotpot, tmp_path, monkeypatch, capsys, complete
):
    source, dataset = imported_hotpot
    prediction_path = tmp_path / "predictions.json"
    prediction_path.write_text(
        json.dumps(predictions(None if complete else hotpot_rows()[:1])), encoding="utf-8"
    )
    output = tmp_path / "score.json"
    status, out, err = invoke(
        monkeypatch,
        capsys,
        "bench-score-hotpot",
        "--dataset",
        dataset,
        "--predictions",
        prediction_path,
        "--output",
        output,
    )
    assert status == 0 and err == ""
    result = json.loads(out)
    assert json.loads(output.read_text(encoding="utf-8")) == result
    assert result["count"] == 2
    assert all(value == (1.0 if complete else 0.5) for value in result["metrics"].values())
    assert result["joint_coverage"] == (1.0 if complete else 0.5)
    assert result["missing_answers"] == ([] if complete else ["hp-second"])
    assert result["missing_supporting_facts"] == ([] if complete else ["hp-second"])
    assert result["official_submission"] is False
    assert result["source_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()


@pytest.mark.parametrize("field", ["answer", "sp"])
def test_hotpot_unknown_prediction_id_rejected(
    imported_hotpot, tmp_path, monkeypatch, capsys, field
):
    _, dataset = imported_hotpot
    prediction = predictions()
    prediction[field]["PRIVATE_SENTINEL_8712"] = "answer" if field == "answer" else []
    path = tmp_path / "predictions.json"
    path.write_text(json.dumps(prediction), encoding="utf-8")
    assert_public_error(
        *invoke(
            monkeypatch, capsys, "bench-score-hotpot", "--dataset", dataset, "--predictions", path
        ),
        "benchmark_data",
    )


def test_hotpot_scorer_rejects_researchbench_dataset(tmp_path, monkeypatch, capsys):
    dataset = tmp_path / "researchbench.json"
    dataset.write_text(load_benchmark().model_dump_json(), encoding="utf-8")
    missing = tmp_path / "PRIVATE_SENTINEL_8712.json"
    assert_public_error(
        *invoke(
            monkeypatch,
            capsys,
            "bench-score-hotpot",
            "--dataset",
            dataset,
            "--predictions",
            missing,
        ),
        "benchmark_data",
    )


def test_hotpot_scorer_will_not_overwrite_output(imported_hotpot, tmp_path, monkeypatch, capsys):
    _, dataset = imported_hotpot
    path = tmp_path / "predictions.json"
    path.write_text(json.dumps(predictions()), encoding="utf-8")
    output = tmp_path / "scores.json"
    output.write_bytes(b"KEEP ORIGINAL")
    assert_public_error(
        *invoke(
            monkeypatch,
            capsys,
            "bench-score-hotpot",
            "--dataset",
            dataset,
            "--predictions",
            path,
            "--output",
            output,
        ),
        "benchmark_output",
    )
    assert output.read_bytes() == b"KEEP ORIGINAL"


def test_judge_cli_default_is_isolated_offline_and_preserves_candidate(
    small_experiment, monkeypatch, capsys
):
    monkeypatch.setattr(ModelProfile, "load_settings", no_credentials)
    monkeypatch.setattr(commands, "ChatCompatibleProvider", no_credentials)
    artifact = runner.load_verified_results(small_experiment)
    row = next(row for row in artifact["records"] if row["preset"] == "baseline")
    original = small_experiment.read_bytes()
    status, out, err = invoke(
        monkeypatch, capsys, "bench-judge", small_experiment, "--case-id", row["case_id"]
    )
    assert status == 0 and err == ""
    result = json.loads(out)
    assert result["status"] == "completed"
    assert result["method"] == "offline_fixture"
    assert result["candidate_run_id"] == row["run_id"]
    assert result["candidate_sha256"] == row["bundle_sha256"]
    assert result["judge_run_id"] != row["run_id"]
    assert result["model_budget"]["reserved"]["calls"] == 1
    assert result["calibrated"] is False and result["quality_claim"] is False
    assert Path(result["result_path"]).is_file()
    assert small_experiment.read_bytes() == original


@pytest.mark.parametrize(
    "flags",
    [
        ["--allow-live"],
        ["--profiles", "PRIVATE_SENTINEL_8712.toml"],
        ["--profile", "PRIVATE_SENTINEL_8712"],
        ["--repeat", "-1"],
        ["--repeat", "3"],
    ],
)
def test_judge_gate_precedes_artifact_and_credentials(tmp_path, monkeypatch, capsys, flags):
    monkeypatch.setattr(ModelProfile, "load_settings", no_credentials)
    response = invoke(
        monkeypatch,
        capsys,
        "bench-judge",
        tmp_path / "missing.json",
        "--case-id",
        "rb-001",
        *flags,
    )
    assert_public_error(*response, "benchmark_configuration")


def test_cli_error_wrapper_retains_run_id_without_raw_payload(monkeypatch, capsys):
    from deep_research.research import ResearchRunError

    async def broken(_results, **kwargs):
        raise ResearchRunError(ResearchError("judge_binding", "安全错误消息。"), "fixture-run")

    monkeypatch.setattr(commands, "judge_saved", broken)
    status, out, err = invoke(
        monkeypatch, capsys, "bench-judge", "unused.json", "--case-id", "rb-001"
    )
    detail = assert_public_error(status, out, err, "judge_binding")
    assert detail["run_id"] == "fixture-run"
    assert detail["status_saved"] is True


def test_cli_keyboard_interrupt_returns_130(monkeypatch, capsys):
    async def interrupted(**kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(commands, "run_benchmark", interrupted)
    status, out, err = invoke(monkeypatch, capsys, "bench-run")
    assert status == 130 and out == ""
    assert "取消" in err
