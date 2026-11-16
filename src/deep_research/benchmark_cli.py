"""Explicit M8 commands. No bulk paid work unless --allow-live is supplied."""

import argparse
import asyncio
from contextlib import AsyncExitStack
from pathlib import Path
from uuid import uuid4

from deep_research.benchmark_data import dataset_fingerprint, import_hotpot, load_benchmark
from deep_research.benchmark_metrics import evaluate_hotpot
from deep_research.benchmark_runner import (
    METRICS,
    PRESETS,
    _offline_profile,
    canonical,
    compare_results,
    fingerprint,
    load_verified_results,
    read_json,
    run_benchmark,
    save_json,
)
from deep_research.errors import ResearchError
from deep_research.gateway import ModelGateway
from deep_research.model_budget import ModelBudgetLimits
from deep_research.model_profiles import load_profiles
from deep_research.providers.chat_compatible import ChatCompatibleProvider


def register_benchmark_commands(subparsers) -> None:
    info = subparsers.add_parser("bench-info", help="查看冻结题库及六种实验配置，不调用模型")
    info.add_argument("--dataset", type=Path)
    run = subparsers.add_parser("bench-run", help="运行冻结语料评测；默认假模型，不消耗额度")
    run.add_argument("--dataset", type=Path)
    run.add_argument("--presets", nargs="+", choices=tuple(PRESETS), default=["baseline"])
    run.add_argument("--limit", type=int, default=3)
    run.add_argument("--repeats", type=int, default=1)
    run.add_argument("--seed", type=int, default=42)
    run.add_argument("--max-total-model-calls", type=int, default=90)
    run.add_argument("--output-dir", type=Path, default=Path("outputs/benchmarks"))
    run.add_argument("--allow-live", action="store_true", help="允许冻结资料发给模型并批量计费")
    run.add_argument("--profiles", type=Path)
    run.add_argument("--profile")
    run.add_argument("--env-file", type=Path, default=Path(".env"))
    compare = subparsers.add_parser("bench-compare", help="逐题配对Bootstrap区间和Cohen dz")
    compare.add_argument("results", type=Path)
    compare.add_argument("--baseline", choices=tuple(PRESETS), default="baseline")
    compare.add_argument("--candidate", choices=tuple(PRESETS), default="full")
    compare.add_argument("--metric", choices=METRICS, default="citation_coverage")
    compare.add_argument("--seed", type=int, default=42)
    compare.add_argument("--resamples", type=int, default=2000)
    compare.add_argument("--output", type=Path)
    hotpot = subparsers.add_parser("bench-import-hotpot", help="导入本地HotpotQA样本，不自动下载")
    hotpot.add_argument("input", type=Path)
    hotpot.add_argument("--limit", type=int, default=35)
    hotpot.add_argument("--output", type=Path, required=True)
    score = subparsers.add_parser("bench-score-hotpot", help="按短答案和支持句评分，不冒充报告评分")
    score.add_argument("--dataset", type=Path, required=True)
    score.add_argument("--predictions", type=Path, required=True)
    score.add_argument("--output", type=Path)
    judge = subparsers.add_parser("bench-judge", help="独立五维裁判；默认固定替身，真实评判须授权")
    judge.add_argument("results", type=Path)
    judge.add_argument("--case-id", required=True)
    judge.add_argument("--preset", choices=tuple(PRESETS), default="baseline")
    judge.add_argument("--repeat", type=int, default=0)
    judge.add_argument("--allow-live", action="store_true")
    judge.add_argument("--profiles", type=Path)
    judge.add_argument("--profile")
    judge.add_argument("--env-file", type=Path, default=Path(".env"))


def _write_new(path: Path, value: object) -> None:
    raw = canonical(value)
    try:
        with path.open("xb") as stream:
            stream.write(raw)
    except OSError:
        raise ResearchError("benchmark_output", "无法新建输出文件；不会覆盖已有文件。") from None


def _resolved_parent(path: Path) -> Path:
    return path.resolve().parent


async def judge_saved(
    results_path: Path,
    *,
    case_id: str,
    preset: str = "baseline",
    repeat: int = 0,
    allow_live: bool = False,
    profiles_file: Path | None = None,
    profile_id: str | None = None,
    env_file: Path = Path(".env"),
) -> dict:
    from deep_research.benchmark_judge import FakeJudgeProvider, judge_report

    if (
        type(allow_live) is not bool
        or type(repeat) is not int
        or not 0 <= repeat <= 2
        or (not allow_live and (profiles_file or profile_id))
        or (allow_live and profiles_file is None)
    ):
        raise ResearchError("benchmark_configuration", "裁判模式或配置无效；真实评判必须显式允许。")
    artifact = load_verified_results(results_path)
    folder = _resolved_parent(results_path)
    try:
        dataset = load_benchmark(folder / "dataset.json")
        if dataset_fingerprint(dataset) != artifact["manifest"]["dataset_fingerprint"]:
            raise ValueError
        rows = [
            row
            for row in artifact["records"]
            if (row["case_id"], row["preset"], row["repeat"]) == (case_id, preset, repeat)
        ]
        if len(rows) != 1 or rows[0]["status"] != "completed":
            raise ValueError
        row = rows[0]
        bundle_path = (folder / row["bundle_path"]).resolve()
        if not bundle_path.is_relative_to(folder / "reports") or bundle_path.suffix != ".json":
            raise ValueError
        bundle = read_json(bundle_path)
        case = next(case for case in dataset.cases if case.id == case_id)
        if (
            fingerprint(bundle) != row["bundle_sha256"]
            or bundle["question"] != case.question
            or bundle["run_id"] != row["run_id"]
        ):
            raise ValueError
    except (KeyError, TypeError, ValueError, StopIteration):
        raise ResearchError(
            "benchmark_artifact", "待评报告、题库或绑定指纹不匹配，拒绝评判。"
        ) from None
    profile = load_profiles(profiles_file).get(profile_id) if allow_live else None
    settings = profile.load_settings(env_file) if profile else None
    identity = profile.public_snapshot() if profile else _offline_profile()
    if profile is None:
        identity.update(
            id="benchmark-judge-fixture", label="M8离线裁判替身", model="scripted-judge"
        )
        identity["fingerprint"] = fingerprint(
            {key: value for key, value in identity.items() if key != "fingerprint"}
        )
    limits = profile.budget if profile else ModelBudgetLimits()
    budget = ModelBudgetLimits(**{**limits.model_dump(), "max_calls": 1})
    destination = folder / "judgements" / str(uuid4())
    destination.mkdir(parents=True, exist_ok=False)
    linkage = {
        "status": "running",
        "case_id": case_id,
        "preset": preset,
        "repeat": repeat,
        "candidate_run_id": row["run_id"],
        "candidate_sha256": row["bundle_sha256"],
        "dataset_fingerprint": artifact["manifest"]["dataset_fingerprint"],
        "candidate_mode": artifact["manifest"]["mode"],
        "judge_profile": identity,
        "quality_claim": False,
    }
    save_json(destination / "result.json", linkage)
    try:
        async with AsyncExitStack() as stack:
            provider = (
                await stack.enter_async_context(ChatCompatibleProvider(settings))
                if settings
                else FakeJudgeProvider()
            )
            result = await judge_report(
                question=case.question,
                bundle=bundle,
                gold=case.gold.model_dump(mode="json"),
                gateway=ModelGateway(provider),
                db=destination / "judge.sqlite3",
                profile=identity,
                budget=budget,
                offline=not allow_live,
            )
        linkage.update(result)
    except asyncio.CancelledError as error:
        linkage["status"] = "cancelled"
        linkage["judge_run_id"] = getattr(error, "run_id", None)
        try:
            save_json(destination / "result.json", linkage)
        except Exception:
            error.add_note("裁判取消记录保存失败；保留原始取消。")
        raise
    except BaseException as error:
        linkage.update(
            status="failed",
            error_code=getattr(error, "code", "unexpected"),
            judge_run_id=getattr(error, "run_id", None),
        )
        try:
            save_json(destination / "result.json", linkage)
        except Exception:
            error.add_note("裁判失败记录保存失败；保留原始异常。")
        raise
    save_json(destination / "result.json", linkage)
    return {**linkage, "result_path": str(destination / "result.json")}


def benchmark_command(args: argparse.Namespace, *, on_progress=None) -> dict:
    if args.command == "bench-info":
        dataset = load_benchmark(args.dataset)
        return {
            "id": dataset.id,
            "fingerprint": dataset_fingerprint(dataset),
            "provenance": dataset.provenance,
            "description": dataset.description,
            "case_count": len(dataset.cases),
            "domains": sorted({c.domain for c in dataset.cases}),
            "presets": PRESETS,
            "model_calls": 0,
            "human_calibrated": False,
        }
    if args.command == "bench-run":
        return asyncio.run(
            run_benchmark(
                output_dir=args.output_dir,
                dataset_path=args.dataset,
                presets=tuple(args.presets),
                limit=args.limit,
                repeats=args.repeats,
                seed=args.seed,
                max_total_model_calls=args.max_total_model_calls,
                allow_live=args.allow_live,
                profiles_file=args.profiles,
                profile_id=args.profile,
                env_file=args.env_file,
                on_progress=on_progress,
            )
        )
    if args.command == "bench-compare":
        result = compare_results(
            args.results,
            baseline=args.baseline,
            candidate=args.candidate,
            metric=args.metric,
            seed=args.seed,
            n_resamples=args.resamples,
        )
        if args.output:
            _write_new(args.output, result)
        return result
    if args.command == "bench-import-hotpot":
        dataset = import_hotpot(args.input, limit=args.limit)
        _write_new(args.output, dataset.model_dump(mode="json"))
        return {
            "status": "imported",
            "count": len(dataset.cases),
            "output": str(args.output),
            "fingerprint": dataset_fingerprint(dataset),
            "downloaded": False,
        }
    if args.command == "bench-score-hotpot":
        dataset = load_benchmark(args.dataset)
        if any(case.kind != "hotpotqa" for case in dataset.cases):
            raise ResearchError("benchmark_data", "此评分入口只接受HotpotQA导入题库。")
        gold = [
            {
                "_id": case.id,
                "answer": case.gold.answer_aliases[0],
                "supporting_facts": [
                    [s.title, s.sentence_index] for s in case.gold.supporting_facts
                ],
            }
            for case in dataset.cases
        ]
        try:
            result = evaluate_hotpot(gold, read_json(args.predictions))
        except (TypeError, ValueError):
            raise ResearchError("benchmark_data", "HotpotQA预测格式无效，拒绝缩小分母。") from None
        result.update(
            dataset_fingerprint=dataset_fingerprint(dataset),
            source_sha256=dataset.source_sha256,
            official_submission=False,
        )
        if args.output:
            _write_new(args.output, result)
        return result
    if args.command == "bench-judge":
        return asyncio.run(
            judge_saved(
                args.results,
                case_id=args.case_id,
                preset=args.preset,
                repeat=args.repeat,
                allow_live=args.allow_live,
                profiles_file=args.profiles,
                profile_id=args.profile,
                env_file=args.env_file,
            )
        )
    raise ResearchError("benchmark_configuration", "未知评测命令。")
