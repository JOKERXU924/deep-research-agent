"""M8 frozen-corpus experiments. Fake runs test machinery, not model quality."""

import asyncio
import hashlib
import json
import platform
import random
from collections import defaultdict
from contextlib import AsyncExitStack
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from deep_research.benchmark_data import dataset_fingerprint, load_benchmark
from deep_research.benchmark_metrics import score_report
from deep_research.benchmark_statistics import compare_paired
from deep_research.budget_demo import offline_profile
from deep_research.compression import CompressionOptions, HierarchicalCompressor
from deep_research.dag_journal import DagJournal
from deep_research.dag_research import DagOptions, run_dag_pipeline
from deep_research.embeddings import build_encoder
from deep_research.errors import ResearchError
from deep_research.gateway import ModelGateway
from deep_research.model_budget import ModelBudgetLimits
from deep_research.model_profiles import load_profiles
from deep_research.providers.chat_compatible import ChatCompatibleProvider
from deep_research.research import ResearchRunError
from deep_research.research_store import ResearchStore
from deep_research.retrieval import LocalCorpusRetriever, SourceDocument
from deep_research.review_journal import ReviewJournal
from deep_research.review_models import ReviewOptions
from deep_research.schemas import ModelRequest, ModelResponse

PRESETS = {
    "baseline": {"compression": False, "review": False, "concurrency": 3, "max_calls": 3},
    "compression": {"compression": True, "review": False, "concurrency": 3, "max_calls": 3},
    "review": {"compression": False, "review": True, "concurrency": 3, "max_calls": 9},
    "full": {"compression": True, "review": True, "concurrency": 3, "max_calls": 9},
    "tight-budget": {"compression": False, "review": False, "concurrency": 3, "max_calls": 1},
    "serial": {"compression": False, "review": False, "concurrency": 1, "max_calls": 3},
}
METRICS = ("citation_coverage", "citation_validity", "gold_fact_coverage")
MAX_ARTIFACT_BYTES = 16 * 1024 * 1024


def canonical(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def fingerprint(value: object) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def read_json(path: Path, *, max_bytes: int = MAX_ARTIFACT_BYTES):
    """A local data reader, never a URL loader or executable deserializer."""

    def unique(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError("duplicate key")
            value[key] = item
        return value

    def invalid(_):
        raise ValueError("non-finite")

    try:
        with path.open("rb") as stream:
            raw = stream.read(max_bytes + 1)
        if len(raw) > max_bytes:
            raise ValueError("too large")
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=unique, parse_constant=invalid)
        canonical(value)  # also reject floating-point overflow and invalid Unicode
        return value
    except (OSError, ValueError, UnicodeError, RecursionError, TypeError):
        raise ResearchError("benchmark_artifact", "评测文件不存在、过大或格式不合法。") from None


def save_json(path: Path, value: object) -> None:
    """Atomic replacement within a newly allocated experiment, not crash transactions."""
    raw = canonical(value)
    if len(raw) > MAX_ARTIFACT_BYTES:
        raise ResearchError("benchmark_artifact", "评测结果超过文件上限，停止继续执行。")
    temporary = path.with_name(path.name + ".tmp-" + uuid4().hex)
    try:
        with temporary.open("xb") as stream:
            stream.write(raw)
            stream.flush()
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _engine_fingerprint() -> str:
    root = Path(__file__).parent
    hasher = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        hasher.update(path.relative_to(root).as_posix().encode())
        hasher.update(path.read_bytes())
    return hasher.hexdigest()


def _new_experiment_folder(output_dir: Path, experiment_id: str) -> Path:
    folder = (output_dir / experiment_id).resolve()
    folder.mkdir(parents=True, exist_ok=False)
    return folder


class BenchmarkFixtureProvider:
    """Only reads the same request as the real provider. Never receives gold."""

    async def complete(self, request: ModelRequest) -> ModelResponse:
        context = json.loads(request.messages[-1].content)
        if "claim_inventory" in context:
            data = {
                "report_hash": context["report_hash"],
                "scores": dict.fromkeys(
                    ("evidence_support", "logical_consistency", "citation_quality"), 90
                ),
                "findings": [],
            }
        elif "evidence" in context:
            # An extractive fixture, not a research-quality oracle. At most one quote/source.
            selected = {}
            for item in context["evidence"]:
                selected.setdefault(item["source_id"], item)
            data = {
                "title": "Offline extractive benchmark fixture",
                "sections": [
                    {
                        "heading": "Observed excerpts",
                        "claims": [
                            {"text": item["quote"][:650], "citations": [item["evidence_id"]]}
                            for item in list(selected.values())[:6]
                        ],
                    }
                ],
                "limitations": [
                    "Scripted fixture; no inference or genuine model quality measured."
                ],
            }
        else:
            # M2's planning input is a JSON object, and only the question is relevant.
            data = {"queries": [context["question"][:120]]}
        return ModelResponse(
            provider="offline_fixture",
            model="benchmark-extractive-v1",
            request_id=uuid4().hex,
            content=json.dumps(data, ensure_ascii=False),
            latency_ms=0.0,
        )


def _offline_profile() -> dict:
    profile = offline_profile()
    profile.update(id="benchmark-fixture", label="M8离线抽取替身", model="benchmark-extractive-v1")
    profile["fingerprint"] = fingerprint({k: v for k, v in profile.items() if k != "fingerprint"})
    return profile


def _plan(dataset, *, presets, limit, repeats, seed, max_total_model_calls, limits):
    if (
        type(limit) is not int
        or not 1 <= limit <= 50
        or type(repeats) is not int
        or not 1 <= repeats <= 3
        or type(seed) is not int
        or not 0 <= seed <= 2**32 - 1
        or type(max_total_model_calls) is not int
        or not 1 <= max_total_model_calls <= 2700
        or not isinstance(presets, (list, tuple))
        or not presets
        or len(presets) > 6
        or any(not isinstance(p, str) or p not in PRESETS for p in presets)
        or len(set(presets)) != len(presets)
    ):
        raise ResearchError("benchmark_configuration", "评测场景、样本、重复次数或总调用上限无效。")
    cases = sorted(dataset.cases, key=lambda item: item.id)
    random.Random(seed).shuffle(cases)
    cases = cases[:limit]
    configs = {}
    for name in presets:
        config = dict(PRESETS[name])
        config["max_calls"] = min(config["max_calls"], limits.max_calls)
        configs[name] = config
    ceiling = len(cases) * repeats * sum(c["max_calls"] for c in configs.values())
    if len(cases) * len(presets) * repeats > 300 or ceiling > max_total_model_calls:
        raise ResearchError(
            "benchmark_budget", "计划最大调用数或运行数超过本次显式上限；未启动模型。"
        )
    tasks = [(case, preset, rep) for case in cases for preset in presets for rep in range(repeats)]
    random.Random(seed).shuffle(tasks)
    return cases, configs, tasks, ceiling


def summarize(records: list[dict], manifest: dict) -> dict:
    """Use every attempted sample; show null separately, never drop failure denominators."""
    output = {}
    for preset in manifest["presets"]:
        rows = [row for row in records if row["preset"] == preset]
        scores = {}
        for metric in METRICS:
            per_case = defaultdict(list)
            for row in rows:
                value = row["metrics"].get(metric)
                if value is not None:
                    per_case[row["case_id"]].append(value)
            means = [sum(v) / len(v) for v in per_case.values()]
            scores[metric] = {
                "mean": sum(means) / len(means) if means else None,
                "scored_cases": len(means),
                "attempted_cases": len({r["case_id"] for r in rows}),
            }
        known = unknown = missing = 0
        for row in rows:
            budget = row["model_budget"]
            if budget is None:
                missing += 1
            else:
                known += budget["known_usage"]["total_tokens"]
                unknown += budget["usage_unknown_calls"]
        output[preset] = {
            "attempted": len(rows),
            "planned": manifest["case_count"] * manifest["repeats"],
            "completed": sum(r["status"] == "completed" for r in rows),
            "failed": sum(r["status"] == "failed" for r in rows),
            "cancelled": sum(r["status"] == "cancelled" for r in rows),
            "degraded": sum(r.get("degraded", False) for r in rows),
            "metrics": scores,
            "known_tokens": known,
            "unknown_usage_calls": unknown,
            "unavailable_budget_rows": missing,
        }
    return output


def render_summary(artifact: dict) -> str:
    manifest = artifact["manifest"]
    lines = [
        "# M8 评测运行记录",
        "",
        f"实验 ID：{manifest['experiment_id']}",
        f"模式：{manifest['mode']}；状态：{manifest['status']}",
        f"题库指纹：`{manifest['dataset_fingerprint']}`",
        "",
        "离线替身分数只用于验证评测流程，不能用于宣称模型效果。",
        "规则分数只检查引用结构与关键词覆盖，不等于事实准确率或幻觉率。",
        "",
        "| 场景 | 已尝试/计划 | 完成 | 失败 | 降级 | 引用覆盖 | 要点字面覆盖 | 未知用量调用 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for name, row in artifact["summary"].items():

        def fmt(key, row=row):
            value = row["metrics"][key]["mean"]
            return "未定义" if value is None else f"{value:.3f}"

        lines.append(
            f"| {name} | {row['attempted']}/{row['planned']} | {row['completed']} | "
            f"{row['failed']} | {row['degraded']} | {fmt('citation_coverage')} | "
            f"{fmt('gold_fact_coverage')} | {row['unknown_usage_calls']} |"
        )
    lines += [
        "",
        "失败样本按零计入上述效用指标；无适用标准答案的指标保持未定义。",
        "重复运行先按题目求均值，再计算题目宏平均；未执行样本不伪装成已完成。",
        "known_tokens 是已知小计；有未知用量或账本缺失时不能作为总费用。",
        "冻结语料检索不是全网研究；所有实验禁用跨题共享记忆，防止答案污染。",
        "种子只控制选题、执行顺序及统计重采样，不保证远端模型确定性。",
        "本轮不自动续跑、不自动重发；原始报告、调用账本和逐题结果另行保存。",
        "",
    ]
    return "\n".join(lines)


async def run_benchmark(
    *,
    output_dir: Path,
    dataset_path: Path | None = None,
    presets: tuple[str, ...] = ("baseline",),
    limit: int = 3,
    repeats: int = 1,
    seed: int = 42,
    max_total_model_calls: int = 90,
    allow_live: bool = False,
    profiles_file: Path | None = None,
    profile_id: str | None = None,
    env_file: Path = Path(".env"),
    on_progress=None,
) -> dict:
    dataset = load_benchmark(dataset_path)
    if type(allow_live) is not bool or (not allow_live and (profiles_file or profile_id)):
        raise ResearchError(
            "benchmark_configuration", "真实评测必须显式允许；离线模式不读取模型配置。"
        )
    if allow_live and profiles_file is None:
        raise ResearchError("benchmark_configuration", "真实评测需要显式指定服务端模型目录。")
    profile = load_profiles(profiles_file).get(profile_id) if allow_live else None
    limits = profile.budget if profile else ModelBudgetLimits()
    cases, configs, tasks, ceiling = _plan(
        dataset,
        presets=presets,
        limit=limit,
        repeats=repeats,
        seed=seed,
        max_total_model_calls=max_total_model_calls,
        limits=limits,
    )
    # All workload and configuration checks precede credential access and output creation.
    settings = profile.load_settings(env_file) if profile else None
    public_profile = profile.public_snapshot() if profile else _offline_profile()
    experiment_id = str(uuid4())
    folder = _new_experiment_folder(output_dir, experiment_id)
    manifest = {
        "schema_version": 1,
        "experiment_id": experiment_id,
        "status": "running",
        "created_at": datetime.now(UTC).isoformat(),
        "python": platform.python_version(),
        "engine_fingerprint": _engine_fingerprint(),
        "dataset_id": dataset.id,
        "dataset_fingerprint": dataset_fingerprint(dataset),
        "provenance": dataset.provenance,
        "case_count": len(cases),
        "case_ids": [case.id for case in cases],
        "presets": configs,
        "repeats": repeats,
        "seed": seed,
        "planned_runs": len(tasks),
        "planned_call_ceiling": ceiling,
        "max_total_model_calls": max_total_model_calls,
        "model_profile": public_profile,
        "model_limits": limits.model_dump(),
        "memory": "disabled",
        "retrieval": "frozen_local_lexical",
        "compression_backend": "lexical",
        "mode": "live_model_frozen_corpus" if allow_live else "offline_fixture",
        "quality_claim": False,
        "metric_version": "m8-rules-v1",
        "task_order": [{"case_id": c.id, "preset": p, "repeat": r} for c, p, r in tasks],
    }
    artifact = {"schema_version": 1, "manifest": manifest, "records": [], "summary": {}}
    save_json(folder / "dataset.json", dataset.model_dump(mode="json"))

    def checkpoint():
        artifact["summary"] = summarize(artifact["records"], manifest)
        save_json(folder / "results.json", artifact)

    checkpoint()
    try:
        for case, preset, rep in tasks:
            await asyncio.sleep(0)  # accept pending cancellation before starting another case
            config = configs[preset]
            if on_progress:
                on_progress(
                    f"评测 {len(artifact['records']) + 1}/{len(tasks)}：{case.id} / {preset}"
                )
            budget = ModelBudgetLimits(**{**limits.model_dump(), "max_calls": config["max_calls"]})
            row = {
                "case_id": case.id,
                "domain": case.domain,
                "preset": preset,
                "repeat": rep,
                "status": "failed",
                "run_id": None,
                "error_code": None,
                "degraded": False,
                "bundle_path": None,
                "bundle_sha256": None,
                "model_budget": None,
                "metrics": {
                    "citation_coverage": 0.0,
                    "citation_validity": 0.0,
                    "gold_fact_coverage": 0.0 if case.gold.facts else None,
                    "nonempty_output": False,
                    "failure_penalty": True,
                },
            }
            started = asyncio.get_running_loop().time()
            fatal = False
            primary_error = None
            db = folder / "research.sqlite3"
            with (
                ResearchStore(db) as store,
                DagJournal(db) as journal,
                ReviewJournal(db) as reviews,
            ):
                try:
                    async with AsyncExitStack() as stack:
                        provider = (
                            await stack.enter_async_context(ChatCompatibleProvider(settings))
                            if settings
                            else BenchmarkFixtureProvider()
                        )
                        compressor = (
                            HierarchicalCompressor(
                                build_encoder("lexical", None),
                                CompressionOptions(max_evidence_bytes=8192),
                            )
                            if config["compression"]
                            else None
                        )
                        result = await run_dag_pipeline(
                            case.question,
                            gateway=ModelGateway(provider),
                            retriever=LocalCorpusRetriever(
                                tuple(
                                    SourceDocument(url=d.url, title=d.title, text=d.text)
                                    for d in case.documents
                                )
                            ),
                            store=store,
                            journal=journal,
                            output_dir=folder / "reports",
                            options=DagOptions(concurrency=config["concurrency"], max_replans=0),
                            language="en",
                            source_mode="local_corpus",
                            compressor=compressor,
                            review_options=ReviewOptions() if config["review"] else None,
                            review_journal=reviews if config["review"] else None,
                            model_budget=budget,
                            model_profile=public_profile,
                            on_run_created=lambda run_id, row=row: row.update(run_id=run_id),
                        )
                        # Collect committed output BEFORE waiting for provider cleanup. A late
                        # cancellation stops the matrix, not an already committed research result.
                        bundle = store.get_report(row["run_id"])
                        row.update(
                            status="completed",
                            degraded=result.get("degraded", False),
                            metrics=score_report(bundle, case.gold.model_dump(mode="json")),
                            bundle_path=Path(result["evidence_path"])
                            .relative_to(folder)
                            .as_posix(),
                            bundle_sha256=fingerprint(bundle),
                        )
                    await asyncio.sleep(0)
                except asyncio.CancelledError as error:
                    primary_error = error
                    if row["status"] != "completed":
                        row.update(status="cancelled", error_code="cancelled")
                    raise
                except ResearchRunError as error:
                    primary_error = error
                    if row["status"] == "completed":
                        row["audit_error"] = "benchmark_audit"
                        fatal = True
                    else:
                        row.update(error_code=error.code, run_id=error.run_id)
                        fatal = not error.status_saved or not getattr(error, "audit_saved", False)
                except ResearchError as error:
                    primary_error = error
                    if row["status"] == "completed":
                        row["audit_error"] = "benchmark_audit"
                    else:
                        row["error_code"] = error.code
                    fatal = True
                except Exception as error:
                    primary_error = error
                    if row["status"] == "completed":
                        row["audit_error"] = "benchmark_audit"
                    else:
                        row["error_code"] = "unexpected"
                    fatal = True
                finally:
                    if row["run_id"] is not None:
                        try:
                            snapshot = journal.snapshot(row["run_id"])
                            row["model_budget"] = snapshot.get("model_budget") if snapshot else None
                            if row["model_budget"] is None:
                                fatal = True
                                row["audit_error"] = "benchmark_audit"
                        except Exception:
                            fatal = True
                            row["audit_error"] = "benchmark_audit"
                    row["elapsed_ms"] = round(
                        (asyncio.get_running_loop().time() - started) * 1000, 3
                    )
                    artifact["records"].append(row)
                    try:
                        checkpoint()  # failed persistence prevents starting another task
                    except Exception:
                        if primary_error is not None:
                            primary_error.add_note("评测检查点保存失败；不会继续后续任务。")
                            raise primary_error from None
                        raise
            if fatal:
                raise ResearchError(
                    "benchmark_audit", "评测执行或审计异常，保留记录并停止后续任务。"
                )
        manifest["status"] = "completed"
    except asyncio.CancelledError as error:
        manifest["status"] = "cancelled"
        try:
            checkpoint()
        except Exception:
            error.add_note("取消状态无法写入评测文件；原始取消仍然传播。")
        raise
    except BaseException as error:
        manifest["status"] = "interrupted"
        try:
            checkpoint()
        except Exception:
            error.add_note("评测中断状态无法写入文件；保留原始异常。")
        raise
    checkpoint()
    (folder / "report.md").write_text(render_summary(artifact), encoding="utf-8")
    return {
        "status": "completed",
        "mode": manifest["mode"],
        "experiment_id": experiment_id,
        "results_path": str(folder / "results.json"),
        "report_path": str(folder / "report.md"),
        "planned_call_ceiling": ceiling,
        "summary": artifact["summary"],
        "quality_claim": False,
    }


def compare_results(
    path: Path,
    *,
    baseline: str = "baseline",
    candidate: str = "full",
    metric: str = "citation_coverage",
    seed: int = 42,
    n_resamples: int = 2000,
) -> dict:
    artifact = load_verified_results(path)
    if (
        not isinstance(artifact, dict)
        or artifact.get("schema_version") != 1
        or metric not in METRICS
        or baseline == candidate
    ):
        raise ResearchError("benchmark_artifact", "评测记录或比较指标无效。")
    try:
        manifest = artifact["manifest"]
        if baseline not in manifest["presets"] or candidate not in manifest["presets"]:
            raise ValueError
        mappings = []
        for name in (baseline, candidate):
            grouped = defaultdict(dict)
            for row in artifact["records"]:
                if row["preset"] != name:
                    continue
                if (
                    row["case_id"] not in manifest["case_ids"]
                    or type(row["repeat"]) is not int
                    or not 0 <= row["repeat"] < manifest["repeats"]
                    or row["repeat"] in grouped[row["case_id"]]
                ):
                    raise ValueError
                grouped[row["case_id"]][row["repeat"]] = row["metrics"][metric]
            means = {}
            for case_id in manifest["case_ids"]:
                values = list(grouped[case_id].values())
                if len(values) != manifest["repeats"] or any(v is None for v in values):
                    means[case_id] = None
                elif any(type(v) not in (float, int) or not 0 <= v <= 1 for v in values):
                    raise ValueError
                else:
                    means[case_id] = sum(values) / len(values)
            mappings.append(means)
        result = compare_paired(*mappings, seed=seed, n_resamples=n_resamples)
    except (KeyError, TypeError, ValueError):
        raise ResearchError("benchmark_artifact", "评测记录不完整或配对数据无效。") from None
    return {
        "baseline": baseline,
        "candidate": candidate,
        "metric": metric,
        "mode": manifest["mode"],
        "experiment_id": manifest["experiment_id"],
        "dataset_fingerprint": manifest["dataset_fingerprint"],
        "paired_by": "case_id",
        "repeats_aggregated_first": True,
        "exploratory": True,
        "quality_claim": False,
        **result,
    }


def load_verified_results(path: Path) -> dict:
    """Recompute scores from bound source artifacts; hashes are integrity, not signatures."""
    from deep_research.benchmark_artifacts import validate_results

    artifact = validate_results(read_json(path))
    folder = path.resolve().parent
    try:
        data_path = (folder / "dataset.json").resolve()
        if not data_path.is_relative_to(folder):
            raise ValueError
        dataset = load_benchmark(data_path)
        if dataset_fingerprint(dataset) != artifact["manifest"]["dataset_fingerprint"]:
            raise ValueError
        cases = {case.id: case for case in dataset.cases}
        if not set(artifact["manifest"]["case_ids"]) <= cases.keys():
            raise ValueError
        for row in artifact["records"]:
            case = cases[row["case_id"]]
            if row["domain"] != case.domain:
                raise ValueError
            if row["status"] != "completed":
                expected = 0.0 if case.gold.facts else None
                if row["metrics"]["gold_fact_coverage"] != expected:
                    raise ValueError
                continue
            bundle_path = (folder / row["bundle_path"]).resolve()
            if not bundle_path.is_relative_to(folder / "reports") or bundle_path.suffix != ".json":
                raise ValueError
            bundle = read_json(bundle_path)
            if (
                fingerprint(bundle) != row["bundle_sha256"]
                or bundle["question"] != case.question
                or bundle["run_id"] != row["run_id"]
            ):
                raise ValueError
            source_map = {doc.url: doc for doc in case.documents}
            for source in bundle["sources"]:
                document = source_map[source["url"]]
                if source["text"] != document.text or source["title"] != document.title:
                    raise ValueError
            if score_report(bundle, case.gold.model_dump(mode="json")) != row["metrics"]:
                raise ValueError
            if (
                row["model_budget"] is not None
                and bundle.get("model_budget") != row["model_budget"]
            ):
                raise ValueError
    except (KeyError, TypeError, ValueError, AttributeError):
        raise ResearchError(
            "benchmark_artifact", "评测题库、报告、指标或来源指纹不匹配。"
        ) from None
    return artifact
