"""v1 -> v2 retrieval experiment. Old test cases are development; new sources form the holdout."""

import argparse
import asyncio
import importlib.util
import json
import statistics
import sys
import time
from pathlib import Path
from uuid import uuid4

from deep_research.benchmark_statistics import compare_paired
from deep_research.compression_runtime import json_bytes
from deep_research.embeddings import LocalE5Embedding
from deep_research.evidence_selection_v2 import select_coverage
from deep_research.retrieval_benchmark import CachedEncoder, digest, load_dataset, metrics, read


def load_v1(root):
    snapshot = root / "baselines/coverage-v1-20261027"
    manifest = read(snapshot / "manifest.json")
    relative = "src/deep_research/evidence_selection.py"
    path = snapshot / relative
    expected = next(r["sha256"] for r in manifest if r["path"].replace("\\", "/") == relative)
    if digest(path) != expected:
        raise ValueError("Frozen v1 baseline changed")
    for dependency in ("evidence_selection.py", "relevance.py", "compression_runtime.py"):
        relative_dependency = "src/deep_research/" + dependency
        record = next(r for r in manifest if r["path"].replace("\\", "/") == relative_dependency)
        if digest(root / relative_dependency) != record["sha256"]:
            raise ValueError("Baseline dependency changed: " + dependency)
    spec = importlib.util.spec_from_file_location("frozen_coverage_v1", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.select_coverage, expected


def fingerprint(root):
    paths = [
        "src/deep_research/evidence_selection_v2.py",
        "src/deep_research/evidence_selection.py",
        "src/deep_research/relevance.py",
        "src/deep_research/embeddings.py",
        "src/deep_research/compression_runtime.py",
        "src/deep_research/retrieval_benchmark.py",
        "src/deep_research/retrieval_benchmark_v2.py",
        "src/deep_research/benchmark_statistics.py",
        "uv.lock",
    ]
    return {p: digest(root / p) for p in paths}


async def run(args):
    root = (await asyncio.to_thread(Path(__file__).resolve)).parents[2]
    data_root = root / "data/retrieval-benchmark-v2"
    sources, data, hashes = await asyncio.to_thread(load_dataset, data_root / args.split)
    code_hashes = await asyncio.to_thread(fingerprint, root)
    if args.split == "test":
        freeze = await asyncio.to_thread(read, data_root / "candidate-freeze.json")
        if freeze["files"] != code_hashes:
            raise ValueError("Candidate changed after holdout freeze")
    old, old_hash = await asyncio.to_thread(load_v1, root)
    encoder = CachedEncoder(await asyncio.to_thread(LocalE5Embedding, args.model_path))
    await encoder.encode_queries(["retrieval evaluation"])
    records = []
    for case in data["cases"]:
        pool = [sources[s] for s in case["source_ids"]]
        variants = {}
        ceiling = case.get("byte_ceiling", 4000)
        for name, select in [("v1", old), ("v2", select_coverage)]:
            start = time.perf_counter()
            try:
                chosen = await select(
                    pool,
                    case["question"],
                    queries=case.get("queries"),
                    encoder=encoder,
                    max_bytes=ceiling,
                    max_items=16,
                )
                scored = metrics(chosen.evidence, case, sources)
                if scored["context_bytes"] > ceiling:
                    raise ValueError("Byte ceiling violated")
                variants[name] = dict(
                    evidence=chosen.evidence,
                    audit=chosen.audit,
                    metrics=scored,
                    elapsed_ms=(time.perf_counter() - start) * 1000,
                    error=None,
                )
                if name == "v1":
                    ceiling = min(ceiling, json_bytes(chosen.evidence))
            except Exception as error:
                variants[name] = dict(
                    evidence=[],
                    metrics=metrics([], case, sources),
                    elapsed_ms=(time.perf_counter() - start) * 1000,
                    error=type(error).__name__,
                )
        records.append(
            dict(
                id=case["id"],
                question=case["question"],
                topic=case["topic"],
                cross_language=case.get("cross_language"),
                gold=case["gold"],
                byte_ceiling=ceiling,
                variants=variants,
            )
        )
        print(
            case["id"]
            + ": "
            + " / ".join(f"{k}={v['metrics']['recall']:.3f}" for k, v in variants.items()),
            flush=True,
        )
    summary = {}
    for name in ("v1", "v2"):
        rows = [r["variants"][name] for r in records]
        summary[name] = dict(
            cases=len(rows),
            recall=statistics.mean(r["metrics"]["recall"] for r in rows),
            complete=sum(r["metrics"]["all_targets_found"] for r in rows),
            errors=sum(r["error"] is not None for r in rows),
            mean_bytes=statistics.mean(r["metrics"]["context_bytes"] for r in rows),
            quote_integrity=all(r["metrics"]["quote_integrity"] for r in rows),
            median_ms=statistics.median(r["elapsed_ms"] for r in rows),
        )
    cross = [r for r in records if r["cross_language"]]
    if cross:
        for name in summary:
            summary[name]["cross_language_recall"] = statistics.mean(
                r["variants"][name]["metrics"]["recall"] for r in cross
            )
            summary[name]["cross_language_cases"] = len(cross)
    paired = compare_paired(
        {r["id"]: r["variants"]["v1"]["metrics"]["recall"] for r in records},
        {r["id"]: r["variants"]["v2"]["metrics"]["recall"] for r in records},
    )
    result = dict(
        split=args.split,
        code_hashes=code_hashes,
        baseline_sha256=old_hash,
        dataset_hashes=hashes,
        encoder_id=encoder.encoder_id,
        annotation_origin=data["annotation_origin"],
        model_api_calls=0,
        network_calls=0,
        summaries=summary,
        comparison=paired,
        records=records,
        protocol=(
            "Same sources, max16 spans, ceiling4000 bytes (known photo3923); "
            "v2 capped at v1 actual bytes. Post-fetch selection only."
        ),
        timing_note=(
            "Evaluation caches document vectors across questions and variants; "
            "not a production speed claim."
        ),
    )
    output = args.output_dir / str(uuid4())
    await asyncio.to_thread(output.mkdir, parents=True)
    await asyncio.to_thread(
        (output / "results.json").write_text,
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    lines = [
        "# 证据检索 v1 / v2 对照",
        "",
        f"划分：{args.split}；AI 辅助标注，尚未独立人工复核。",
        "",
        "|版本|宏平均召回|全部要点命中|平均字节|错误题数|",
        "|---|---:|---:|---:|---:|",
    ]
    for name, s in summary.items():
        lines.append(
            f"|{name}|{s['recall']:.2%}|{s['complete']}/{s['cases']}|{s['mean_bytes']:.1f}|{s['errors']}|"
        )
    lines += ["", "## 未完全命中与退步", ""]
    for r in records:
        if r["variants"]["v2"]["metrics"]["recall"] < 1:
            lines.append(
                f"- {r['id']}: v1 {r['variants']['v1']['metrics']['recall']:.2%} → "
                f"v2 {r['variants']['v2']['metrics']['recall']:.2%}"
            )
    lines += [
        "",
        "只评测固定原文选段，非全网检索或报告事实准确率。新版本逐题不超过旧版实际字节量；失败保留在分母。",
        "AI 辅助构造题目与精确原文标注，未经独立人工复核；同源题目相关，不能作为总体质量保证。",
        "旧版此前的 77.5% 属于旧留出集和旧预算协议，不可与本页不同数据的成绩直接相减。",
        "详细位置、逐题选段、代码与数据哈希见 results.json。",
    ]
    await asyncio.to_thread((output / "report.md").write_text, "\n".join(lines), encoding="utf-8")
    print(json.dumps(dict(output=str(output), summaries=summary), ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["dev", "test"], required=True)
    parser.add_argument("--model-path", type=Path, default=Path("models/multilingual-e5-small"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/retrieval-v2"))
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
