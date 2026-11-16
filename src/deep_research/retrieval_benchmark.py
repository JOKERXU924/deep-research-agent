"""Offline, frozen-source retrieval A/B; not a research-report accuracy benchmark."""

import argparse
import asyncio
import hashlib
import importlib.util
import json
import platform
import statistics
import time
from pathlib import Path
from uuid import uuid4

import numpy as np

from deep_research.benchmark_statistics import compare_paired
from deep_research.compression_runtime import json_bytes
from deep_research.embeddings import LocalE5Embedding
from deep_research.evidence_selection import select_coverage


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_dataset(root):
    manifest = read(root / "manifest.json")
    for name in ("corpus.json", "cases.json"):
        if digest(root / name) != manifest[name]:
            raise ValueError("Frozen dataset hash mismatch")
    sources = read(root / "corpus.json")
    dataset = read(root / "cases.json")
    index = {s["source_id"]: s for s in sources}
    if len(index) != len(sources) or len({c["id"] for c in dataset["cases"]}) != len(
        dataset["cases"]
    ):
        raise ValueError("Duplicate IDs")
    for s in sources:
        if hashlib.sha256(s["text"].encode()).hexdigest() != s["sha256"]:
            raise ValueError("Source hash mismatch")
    for case in dataset["cases"]:
        if not case["gold"] or any(s not in index for s in case["source_ids"]):
            raise ValueError("Missing sources/gold")
        for label in case["gold"]:
            if index[label["source_id"]]["text"][label["start"] : label["end"]] != label["quote"]:
                raise ValueError("Gold offset mismatch")
    return index, dataset, manifest


def baseline_selector(root):
    path = root / "src/deep_research/research.py"
    records = read(root / "manifest.json")
    expected = next(
        r["sha256"]
        for r in records
        if r["path"].replace("\\", "/") == "src/deep_research/research.py"
    )
    if digest(path) != expected:
        raise ValueError("Baseline source has changed")
    spec = importlib.util.spec_from_file_location("frozen_research_baseline", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.select_evidence, expected


def metrics(selected, case, sources):
    valid = all(
        sources[e["source_id"]]["text"][e["start"] : e["end"]] == e["quote"] for e in selected
    )
    hits = []
    for gold in case["gold"]:
        cursor = gold["start"]
        for item in sorted(selected, key=lambda e: e["start"]):
            if item["source_id"] == gold["source_id"] and item["start"] <= cursor < item["end"]:
                cursor = item["end"]
        hits.append(cursor >= gold["end"])
    target_sources = {g["source_id"] for g in case["gold"]}
    support = sum(
        any(
            e["source_id"] == g["source_id"] and e["start"] < g["end"] and e["end"] > g["start"]
            for g in case["relevant_paragraphs"]
        )
        for e in selected
    )
    return dict(
        recall=sum(hits) / len(hits),
        hits=hits,
        gold_count=len(hits),
        all_targets_found=all(hits),
        context_bytes=json_bytes(selected),
        selected_count=len(selected),
        quote_integrity=valid,
        off_topic_source_rate=sum(e["source_id"] not in target_sources for e in selected)
        / len(selected)
        if selected
        else None,
        annotated_paragraph_precision=support / len(selected) if selected else None,
    )


class CachedEncoder:
    """Evaluation-only in-memory cache; no labels, across cases on the same corpus."""

    def __init__(self, encoder):
        self.encoder = encoder
        self.encoder_id, self.dimensions, self.kind = (
            encoder.encoder_id,
            encoder.dimensions,
            encoder.kind,
        )
        self.documents = {}

    async def encode_queries(self, texts):
        return await self.encoder.encode_queries(texts)

    async def encode_documents(self, texts):
        missing = list(dict.fromkeys(t for t in texts if t not in self.documents))
        if missing:
            vectors = await self.encoder.encode_documents(missing)
            self.documents.update(zip(missing, vectors, strict=True))
        return np.stack([self.documents[t] for t in texts])


async def run(args):
    root = (await asyncio.to_thread(Path(__file__).resolve)).parents[2]
    dataset_root = root / "data/retrieval-benchmark-v1"
    sources, dataset, dataset_hashes = await asyncio.to_thread(load_dataset, dataset_root)
    freeze = await asyncio.to_thread(read, dataset_root / "candidate-freeze.json")
    current_hash = await asyncio.to_thread(
        digest, Path(__file__).with_name("evidence_selection.py")
    )
    if freeze["sha256"] != current_hash:
        raise ValueError(
            "Candidate differs from frozen ranking version; create a new experiment version"
        )
    baseline, baseline_hash = await asyncio.to_thread(
        baseline_selector, root / "baselines/retrieval-v1-20261027"
    )
    encoder = None
    preflight = time.perf_counter()
    if args.model_path:
        encoder = CachedEncoder(await asyncio.to_thread(LocalE5Embedding, args.model_path))
        await encoder.encode_queries(["retrieval benchmark warmup"])
    initialization_seconds = time.perf_counter() - preflight
    records = []
    for case in dataset["cases"]:
        if case["split"] != args.split:
            continue
        pool = [sources[s] for s in case["source_ids"]]
        start = time.perf_counter()
        old = baseline(pool, case["question"])
        baseline_ms = (time.perf_counter() - start) * 1000
        ceiling = min(12000, json_bytes(old))
        # Baseline never exceeds the current production cap on this frozen corpus.
        if json_bytes(old) > ceiling:
            raise ValueError("Baseline exceeds fixed budget; redefine protocol before comparison")
        variants = {
            "baseline": dict(
                evidence=old, metrics=metrics(old, case, sources), elapsed_ms=baseline_ms
            )
        }
        for name, backend in [
            ("coverage_lexical", None),
            *([("coverage_e5", encoder)] if encoder else []),
        ]:
            start = time.perf_counter()
            try:
                chosen = await select_coverage(
                    pool, case["question"], encoder=backend, max_bytes=ceiling
                )
                variants[name] = dict(
                    evidence=chosen.evidence,
                    audit=chosen.audit,
                    metrics=metrics(chosen.evidence, case, sources),
                    elapsed_ms=(time.perf_counter() - start) * 1000,
                    error=None,
                )
            except Exception as exc:
                # Failures stay in denominator with zero recall. Never print upstream bodies.
                variants[name] = dict(
                    evidence=[],
                    metrics=metrics([], case, sources),
                    elapsed_ms=(time.perf_counter() - start) * 1000,
                    error=type(exc).__name__,
                )
        records.append(
            dict(
                id=case["id"],
                topic=case["topic"],
                question=case["question"],
                byte_ceiling=ceiling,
                gold=case["gold"],
                variants=variants,
            )
        )
        print(
            f"{case['id']}: "
            + " / ".join(f"{k}={v['metrics']['recall']:.2f}" for k, v in variants.items()),
            flush=True,
        )
    summaries, comparisons = {}, {}
    for variant in records[0]["variants"]:
        rows = [r["variants"][variant] for r in records]
        summaries[variant] = dict(
            cases=len(rows),
            errors=sum(bool(r.get("error")) for r in rows),
            recall=statistics.mean(r["metrics"]["recall"] for r in rows),
            all_targets_found=sum(r["metrics"]["all_targets_found"] for r in rows),
            mean_context_bytes=statistics.mean(r["metrics"]["context_bytes"] for r in rows),
            quote_integrity=all(r["metrics"]["quote_integrity"] for r in rows),
            mean_off_topic_source_rate=statistics.mean(
                r["metrics"]["off_topic_source_rate"] or 0 for r in rows
            ),
            mean_annotated_paragraph_precision=statistics.mean(
                r["metrics"]["annotated_paragraph_precision"] or 0 for r in rows
            ),
            median_ms=statistics.median(r["elapsed_ms"] for r in rows),
            p95_ms=float(np.quantile([r["elapsed_ms"] for r in rows], 0.95)),
        )
        if variant != "baseline":
            comparisons[variant] = compare_paired(
                {r["id"]: r["variants"]["baseline"]["metrics"]["recall"] for r in records},
                {r["id"]: r["variants"][variant]["metrics"]["recall"] for r in records},
            )
    result = dict(
        protocol=dataset["protocol"],
        annotation_origin=dataset["annotation_origin"],
        split=args.split,
        dataset_hashes=dataset_hashes,
        baseline_sha256=baseline_hash,
        candidate_sha256=current_hash,
        platform=platform.platform(),
        python=platform.python_version(),
        initialization_seconds=initialization_seconds,
        timing_note=(
            "Single pass; E5 includes first document encodings then in-memory reuse. "
            "Not production latency or a speedup claim."
        ),
        model_api_calls=0,
        network_calls=0,
        summaries=summaries,
        comparisons=comparisons,
        records=records,
    )
    destination = args.output_dir / str(uuid4())
    await asyncio.to_thread(destination.mkdir, parents=True)
    await asyncio.to_thread(
        (destination / "results.json").write_text,
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    lines = [
        "# 冻结原文证据检索对照",
        "",
        f"划分：{args.split}；题数：{len(records)}。标注：AI 草拟，未经独立人工复核。",
        "",
        "每题新旧使用相同原文，新版 UTF-8 证据数组上限等于旧版实际字节数。"
        "只测抓取后的证据选择，不测搜索召回或报告事实准确率。",
        "",
        "|版本|关键证据召回|全部要点命中题数|平均字节|跨主题来源片段占比|错误题数|",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, s in summaries.items():
        lines.append(
            f"|{name}|{s['recall']:.2%}|{s['all_targets_found']}|{s['mean_context_bytes']:.0f}|{s['mean_off_topic_source_rate']:.2%}|{s['errors']}|"
        )
    lines += ["", "## 未完全命中的题目（包括退步）", ""]
    for record in records:
        for name, value in record["variants"].items():
            if name != "baseline" and value["metrics"]["recall"] < 1:
                lines.append(
                    f"- {record['id']} / {name}: {value['metrics']['recall']:.2%}；"
                    f"基线 {record['variants']['baseline']['metrics']['recall']:.2%}"
                )
    lines += [
        "",
        "逐题原文、位置、选中片段、未命中标注、代码/数据哈希及配对 Bootstrap 见 results.json。",
        "跨主题比例不是全部无关片段比例；同主题中的非答案内容未全部人工判定。五类都是技术文档，测试与开发共享文档，不能推断跨领域泛化。",
        "E5 时间含首次原文编码与后续内存复用，不与旧版宣称加速。"
        "CPU 编码不消耗报告模型额度；本次没有 DeepSeek 调用。",
    ]
    await asyncio.to_thread(
        (destination / "report.md").write_text, "\n".join(lines), encoding="utf-8"
    )
    print(
        json.dumps(dict(output=str(destination), summaries=summaries), ensure_ascii=False, indent=2)
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["dev", "test"], required=True)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/retrieval-benchmark"))
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
