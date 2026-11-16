"""Keep archived evidence separate from the exact bounded report context."""

import json

from deep_research.errors import ResearchError
from deep_research.research_store import ResearchStore


def json_bytes(value: object) -> int:
    return len(json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))


def save_compressed_spans(
    store: ResearchStore, run_id: str, archived: list[dict], spans: list[dict]
) -> list[dict]:
    """Reuse identical span IDs; save all new spans in one transaction before exposing them."""
    existing = {
        (item["source_id"], item["start"], item["end"], item["quote"]): item for item in archived
    }
    selected: list[dict] = []
    additions: list[dict] = []
    seen: set[tuple] = set()
    for span in spans:
        if not isinstance(span, dict) or set(span) != {"source_id", "start", "end", "quote"}:
            raise ResearchError("compression_validation", "压缩器返回了不合法的证据。")
        key = (span["source_id"], span["start"], span["end"], span["quote"])
        if key in seen:
            continue
        seen.add(key)
        item = existing.get(key)
        if item is None:
            item = {"evidence_id": f"E{len(archived) + len(additions) + 1}", **span}
            additions.append(item)
        selected.append(item)
    if additions:
        store.save_sources(run_id, [], additions)
        archived.extend(additions)
    return selected


def fit_report_evidence(
    fresh: list[dict], historical: list[dict], *, max_bytes: int
) -> tuple[list[dict], list[str]]:
    """Reserve ALL historical observations and conflict pairs; never slice their quotes."""
    if json_bytes(historical) > max_bytes:
        raise ResearchError(
            "compression_budget", "历史证据超过上下文证据预算；请增加预算，不会静默删除冲突材料。"
        )
    kept: list[dict] = []
    omitted: list[str] = []
    for item in fresh:
        if json_bytes([*kept, item, *historical]) <= max_bytes:
            kept.append(item)
        else:
            omitted.append(item["evidence_id"])
    result = [*kept, *historical]
    if not result:
        raise ResearchError("compression_budget", "预算内没有完整证据片段，不发送空证据报告请求。")
    return result, omitted
