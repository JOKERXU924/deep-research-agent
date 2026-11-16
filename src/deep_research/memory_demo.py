"""Offline acceptance for reuse and conflict handling; no external model or data."""

from pathlib import Path
from uuid import uuid4

from deep_research.dag_demo import DagDemoProvider, DagDemoRetriever
from deep_research.dag_journal import DagJournal
from deep_research.dag_research import run_dag_pipeline
from deep_research.demo import DEMO_QUESTION
from deep_research.errors import ResearchError
from deep_research.gateway import ModelGateway
from deep_research.memory_store import MemoryStore
from deep_research.research_store import ResearchStore
from deep_research.retrieval import SearchHit, SourceDocument


class MemoryDemoRetriever:
    def __init__(self, document: SourceDocument) -> None:
        self.document = document

    async def search(self, query: str, limit: int = 3) -> list[SearchHit]:
        return [SearchHit(self.document.url, self.document.title)][:limit]

    async def fetch(self, hit: SearchHit) -> SourceDocument:
        if hit.url != self.document.url:
            raise ResearchError("retrieval_not_found", "演示资料不存在。")
        return self.document


async def run_memory_demo(*, db: Path, output_dir: Path, scenario: str = "reuse") -> dict:
    if scenario not in ("reuse", "conflict"):
        raise ResearchError("memory_configuration", "不支持的记忆演示场景。")
    # Each invocation gets its own scope, never mixing fictitious claims with live work.
    scope = "demo-" + uuid4().hex[:16]
    mode = "offline_demo"
    with ResearchStore(db) as store, DagJournal(db) as journal, MemoryStore(db) as memory:
        if scenario == "reuse":
            seed = await run_dag_pipeline(
                DEMO_QUESTION,
                gateway=ModelGateway(DagDemoProvider()),
                retriever=DagDemoRetriever(),
                store=store,
                journal=journal,
                output_dir=output_dir,
                source_mode=mode,
                memory=memory,
                memory_scope=scope,
            )
            extra = SourceDocument(
                "https://example.org/demo/m3-new-observation",
                "项目自编教学资料：历史证据与新观察",
                "并发上限用于限制同时运行的任务数量。历史证据需要保留原获取时间，"
                "不能因为再次被引用就假装完成了最新核验。",
            )
            reused = await run_dag_pipeline(
                DEMO_QUESTION,
                gateway=ModelGateway(DagDemoProvider()),
                retriever=MemoryDemoRetriever(extra),
                store=store,
                journal=journal,
                output_dir=output_dir,
                source_mode=mode,
                memory=memory,
                memory_scope=scope,
            )
            return {
                "status": "completed",
                "mode": mode,
                "scenario": scenario,
                "scope": scope,
                "seed_run": seed,
                "reuse_run": reused,
                "stats": memory.stats(scope=scope, source_mode=mode),
            }
        imports = []
        run_ids = []
        for label, text in (
            ("旧说法", "该演示服务允许匿名访问。"),
            ("新说法", "该演示服务禁止匿名访问。"),
        ):
            document = SourceDocument(
                "https://example.org/demo/m3-policy",
                "离线虚构规则：" + label,
                text,
            )
            run = await run_dag_pipeline(
                "该演示服务如何规定匿名访问？",
                gateway=ModelGateway(DagDemoProvider()),
                retriever=MemoryDemoRetriever(document),
                store=store,
                journal=journal,
                output_dir=output_dir,
                source_mode=mode,
            )
            run_ids.append(run["run_id"])
            imports.append(memory.ingest_run(run["run_id"], scope=scope))
        conflicts = memory.list_conflicts(scope=scope, source_mode=mode)
        before = memory.search("该演示服务匿名访问", scope=scope, source_mode=mode)
        # Explicit fixture action, not an autonomous choice about real sources.
        resolutions = [
            memory.resolve_conflict(
                item["id"],
                scope=scope,
                source_mode=mode,
                policy="keep_both",
                reason="离线演示选择保留双方用于比较；不判定哪一种说法真实。",
            )
            for item in conflicts
        ]
        after = memory.search("该演示服务匿名访问", scope=scope, source_mode=mode)
        return {
            "status": "completed",
            "mode": mode,
            "scenario": scenario,
            "scope": scope,
            "run_ids": run_ids,
            "imports": imports,
            "conflicts_before": conflicts,
            "search_before": before,
            "resolutions": resolutions,
            "search_after": after,
            "stats": memory.stats(scope=scope, source_mode=mode),
        }
