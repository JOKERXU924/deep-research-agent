"""Deterministic M7 demos: no credential access, downloads or provider requests."""

from hashlib import sha256
from pathlib import Path

from deep_research.dag_demo import DagDemoRetriever
from deep_research.dag_journal import DagJournal
from deep_research.dag_research import run_dag_pipeline
from deep_research.demo import DEMO_QUESTION
from deep_research.errors import ResearchError
from deep_research.gateway import ModelGateway
from deep_research.model_budget import ModelBudgetLimits
from deep_research.research import ResearchRunError
from deep_research.research_store import ResearchStore
from deep_research.review_demo import ReviewDemoProvider
from deep_research.review_journal import ReviewJournal
from deep_research.review_models import ReviewOptions
from deep_research.schemas import TokenUsage


class _TeachingUsage(ReviewDemoProvider):
    """Fixed test counters, never measured token usage or a cost estimate."""

    async def complete(self, request):
        response = await super().complete(request)
        return response.model_copy(
            update={"usage": TokenUsage(input_tokens=10, output_tokens=5, total_tokens=15)}
        )


class _UnknownUsage(_TeachingUsage):
    async def complete(self, request):
        response = await super().complete(request)
        return response.model_copy(update={"usage": None})


def offline_profile() -> dict:
    return {
        "id": "offline-budget-demo",
        "label": "固定资料与假模型（不联网）",
        "provider": "offline_demo",
        "model": "offline-fixtures",
        "fingerprint": sha256(b"m7-offline-fixtures-v1").hexdigest(),
    }


async def run_budget_demo(*, db: Path, output_dir: Path, scenario: str = "normal") -> dict:
    if scenario not in {"normal", "calls", "input", "output", "unknown"}:
        raise ResearchError("configuration", "未知预算演示场景。")
    limits = ModelBudgetLimits(
        max_calls=2 if scenario == "calls" else 9,
        max_input_bytes=1 if scenario == "input" else 9 * 65536,
        max_output_tokens=1 if scenario == "output" else 9 * 4096,
    )
    profile = offline_profile()
    provider = _UnknownUsage("repair") if scenario == "unknown" else _TeachingUsage("repair")
    with ResearchStore(db) as store, DagJournal(db) as journal, ReviewJournal(db) as review:
        try:
            result = await run_dag_pipeline(
                DEMO_QUESTION,
                gateway=ModelGateway(provider),
                retriever=DagDemoRetriever(),
                store=store,
                journal=journal,
                output_dir=output_dir,
                source_mode="offline_demo",
                review_options=ReviewOptions(),
                review_journal=review,
                model_budget=limits,
                model_profile=profile,
            )
        except ResearchRunError as error:
            if error.code != "model_budget_exceeded":
                raise
            snapshot = journal.snapshot(error.run_id)
            result = {
                "status": "failed",
                "expected_budget_stop": True,
                "error_code": error.code,
                "run_id": error.run_id,
                "model_profile": profile,
                "model_budget": snapshot["model_budget"],
                "note": "预算不足已阻止下一次请求；保留账本，不交付未完成报告。",
            }
    return {**result, "mode": "offline_demo", "scenario": scenario}
