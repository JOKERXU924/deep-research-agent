"""Offline M5 fixtures: deliberate reviewer opinions, never quality benchmarks."""

import asyncio
import json
from pathlib import Path

from deep_research.dag_demo import DagDemoProvider, DagDemoRetriever
from deep_research.dag_journal import DagJournal
from deep_research.dag_research import DagOptions, run_dag_pipeline
from deep_research.demo import DEMO_QUESTION
from deep_research.errors import ResearchError
from deep_research.gateway import ModelGateway
from deep_research.research_models import ResearchReport
from deep_research.research_store import ResearchStore
from deep_research.review_journal import ReviewJournal
from deep_research.review_models import ReviewOptions
from deep_research.schemas import ModelRequest, ModelResponse

SCENARIOS = frozenset(
    {"passed", "repair", "oscillation", "stalled", "max-rounds", "format", "deadline"}
)


class ReviewDemoProvider:
    """Scripted scores and repairs exercise state transitions, not factual accuracy."""

    def __init__(self, scenario: str = "repair") -> None:
        if scenario not in SCENARIOS:
            raise ResearchError("review_configuration", "未知的离线审查演示场景。")
        self.scenario = scenario
        self.base = DagDemoProvider()
        self.review_calls = 0
        self.red_count = 0
        self.blue_count = 0
        self.format_sent = False
        self.initial: ResearchReport | None = None

    async def complete(self, request: ModelRequest) -> ModelResponse:
        context = json.loads(request.messages[-1].content)
        if "claim_inventory" not in context:
            return await self.base.complete(request)
        self.review_calls += 1
        if self.initial is None:
            self.initial = ResearchReport.model_validate(context["report"])
        if self.scenario == "deadline":
            await asyncio.Event().wait()
        if self.scenario == "format" and not self.format_sent:
            self.format_sent = True
            return self._response("{deliberately invalid JSON")
        if "red_review" in context:
            self.blue_count += 1
            data = self._blue(context)
        else:
            self.red_count += 1
            data = self._red(context)
        return self._response(json.dumps(data, ensure_ascii=False))

    def _response(self, content: str) -> ModelResponse:
        return ModelResponse(
            provider="offline_fixture",
            model="review_demo",
            request_id=f"review-demo-{self.review_calls}",
            content=content,
            latency_ms=0.0,
        )

    def _red(self, context: dict) -> dict:
        if self.scenario == "passed":
            score = 90
        elif self.scenario == "repair":
            score = 55 if self.red_count == 1 else 90
        elif self.scenario == "format":
            score = (50, 70, 90)[min(self.red_count - 1, 2)]
        elif self.scenario == "max-rounds":
            score = (40, 60, 70)[min(self.red_count - 1, 2)]
        elif self.scenario == "oscillation":
            score = 50 if self.red_count == 1 else 70
        else:
            score = 55
        findings = []
        if score < 80:
            findings.append(
                {
                    "id": "F1",
                    "category": "evidence",
                    "severity": "medium",
                    "claim_ids": ["C1"],
                    "evidence_ids": context["claim_inventory"][0]["citations"],
                    "description": (
                        "离线教学替身预设的问题：给首条陈述补充资料范围限定。"
                        "此发现和分数用于演示状态迁移，不代表真实质量评价。"
                    ),
                }
            )
        return {
            "report_hash": context["report_hash"],
            "scores": {
                "evidence_support": score,
                "logical_consistency": score,
                "citation_quality": score,
            },
            "findings": findings,
        }

    def _blue(self, context: dict) -> dict:
        assert self.initial is not None
        original = self.initial.sections[0].claims[0]
        if self.scenario == "stalled":
            claim = context["report"]["sections"][0]["claims"][0]
        elif self.scenario == "oscillation" and self.blue_count == 2:
            claim = original.model_dump()
        else:
            prefix = (
                "教学修订一（仅反映所给资料）："
                if self.blue_count == 1
                else "教学修订二（仍须人工核对原文）："
            )
            claim = {"text": prefix + original.text, "citations": list(original.citations)}
        return {
            "report_hash": context["report_hash"],
            "operations": [
                {
                    "op": "MODIFY",
                    "target_id": "C1",
                    "claim": claim,
                    "rationale": "固定教学修订，仅演示有界补丁与审计，不增加来源。",
                }
            ],
        }


async def run_review_demo(*, db: Path, output_dir: Path, scenario: str = "repair") -> dict:
    if scenario not in SCENARIOS:
        raise ResearchError("review_configuration", "未知的离线审查演示场景。")
    provider = ReviewDemoProvider(scenario)
    options = DagOptions(total_timeout_seconds=0.5 if scenario == "deadline" else 120.0)
    with ResearchStore(db) as store, DagJournal(db) as journal, ReviewJournal(db) as reviews:
        result = await run_dag_pipeline(
            DEMO_QUESTION,
            gateway=ModelGateway(provider),
            retriever=DagDemoRetriever(),
            store=store,
            journal=journal,
            output_dir=output_dir,
            options=options,
            source_mode="offline_demo",
            review_options=ReviewOptions(),
            review_journal=reviews,
        )
    return {**result, "scenario": scenario, "mode": "offline_demo"}
