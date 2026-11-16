"""Offline M2 scenarios; failures are deliberate fixtures, never real outages."""

import asyncio
import json

from deep_research.demo import DEMO_DOCUMENTS
from deep_research.retrieval import ResearchRetrievalError, SearchHit, SourceDocument
from deep_research.schemas import ModelRequest, ModelResponse


class DagDemoProvider:
    def __init__(self) -> None:
        self.calls = 0

    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.calls += 1
        context = json.loads(request.messages[-1].content)
        if "evidence" in context:
            data = {
                "title": "M2 离线演示：DAG 证据报告",
                "sections": [
                    {
                        "heading": "固定资料摘录",
                        "claims": [
                            {"text": item["quote"], "citations": [item["evidence_id"]]}
                            for item in context["evidence"][:2]
                        ],
                    }
                ],
                "limitations": ["这是固定教学资料和假模型的演示，不是真实研究结果。"],
            }
        elif "previous_queries" in context:
            data = {"queries": ["恢复资料"]}
        else:
            data = {"queries": ["资料A", "资料B"]}
        return ModelResponse(
            provider="offline_fixture",
            model="dag_demo",
            request_id=f"dag-demo-{self.calls}",
            content=json.dumps(data, ensure_ascii=False),
            latency_ms=0.0,
        )


class DagDemoRetriever:
    def __init__(self, scenario: str = "normal") -> None:
        if scenario not in ("normal", "replan", "deadline"):
            raise ValueError("unknown offline scenario")
        self.scenario = scenario

    async def search(self, query: str, limit: int = 3) -> list[SearchHit]:
        if self.scenario == "replan" and query != "恢复资料":
            raise ResearchRetrievalError("retrieval_network", "离线演示模拟检索失败。")
        if self.scenario == "deadline" and query == "资料B":
            await asyncio.Event().wait()
        documents = (
            DEMO_DOCUMENTS
            if query == "恢复资料"
            else (DEMO_DOCUMENTS[0 if query == "资料A" else 1],)
        )
        return [SearchHit(document.url, document.title) for document in documents][:limit]

    async def fetch(self, hit: SearchHit) -> SourceDocument:
        return next(document for document in DEMO_DOCUMENTS if document.url == hit.url)
