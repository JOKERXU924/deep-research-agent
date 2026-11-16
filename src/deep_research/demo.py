"""Explicit offline fixture; never pretend this output came from a live model."""

import json

from deep_research.retrieval import SearchHit, SourceDocument
from deep_research.schemas import ModelRequest, ModelResponse

DEMO_QUESTION = "为什么异步适合处理网络等待，并发又需要什么限制？"
DEMO_DOCUMENTS = (
    SourceDocument(
        url="https://example.org/demo/async",
        title="项目自编教学资料：异步等待（非网页抓取）",
        text=(
            "异步程序在等待网络响应时，可以让出执行机会给其他任务。"
            "异步不会让远端服务本身计算得更快。\n"
            "本资料是本项目编写的离线测试样本，不对应真实抓取网页。"
        ),
    ),
    SourceDocument(
        url="https://example.org/demo/concurrency",
        title="项目自编教学资料：并发限制（非网页抓取）",
        text=(
            "并发网络请求需要限制数量，避免超出服务速率限制或本机资源容量。"
            "并发控制与超时、取消共同决定任务如何占用资源。\n"
            "本资料仅用于演示引用链，不用于衡量研究质量。"
        ),
    ),
)


class DemoRetriever:
    async def search(self, query: str, limit: int = 3) -> list[SearchHit]:
        return [SearchHit(url=item.url, title=item.title) for item in DEMO_DOCUMENTS][:limit]

    async def fetch(self, hit: SearchHit) -> SourceDocument:
        return next(item for item in DEMO_DOCUMENTS if item.url == hit.url)


class DemoProvider:
    """A deterministic test double implementing the M0 provider contract."""

    def __init__(self) -> None:
        self.calls = 0

    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.calls += 1
        if self.calls == 1:
            data = {"queries": ["异步", "并发限制"]}
        else:
            context = json.loads(request.messages[-1].content)
            evidence = context["evidence"]
            first = next(item for item in evidence if item["source_id"] == "S1")
            second = next(item for item in evidence if item["source_id"] == "S2")
            data = {
                "title": "离线演示：异步等待与并发限制",
                "sections": [
                    {
                        "heading": "教学资料中的两个要点",
                        "claims": [
                            {
                                "text": (
                                    "等待网络响应时，异步程序可以让其他任务运行，"
                                    "但不会加速远端计算。"
                                ),
                                "citations": [first["evidence_id"]],
                            },
                            {
                                "text": "并发请求需要数量限制，并结合超时、取消管理资源。",
                                "citations": [second["evidence_id"]],
                            },
                        ],
                    }
                ],
                "limitations": ["本例使用固定资料和假模型，只验证流程，不评价真实模型或检索质量。"],
            }
        return ModelResponse(
            provider="offline_fixture",
            model="deterministic_demo",
            request_id=f"demo-{self.calls}",
            content=json.dumps(data, ensure_ascii=False),
            latency_ms=0.0,
        )
