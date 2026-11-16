"""Inspectable M4 fixtures: real extraction, fake report model, no paid calls."""

import asyncio
import json
from pathlib import Path

from deep_research.compression import CompressionOptions, HierarchicalCompressor
from deep_research.dag_demo import DagDemoProvider
from deep_research.dag_journal import DagJournal
from deep_research.dag_research import DagOptions, run_dag_pipeline
from deep_research.embeddings import LexicalEmbedding, build_encoder
from deep_research.errors import ResearchError
from deep_research.gateway import ModelGateway
from deep_research.research_store import ResearchStore
from deep_research.retrieval import SearchHit, SourceDocument
from deep_research.schemas import ModelRequest, ModelResponse

QUESTION = "研究系统如何保留证据出处、控制并发，并在超时后如实报告？"
DOCUMENTS = (
    SourceDocument(
        "https://example.org/m4/evidence",
        "M4 虚构教学资料：证据",
        "\n".join(
            [f"教学背景第{i}段。图书馆的书架按编号排列，读者可以借阅纸质书。" for i in range(16)]
            + [
                "研究系统必须保存原文、来源地址和获取时间，以便读者核对证据出处。",
                "引用的字符位置必须对应保存的正文，不能把改写的句子伪装成原文。",
                "相似度排名只表示检索关联，不证明事实正确；压缩可能遗漏限定条件。",
            ]
        ),
    ),
    SourceDocument(
        "https://example.org/m4/scheduling",
        "M4 虚构教学资料：调度",
        "\n".join(
            [f"教学背景第{i}段。花园里种植多种花卉，园丁每天检查土壤湿度。" for i in range(16)]
            + [
                "并发调度使用信号量限制同时运行的任务，依赖完成后才能开始后续任务。",
                "全局超时后应取消并等待在途任务清理，不再追加模型请求。",
                "研究未完成时必须如实说明缺口，只能整理已经保存的原文证据。",
            ]
        ),
    ),
)


class CompressionDemoProvider(DagDemoProvider):
    async def complete(self, request: ModelRequest) -> ModelResponse:
        response = await super().complete(request)
        content = json.loads(response.content)
        if "sections" in content:
            content["title"] = "M4 离线演示：分层压缩证据报告"
        return response.model_copy(
            update={"content": json.dumps(content, ensure_ascii=False), "model": "compression_demo"}
        )


class CompressionDemoRetriever:
    async def search(self, query: str, limit: int = 3) -> list[SearchHit]:
        document = DOCUMENTS[0 if query == "资料A" else 1]
        return [SearchHit(document.url, document.title)][:limit]

    async def fetch(self, hit: SearchHit) -> SourceDocument:
        return next(document for document in DOCUMENTS if document.url == hit.url)


class DeadlineEncoder(LexicalEmbedding):
    """Deliberately wait until cancellation; never called by a real run."""

    async def encode_queries(self, texts: list[str]):
        await asyncio.Event().wait()


async def run_compression_demo(
    *,
    db: Path,
    output_dir: Path,
    scenario: str = "normal",
    embedding_backend: str = "lexical",
    embedding_model_path: Path | None = None,
) -> dict:
    if scenario not in ("normal", "tight", "deadline"):
        raise ResearchError("compression_configuration", "未知的压缩演示。")
    if scenario == "deadline" and (
        embedding_backend != "lexical" or embedding_model_path is not None
    ):
        raise ResearchError("compression_configuration", "超时演示只使用可控的离线替身。")
    encoder = (
        DeadlineEncoder()
        if scenario == "deadline"
        else build_encoder(embedding_backend, embedding_model_path)
    )
    if embedding_backend == "local-e5":
        await encoder.encode_queries([QUESTION])
    compressor = HierarchicalCompressor(
        encoder, CompressionOptions(max_evidence_bytes=512 if scenario == "tight" else 8192)
    )
    options = DagOptions(total_timeout_seconds=0.2 if scenario == "deadline" else 120.0)
    with ResearchStore(db) as store, DagJournal(db) as journal:
        result = await run_dag_pipeline(
            QUESTION,
            gateway=ModelGateway(CompressionDemoProvider()),
            retriever=CompressionDemoRetriever(),
            store=store,
            journal=journal,
            output_dir=output_dir,
            options=options,
            source_mode="offline_demo",
            compressor=compressor,
        )
    return {**result, "scenario": scenario, "mode": "offline_demo"}


async def check_local_embedding(model_path: Path) -> dict:
    encoder = build_encoder("local-e5", model_path)
    queries = await encoder.encode_queries(["植物怎样把光能变成化学能？"])
    passages = await encoder.encode_documents(
        [
            "光合作用将光能转化为化学能，并参与植物有机物的合成。",
            "操作系统使用进程调度算法安排 CPU 运行任务。",
        ]
    )
    scores = (queries @ passages.T)[0].tolist()
    return {
        "status": "ok",
        "encoder_id": encoder.encoder_id,
        "encoder_kind": encoder.kind,
        "dimensions": encoder.dimensions,
        "scores": scores,
        "expected_related_ranked_first": scores[0] > scores[1],
        "note": "仅验证本地模型加载和两条教学例句的排序，不是质量基准或准确率。",
    }
