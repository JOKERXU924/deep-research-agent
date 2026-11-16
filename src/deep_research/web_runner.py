"""M6 adapter: one existing research pipeline, not a second agent engine."""

import asyncio
from collections.abc import Callable
from contextlib import nullcontext
from pathlib import Path

from deep_research.budget_demo import offline_profile
from deep_research.cli import build_compressor
from deep_research.config import load_settings
from deep_research.dag_demo import DagDemoProvider, DagDemoRetriever
from deep_research.dag_journal import DagJournal
from deep_research.dag_research import DagOptions, run_dag_pipeline
from deep_research.demo import DEMO_QUESTION
from deep_research.embeddings import build_encoder
from deep_research.errors import ResearchError
from deep_research.gateway import ModelGateway
from deep_research.memory_store import MemoryStore
from deep_research.model_budget import ModelBudgetLimits
from deep_research.model_profiles import ProfileRegistry
from deep_research.providers.deepseek import DeepSeekProvider
from deep_research.research_store import ResearchStore
from deep_research.retrieval import WikipediaRetriever
from deep_research.retrieval_quality import RetrievalQualityOptions
from deep_research.review_demo import ReviewDemoProvider
from deep_research.review_journal import ReviewJournal
from deep_research.review_models import ReviewOptions
from deep_research.web_models import WebRequest


class _SlowProvider:
    """An explicitly labelled offline cancellation fixture, not provider latency."""

    def __init__(self, wrapped):
        self.wrapped = wrapped
        self.first = True

    async def complete(self, request):
        if self.first:
            self.first = False
            await asyncio.sleep(30)
        return await self.wrapped.complete(request)


def _summary(result: dict) -> dict:
    safe = {
        key: result[key]
        for key in (
            "status",
            "run_id",
            "engine",
            "source_count",
            "evidence_count",
            "model_call_count",
            "degraded",
            "reasons",
            "report_kind",
            "replan_count",
            "counters",
            "model_profile",
            "model_budget",
            "retrieval_quality",
            "gap_retrieval",
        )
        if key in result
    }
    if result.get("review") is not None:
        safe["review"] = {
            key: result["review"][key]
            for key in ("status", "stop_reason", "rounds", "calls", "final_scores")
            if key in result["review"]
        }
    if result.get("compression") is not None:
        safe["compression"] = {
            key: result["compression"][key]
            for key in ("status", "encoder_kind", "report_evidence_bytes")
            if key in result["compression"]
        }
    if result.get("memory_write") is not None:
        safe["memory_write"] = result["memory_write"]
    return safe


def make_runner(
    *,
    db: Path,
    output_dir: Path,
    env_file: Path = Path(".env"),
    allow_live: bool = False,
    embedding_backend: str = "lexical",
    embedding_model_path: Path | None = None,
    profiles: ProfileRegistry | None = None,
    evidence_selection: str = "coverage",
):
    """Paths and backend are startup configuration; a web request cannot choose them."""
    if evidence_selection not in ("coverage", "multilingual", "contextual"):
        raise ResearchError("web_configuration", "不支持该证据检索版本。")
    if embedding_backend not in ("lexical", "local-e5"):
        raise ResearchError("web_configuration", "不支持该服务端编码方式。")
    if embedding_backend == "local-e5" and embedding_model_path is None:
        raise ResearchError("web_configuration", "请指定已经准备好的本地模型目录。")
    if embedding_backend == "lexical" and embedding_model_path is not None:
        raise ResearchError("web_configuration", "词法模式不使用本地模型目录。")

    shared_encoder = None
    shared_encoder_lock = asyncio.Lock()

    async def reusable_encoder():
        nonlocal shared_encoder
        async with shared_encoder_lock:
            if shared_encoder is None:
                shared_encoder = await asyncio.to_thread(
                    build_encoder, "local-e5", embedding_model_path
                )
            return shared_encoder

    async def run(
        request: WebRequest,
        on_progress: Callable[[str], None],
        on_run_created: Callable[[str], None],
    ) -> dict:
        # Revalidate even a model_construct or mutated caller object. Never read
        # model credentials merely because a browser opens or submits a demo.
        request = WebRequest.model_validate(request.model_dump())
        live = request.mode == "live"
        if live and (not allow_live or not request.confirm_live):
            raise ResearchError("web_live_disabled", "未启用并确认真实研究。")
        selected_profile = (
            profiles.get(request.profile_id) if live and profiles is not None else None
        )
        if request.profile_id is not None and selected_profile is None:
            raise ResearchError("web_profile", "所选模型配置不可用。")
        question = request.question if live else DEMO_QUESTION
        compressor = build_compressor(
            "hierarchical" if request.compression else None,
            embedding_backend=embedding_backend if request.compression and live else "lexical",
            embedding_model_path=embedding_model_path if request.compression and live else None,
        )
        if live and compressor is not None and embedding_backend == "local-e5":
            if request.optimize_research:
                compressor.encoder = await reusable_encoder()
            on_progress("检查本地语义编码模型；尚未调用研究模型。")
            await compressor.encoder.encode_queries([question])
        evidence_encoder = None
        if live and embedding_backend == "local-e5" and compressor is None:
            evidence_encoder = (
                await reusable_encoder()
                if request.optimize_research
                else await asyncio.to_thread(build_encoder, "local-e5", embedding_model_path)
            )
            on_progress("检查本地 E5 证据检索模型；尚未调用研究模型。")
            await evidence_encoder.encode_queries([question[:300]])
        options = DagOptions(
            total_timeout_seconds=0.5 if not live and request.scenario == "deadline" else 240.0
        )
        with (
            ResearchStore(db) as store,
            DagJournal(db) as journal,
            ReviewJournal(db) if request.review else nullcontext() as reviews,
            MemoryStore(db) if request.memory_scope else nullcontext() as memory,
        ):
            common = {
                "store": store,
                "journal": journal,
                "output_dir": output_dir,
                "options": options,
                "on_progress": on_progress,
                "on_run_created": on_run_created,
            }
            if request.review:
                common.update(review_options=ReviewOptions(), review_journal=reviews)
            if compressor is not None:
                common["compressor"] = compressor
            if evidence_encoder is not None:
                common["evidence_encoder"] = evidence_encoder
            if memory is not None:
                common.update(memory=memory, memory_scope=request.memory_scope)
            if selected_profile is not None:
                common.update(
                    model_budget=selected_profile.budget,
                    model_profile=selected_profile.public_snapshot(),
                )
            elif not live:
                common.update(model_budget=ModelBudgetLimits(), model_profile=offline_profile())
            if live:
                if selected_profile is not None:
                    from deep_research.providers.chat_compatible import ChatCompatibleProvider

                    settings = selected_profile.load_settings(env_file)
                    provider_context = ChatCompatibleProvider(settings)
                else:
                    settings = load_settings(env_file)
                    provider_context = DeepSeekProvider(settings)
                async with (
                    provider_context as provider,
                    WikipediaRetriever(language="zh") as retriever,
                ):
                    result = await run_dag_pipeline(
                        question,
                        gateway=ModelGateway(provider),
                        retriever=retriever,
                        source_mode="wikipedia",
                        quality_options=RetrievalQualityOptions(
                            evidence_selection=evidence_selection,
                            robust_contracts=request.robust_contracts,
                            directed_gaps=request.robust_contracts,
                            rerank_evidence=request.rerank_evidence,
                            check_support=request.check_support,
                            optimize_research=request.optimize_research,
                        ),
                        **common,
                    )
            else:
                scenario = "repair" if request.scenario == "slow" else request.scenario
                provider = ReviewDemoProvider(scenario) if request.review else DagDemoProvider()
                if request.scenario == "slow":
                    provider = _SlowProvider(provider)
                result = await run_dag_pipeline(
                    question,
                    gateway=ModelGateway(provider),
                    retriever=DagDemoRetriever(),
                    source_mode="offline_demo",
                    **common,
                )
        return _summary(result)

    return run
