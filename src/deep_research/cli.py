"""Bounded research and explicit local evidence memory; live calls are explicit."""

import argparse
import asyncio
import json
import re
import sys
from contextlib import nullcontext
from pathlib import Path

from pydantic import ValidationError

from deep_research.config import load_settings
from deep_research.dag_demo import DagDemoProvider, DagDemoRetriever
from deep_research.dag_journal import DagJournal
from deep_research.dag_research import DagOptions, run_dag_pipeline
from deep_research.demo import DEMO_QUESTION, DemoProvider, DemoRetriever
from deep_research.errors import ResearchError
from deep_research.gateway import ModelGateway
from deep_research.memory_store import MemoryStore
from deep_research.providers.deepseek import DeepSeekProvider
from deep_research.research import run_research_pipeline
from deep_research.research_models import ResearchInput
from deep_research.research_store import ResearchStore
from deep_research.retrieval import LocalCorpusRetriever, WikipediaRetriever
from deep_research.schemas import ChatMessage, ModelRequest, ResearchOutline


async def run_smoke(env_file: Path, prompt: str) -> dict[str, object]:
    settings = load_settings(env_file)
    request = ModelRequest(messages=(ChatMessage(role="user", content=prompt),))
    async with DeepSeekProvider(settings) as provider:
        result = await ModelGateway(provider).generate_structured(request, ResearchOutline)
    response = result.response
    return {
        "status": "ok",
        "provider": response.provider,
        "model": response.model,
        "request_id": response.request_id,
        "latency_ms": round(response.latency_ms, 2),
        "usage": response.usage.model_dump() if response.usage is not None else None,
        "outline": result.data.model_dump(),
    }


def _progress(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


async def run_research(
    question: str,
    *,
    env_file: Path,
    language: str,
    corpus: Path | None,
    db: Path,
    output_dir: Path,
) -> dict:
    # Validate before opening the database or creating paid model calls.
    ResearchInput(question=question, language=language)
    settings = load_settings(env_file)
    local = LocalCorpusRetriever.from_file(corpus) if corpus is not None else None
    with ResearchStore(db) as store:
        async with DeepSeekProvider(settings) as provider:
            gateway = ModelGateway(provider)
            if local is not None:
                return await run_research_pipeline(
                    question,
                    gateway=gateway,
                    retriever=local,
                    store=store,
                    output_dir=output_dir,
                    language=language,
                    source_mode="local_corpus",
                    on_progress=_progress,
                )
            async with WikipediaRetriever(language=language) as retriever:
                return await run_research_pipeline(
                    question,
                    gateway=gateway,
                    retriever=retriever,
                    store=store,
                    output_dir=output_dir,
                    language=language,
                    source_mode="wikipedia",
                    on_progress=_progress,
                )


async def run_demo(*, db: Path, output_dir: Path) -> dict:
    with ResearchStore(db) as store:
        result = await run_research_pipeline(
            DEMO_QUESTION,
            gateway=ModelGateway(DemoProvider()),
            retriever=DemoRetriever(),
            store=store,
            output_dir=output_dir,
            source_mode="offline_demo",
            on_progress=_progress,
        )
    return {**result, "mode": "offline_demo"}


def show_research(run_id: str, db: Path) -> dict:
    if not db.is_file():
        raise ResearchError("not_found", "研究数据库不存在，请先运行研究任务。")
    with ResearchStore(db) as store:
        result = store.get_run(run_id)
    if result is None:
        raise ResearchError("not_found", "没有找到该研究任务。")
    return result


def _validate_memory_options(scope: str | None, max_age_days: int) -> None:
    if scope is not None and (
        not isinstance(scope, str) or re.fullmatch(r"[A-Za-z0-9_-]{1,64}", scope) is None
    ):
        raise ResearchError("memory_configuration", "scope 需为1到64位字母、数字、下划线或连字符。")
    if type(max_age_days) is not int or not 1 <= max_age_days <= 3650:
        raise ResearchError("memory_configuration", "记忆年龄上限需为1到3650天。")


def build_compressor(
    compression: str | None,
    *,
    embedding_backend: str = "lexical",
    embedding_model_path: Path | None = None,
    context_bytes: int = 8192,
):
    """Explicit opt-in; validate before model settings, storage, or paid requests."""
    if compression is None:
        if (
            embedding_backend != "lexical"
            or embedding_model_path is not None
            or context_bytes != 8192
        ):
            raise ResearchError(
                "compression_configuration", "请同时指定 --compression hierarchical。"
            )
        return None
    if compression != "hierarchical":
        raise ResearchError("compression_configuration", "不支持该压缩方式。")
    from deep_research.compression import CompressionOptions, HierarchicalCompressor
    from deep_research.embeddings import build_encoder

    try:
        options = CompressionOptions(max_evidence_bytes=context_bytes)
    except ValidationError:
        raise ResearchError(
            "compression_configuration", "证据预算必须为512到24576个UTF-8字节。"
        ) from None
    return HierarchicalCompressor(build_encoder(embedding_backend, embedding_model_path), options)


def build_review_options(
    review: str | None,
    *,
    review_rounds: int = 2,
    review_threshold: int = 80,
    review_format_repairs: int = 1,
):
    from deep_research.review_models import ReviewOptions

    if review is None:
        if (review_rounds, review_threshold, review_format_repairs) != (2, 80, 1):
            raise ResearchError("review_configuration", "请同时指定 --review red-blue。")
        return None
    if review != "red-blue":
        raise ResearchError("review_configuration", "不支持该审查方式。")
    try:
        return ReviewOptions(
            max_rounds=review_rounds,
            score_threshold=review_threshold,
            max_format_repairs=review_format_repairs,
        )
    except ValidationError:
        raise ResearchError(
            "review_configuration", "审查轮数应为0到2、阈值0到100、格式重试0到1。"
        ) from None


async def run_dag_research(
    question: str,
    *,
    env_file: Path,
    language: str,
    corpus: Path | None,
    db: Path,
    output_dir: Path,
    concurrency: int = 3,
    task_timeout: float = 15,
    total_timeout: float = 240,
    max_replans: int = 1,
    memory_scope: str | None = None,
    memory_max_age_days: int = 30,
    compression: str | None = None,
    embedding_backend: str = "lexical",
    embedding_model_path: Path | None = None,
    context_bytes: int = 8192,
    review: str | None = None,
    review_rounds: int = 2,
    review_threshold: int = 80,
    review_format_repairs: int = 1,
    profiles_file: Path | None = None,
    profile_id: str | None = None,
    max_model_calls: int | None = None,
    evidence_selection: str = "coverage",
    robust_contracts: bool = False,
    optimize_research: bool = False,
) -> dict:
    ResearchInput(question=question, language=language)
    if evidence_selection not in ("coverage", "multilingual", "contextual"):
        raise ResearchError("configuration", "不支持该证据检索版本。")
    selected_profile = None
    if profile_id is not None and profiles_file is None:
        raise ResearchError("configuration", "--profile 需要同时指定 --profiles。")
    if profiles_file is not None:
        from deep_research.model_profiles import load_profiles

        selected_profile = load_profiles(profiles_file).get(profile_id)
    budget = None
    if selected_profile is not None or max_model_calls is not None:
        from deep_research.model_budget import ModelBudgetLimits

        budget = selected_profile.budget if selected_profile is not None else ModelBudgetLimits()
        if max_model_calls is not None:
            if type(max_model_calls) is not int or not 1 <= max_model_calls <= budget.max_calls:
                raise ResearchError("configuration", "调用上限必须为正整数且不能超过配置预算。")
            budget = ModelBudgetLimits(**{**budget.model_dump(), "max_calls": max_model_calls})
    options = DagOptions(
        concurrency=concurrency,
        task_timeout_seconds=task_timeout,
        total_timeout_seconds=total_timeout,
        max_replans=max_replans,
    )
    _validate_memory_options(memory_scope, memory_max_age_days)
    review_options = build_review_options(
        review,
        review_rounds=review_rounds,
        review_threshold=review_threshold,
        review_format_repairs=review_format_repairs,
    )
    from deep_research.review_journal import ReviewJournal

    compressor = build_compressor(
        compression,
        embedding_backend=embedding_backend,
        embedding_model_path=embedding_model_path,
        context_bytes=context_bytes,
    )
    if compressor is not None and embedding_backend == "local-e5":
        # Surface missing weights/runtime or overlong query before a paid plan call.
        await compressor.encoder.encode_queries([question])
    optimizer_encoder = None
    if optimize_research and compressor is None and embedding_backend == "local-e5":
        from deep_research.embeddings import build_encoder

        optimizer_encoder = await asyncio.to_thread(build_encoder, "local-e5", embedding_model_path)
        await optimizer_encoder.encode_queries([question[:300]])
    settings = (
        selected_profile.load_settings(env_file)
        if selected_profile is not None
        else load_settings(env_file)
    )
    local = LocalCorpusRetriever.from_file(corpus) if corpus is not None else None
    with (
        ResearchStore(db) as store,
        DagJournal(db) as journal,
        MemoryStore(db) if memory_scope is not None else nullcontext() as memory,
        ReviewJournal(db) if review_options is not None else nullcontext() as review_journal,
    ):
        memory_options = (
            {
                "memory": memory,
                "memory_scope": memory_scope,
                "memory_max_age_days": memory_max_age_days,
            }
            if memory is not None
            else {}
        )
        if compressor is not None:
            memory_options["compressor"] = compressor
        if optimizer_encoder is not None:
            memory_options["evidence_encoder"] = optimizer_encoder
        if review_options is not None:
            memory_options.update(review_options=review_options, review_journal=review_journal)
        if budget is not None:
            memory_options["model_budget"] = budget
        if selected_profile is not None:
            from deep_research.providers.chat_compatible import ChatCompatibleProvider

            memory_options["model_profile"] = selected_profile.public_snapshot()
            provider_context = ChatCompatibleProvider(settings)
        else:
            provider_context = DeepSeekProvider(settings)
        async with provider_context as provider:
            gateway = ModelGateway(provider)
            if local is not None:
                if optimize_research:
                    from deep_research.retrieval_quality import RetrievalQualityOptions

                    memory_options["quality_options"] = RetrievalQualityOptions(
                        optimize_research=True
                    )
                return await run_dag_pipeline(
                    question,
                    gateway=gateway,
                    retriever=local,
                    store=store,
                    journal=journal,
                    output_dir=output_dir,
                    options=options,
                    language=language,
                    source_mode="local_corpus",
                    on_progress=_progress,
                    **memory_options,
                )
            async with WikipediaRetriever(language=language) as retriever:
                from deep_research.retrieval_quality import RetrievalQualityOptions

                return await run_dag_pipeline(
                    question,
                    gateway=gateway,
                    retriever=retriever,
                    store=store,
                    journal=journal,
                    output_dir=output_dir,
                    options=options,
                    language=language,
                    source_mode="wikipedia",
                    quality_options=RetrievalQualityOptions(
                        evidence_selection=evidence_selection,
                        robust_contracts=robust_contracts,
                        directed_gaps=robust_contracts,
                        optimize_research=optimize_research,
                    ),
                    on_progress=_progress,
                    **memory_options,
                )


async def run_dag_demo(
    *,
    db: Path,
    output_dir: Path,
    scenario: str = "normal",
    concurrency: int = 3,
) -> dict:
    options = DagOptions(
        concurrency=concurrency,
        total_timeout_seconds=0.5 if scenario == "deadline" else 30.0,
        task_timeout_seconds=5.0,
    )
    with ResearchStore(db) as store, DagJournal(db) as journal:
        result = await run_dag_pipeline(
            DEMO_QUESTION,
            gateway=ModelGateway(DagDemoProvider()),
            retriever=DagDemoRetriever(scenario),
            store=store,
            journal=journal,
            output_dir=output_dir,
            options=options,
            source_mode="offline_demo",
            on_progress=_progress,
        )
    return {**result, "mode": "offline_demo", "scenario": scenario}


def show_dag(run_id: str, db: Path) -> dict:
    run = show_research(run_id, db)
    with DagJournal(db) as journal:
        audit = journal.snapshot(run_id)
    if audit is None:
        raise ResearchError("not_found", "该任务没有 M2 调度记录。")
    return {"run": run, "audit": audit}


def show_review(run_id: str, db: Path) -> dict:
    from deep_research.review_journal import ReviewJournal

    result = show_dag(run_id, db)
    with ReviewJournal(db) as journal:
        events = journal.snapshot(run_id)
    if not events:
        raise ResearchError("review_not_found", "该任务没有 M5 审查事件；请检查调度调用账本。")
    return {"run": result["run"], "events": events, "model_calls": result["audit"]["model_calls"]}


def memory_command(args: argparse.Namespace) -> dict:
    """Local commands never load model settings or transmit recalled evidence."""
    if not args.db.is_file():
        raise ResearchError("not_found", "研究数据库不存在，请先完成一个研究任务。")
    with MemoryStore(args.db) as memory:
        if args.command == "memory-import":
            return memory.ingest_run(args.run_id, scope=args.scope)
        if args.command == "memory-search":
            return memory.search(
                args.query,
                scope=args.scope,
                source_mode=args.source_mode,
                limit=args.limit,
                max_age_days=args.max_age_days,
            )
        if args.command == "memory-conflicts":
            return {
                "scope": args.scope,
                "source_mode": args.source_mode,
                "conflicts": memory.list_conflicts(
                    scope=args.scope,
                    source_mode=args.source_mode,
                    status=args.status,
                ),
            }
        if args.command == "memory-resolve":
            return memory.resolve_conflict(
                args.conflict_id,
                scope=args.scope,
                source_mode=args.source_mode,
                policy=args.policy,
                reason=args.reason,
            )
        return memory.stats(scope=args.scope, source_mode=args.source_mode)


def serve_research(
    *,
    db: Path,
    output_dir: Path,
    env_file: Path,
    port: int = 8765,
    allow_live: bool = False,
    embedding_backend: str = "lexical",
    embedding_model_path: Path | None = None,
    profiles_file: Path | None = None,
    evidence_selection: str = "coverage",
) -> dict:
    """One local process. Network exposure, reload and multiple workers are not options."""
    if type(port) is not int or not 1024 <= port <= 65535:
        raise ResearchError("web_configuration", "本地端口必须为1024到65535。")
    if evidence_selection not in ("coverage", "multilingual", "contextual"):
        raise ResearchError("web_configuration", "不支持该证据检索版本。")
    if embedding_backend not in ("lexical", "local-e5") or (
        (embedding_backend == "local-e5") != (embedding_model_path is not None)
    ):
        raise ResearchError("web_configuration", "local-e5需指定本地模型目录；词法模式无需该目录。")
    import uvicorn

    from deep_research.web_app import create_app

    profile_options = {}
    if profiles_file is not None:
        from deep_research.model_profiles import load_profiles

        profile_options["profiles"] = load_profiles(profiles_file)
    app = create_app(
        db=db.resolve(),
        output_dir=output_dir.resolve(),
        env_file=env_file.resolve(),
        allow_live=allow_live,
        port=port,
        embedding_backend=embedding_backend,
        embedding_model_path=embedding_model_path.resolve() if embedding_model_path else None,
        evidence_selection=evidence_selection,
        **profile_options,
    )
    _progress(f"本地研究工作台：http://127.0.0.1:{port} （按 Ctrl+C 关闭）")
    _progress(
        "真实研究已启用：每次提交仍需确认数据发送和可能产生的费用。"
        if allow_live
        else "仅离线演示，不读取模型密钥，不调用真实研究服务。"
    )
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=port,
        workers=1,
        reload=False,
        proxy_headers=False,
        access_log=False,
        log_level="warning",
    )
    return {"status": "stopped", "mode": "local_web"}


def main() -> int:
    # Configure before argparse: --help and argument errors can exit immediately.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(
        description="DeepResearch Agent — M7 多模型与预算、本地工作台、DAG 证据与引用报告"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    smoke = subparsers.add_parser("smoke", help="执行一次真实 DeepSeek 调用并验证 JSON 结构")
    smoke.add_argument("--env-file", type=Path, default=Path(".env"))
    smoke.add_argument(
        "--prompt",
        default="请用中文为‘为什么研究报告需要来源引用？’生成提纲，包含 topic 和两个 questions。",
    )
    research = subparsers.add_parser("run", help="维基百科检索并生成引用报告；会调用真实模型")
    research.add_argument("question", help="研究问题；建议使用引号包住完整的一行")
    research.add_argument("--language", choices=("zh", "en"), default="zh")
    research.add_argument("--env-file", type=Path, default=Path(".env"))
    research.add_argument("--corpus", type=Path, help="改用本地JSON资料；仍会调用真实模型")
    demo = subparsers.add_parser("demo", help="固定资料+假模型：完全离线演示，不需要密钥")
    for command in (research, demo):
        command.add_argument("--db", type=Path, default=Path("data/research.sqlite3"))
        command.add_argument("--output-dir", type=Path, default=Path("outputs"))
    show = subparsers.add_parser("show", help="查看已保存任务状态，不请求模型")
    show.add_argument("run_id")
    show.add_argument("--db", type=Path, default=Path("data/research.sqlite3"))
    dag = subparsers.add_parser("run-dag", help="DAG研究：默认最多3次模型调用，启用M5默认最多9次")
    dag.add_argument("question")
    dag.add_argument("--language", choices=("zh", "en"), default="zh")
    dag.add_argument("--env-file", type=Path, default=Path(".env"))
    dag.add_argument("--corpus", type=Path)
    dag.add_argument("--task-timeout", type=float, default=15.0)
    dag.add_argument("--total-timeout", type=float, default=240.0)
    dag.add_argument("--max-replans", type=int, choices=(0, 1), default=1)
    dag.add_argument("--memory-scope", help="显式启用历史证据召回及写回；片段会发送给模型")
    dag.add_argument("--memory-max-age-days", type=int, default=30)
    dag.add_argument("--compression", choices=("hierarchical",), help="显式启用 M4 抽取式压缩")
    dag.add_argument("--embedding-backend", choices=("lexical", "local-e5"), default="lexical")
    dag.add_argument("--embedding-model-path", type=Path, help="已准备好的本地 E5 模型目录")
    dag.add_argument(
        "--context-bytes", type=int, default=8192, help="证据数组UTF-8字节预算，非token"
    )
    dag.add_argument("--review", choices=("red-blue",), help="显式启用M5；会增加真实模型调用")
    dag.add_argument("--review-rounds", type=int, choices=(0, 1, 2), default=2)
    dag.add_argument("--review-threshold", type=int, default=80)
    dag.add_argument("--review-format-repairs", type=int, choices=(0, 1), default=1)
    dag.add_argument("--profiles", type=Path, help="服务端模型配置TOML；不包含密钥值")
    dag.add_argument("--profile", help="选择配置中的模型ID；不自动回退其他服务")
    dag.add_argument("--max-model-calls", type=int, help="只收紧所选配置的调用预算")
    dag.add_argument(
        "--optimize-research",
        action="store_true",
        help="实验：逐要点改写、支持缺口补查、局部修订与编码缓存",
    )
    dag.add_argument(
        "--robust-contracts",
        action="store_true",
        help="实验版：逐要点生成、有界定向补查及局部修复；可能增加模型调用",
    )
    dag.add_argument(
        "--evidence-selection",
        choices=("coverage", "multilingual", "contextual"),
        default="coverage",
        help="Wikipedia 证据选段；multilingual 为 v2，contextual 为诊断与上下文 v3 实验版",
    )
    profiles_command = subparsers.add_parser(
        "profiles", help="校验并列出脱敏模型配置；不联网、不读密钥"
    )
    profiles_command.add_argument("--profiles", type=Path, required=True)
    budget_demo = subparsers.add_parser("demo-budget", help="M7离线演示调用预算与未知用量；不联网")
    budget_demo.add_argument(
        "--scenario", choices=("normal", "calls", "input", "output", "unknown"), default="normal"
    )
    budget_demo.add_argument("--db", type=Path, default=Path("data/budget-demo.sqlite3"))
    budget_demo.add_argument("--output-dir", type=Path, default=Path("outputs"))
    dag_demo = subparsers.add_parser("demo-dag", help="完全离线演示正常、重规划、超时降级")
    dag_demo.add_argument("--scenario", choices=("normal", "replan", "deadline"), default="normal")
    for command in (dag, dag_demo):
        command.add_argument("--concurrency", type=int, default=3)
        command.add_argument("--db", type=Path, default=Path("data/research.sqlite3"))
        command.add_argument("--output-dir", type=Path, default=Path("outputs"))
    dag_show = subparsers.add_parser("show-dag", help="读取DAG状态事件、重规划和调用审计，不联网")
    dag_show.add_argument("run_id")
    dag_show.add_argument("--db", type=Path, default=Path("data/research.sqlite3"))
    review_demo = subparsers.add_parser(
        "demo-review", help="M5假模型演示：修复、震荡和超时，不联网"
    )
    review_demo.add_argument(
        "--scenario",
        choices=("passed", "repair", "oscillation", "stalled", "max-rounds", "format", "deadline"),
        default="repair",
    )
    review_demo.add_argument("--db", type=Path, default=Path("data/review-demo.sqlite3"))
    review_demo.add_argument("--output-dir", type=Path, default=Path("outputs"))
    review_show = subparsers.add_parser(
        "show-review", help="读取M5逐轮审查、补丁及模型账本，不联网"
    )
    review_show.add_argument("run_id")
    review_show.add_argument("--db", type=Path, default=Path("data/review-demo.sqlite3"))
    memory_import = subparsers.add_parser(
        "memory-import", help="把已完成、未降级研究的原文证据加入记忆"
    )
    memory_import.add_argument("run_id")
    memory_search = subparsers.add_parser("memory-search", help="本地检索历史证据，不请求模型")
    memory_search.add_argument("query")
    memory_search.add_argument("--limit", type=int, default=4)
    memory_search.add_argument("--max-age-days", type=int, default=30)
    memory_conflicts = subparsers.add_parser(
        "memory-conflicts", help="查看待处理或已处理的候选冲突"
    )
    memory_conflicts.add_argument("--status", choices=("open", "resolved", "all"), default="open")
    memory_resolve = subparsers.add_parser(
        "memory-resolve", help="人工选择冲突策略；保留原文与处理记录"
    )
    memory_resolve.add_argument("conflict_id")
    memory_resolve.add_argument(
        "--policy", required=True, choices=("prefer_existing", "prefer_incoming", "keep_both")
    )
    memory_resolve.add_argument("--reason", required=True)
    memory_stats = subparsers.add_parser("memory-stats", help="查看指定命名空间的记忆统计")
    for command in (memory_import, memory_search, memory_conflicts, memory_resolve, memory_stats):
        command.add_argument("--scope", required=True)
        command.add_argument("--db", type=Path, default=Path("data/research.sqlite3"))
        if command is not memory_import:
            command.add_argument(
                "--source-mode",
                choices=("wikipedia", "local_corpus", "offline_demo"),
                default="wikipedia",
            )
    memory_demo = subparsers.add_parser(
        "demo-memory", help="离线演示跨任务复用或冲突处理，不需要密钥"
    )
    memory_demo.add_argument("--scenario", choices=("reuse", "conflict"), default="reuse")
    memory_demo.add_argument("--db", type=Path, default=Path("data/memory-demo.sqlite3"))
    memory_demo.add_argument("--output-dir", type=Path, default=Path("outputs"))
    compression_demo = subparsers.add_parser(
        "demo-compression", help="固定资料与假报告模型，演示分层压缩；不调用收费服务"
    )
    compression_demo.add_argument(
        "--scenario", choices=("normal", "tight", "deadline"), default="normal"
    )
    compression_demo.add_argument("--db", type=Path, default=Path("data/compression-demo.sqlite3"))
    compression_demo.add_argument("--output-dir", type=Path, default=Path("outputs"))
    compression_demo.add_argument(
        "--embedding-backend", choices=("lexical", "local-e5"), default="lexical"
    )
    compression_demo.add_argument("--embedding-model-path", type=Path)
    prepare = subparsers.add_parser("embedding-prepare", help="显式联网下载固定版本的本地 E5 模型")
    prepare.add_argument("--model-path", type=Path, default=Path("models/multilingual-e5-small"))
    check = subparsers.add_parser(
        "embedding-check", help="本地编码教学例句；不联网、不调用报告模型"
    )
    check.add_argument("--model-path", type=Path, default=Path("models/multilingual-e5-small"))
    serve = subparsers.add_parser("serve", help="启动本机Web/SSE工作台；默认仅离线演示")
    serve.add_argument("--port", type=int, default=8765)
    serve.add_argument("--db", type=Path, default=Path("data/web.sqlite3"))
    serve.add_argument("--output-dir", type=Path, default=Path("outputs/web"))
    serve.add_argument("--env-file", type=Path, default=Path(".env"))
    serve.add_argument(
        "--allow-live", action="store_true", help="允许用户确认后发起可能计费的真实研究"
    )
    serve.add_argument("--embedding-backend", choices=("lexical", "local-e5"), default="lexical")
    serve.add_argument("--embedding-model-path", type=Path)
    serve.add_argument("--profiles", type=Path, help="启动时读取不可变模型目录；网页仅选择已登记ID")
    serve.add_argument(
        "--evidence-selection",
        choices=("coverage", "multilingual", "contextual"),
        default="coverage",
        help="multilingual/contextual 显式启用 v2/v3 实验检索；默认保留 v1",
    )
    from deep_research.benchmark_cli import register_benchmark_commands

    register_benchmark_commands(subparsers)
    harness_start = subparsers.add_parser("harness-start", help="可恢复研究的离线教学演示")
    harness_start.add_argument("question", nargs="?", default=DEMO_QUESTION)
    harness_start.add_argument("--seconds", type=float, default=600)
    harness_start.add_argument("--force-refresh", action="store_true")
    harness_start.add_argument(
        "--stop-after-fetch", type=int, default=0, help="离线故障注入：保存指定数量资料后取消"
    )
    harness_resume = subparsers.add_parser("harness-resume", help="按原配置和剩余预算恢复")
    harness_resume.add_argument("harness_id")
    harness_show = subparsers.add_parser("harness-show", help="读取保存的执行轨迹，不调用模型")
    harness_show.add_argument("harness_id")
    for command in (harness_start, harness_resume, harness_show):
        command.add_argument("--db", type=Path, default=Path("data/harness-research.sqlite3"))
        command.add_argument("--output-dir", type=Path, default=Path("outputs/harness"))
    args = parser.parse_args()
    try:
        if args.command.startswith("bench-"):
            from deep_research.benchmark_cli import benchmark_command

            result = benchmark_command(args, on_progress=_progress)
        elif args.command in {"harness-start", "harness-resume", "harness-show"}:
            from deep_research.harness_runner import run_harness
            from deep_research.harness_store import HarnessStore

            if args.command == "harness-show":
                result = HarnessStore(args.db.with_suffix(".harness.sqlite3")).trace(
                    args.harness_id
                )
            elif args.command == "harness-resume":
                result = asyncio.run(
                    run_harness(db=args.db, output_dir=args.output_dir, resume=args.harness_id)
                )
            else:
                try:
                    result = asyncio.run(
                        run_harness(
                            db=args.db,
                            output_dir=args.output_dir,
                            question=args.question,
                            seconds=args.seconds,
                            force_refresh=args.force_refresh,
                            stop_after_fetch=args.stop_after_fetch,
                        )
                    )
                except asyncio.CancelledError:
                    print("已中断。检查点已保存；使用上方 Harness ID 显式恢复。")
                    return 1
        elif args.command == "serve":
            profile_options = {"profiles_file": args.profiles} if args.profiles is not None else {}
            result = serve_research(
                db=args.db,
                output_dir=args.output_dir,
                env_file=args.env_file,
                port=args.port,
                allow_live=args.allow_live,
                embedding_backend=args.embedding_backend,
                embedding_model_path=args.embedding_model_path,
                evidence_selection=args.evidence_selection,
                **profile_options,
            )
        elif args.command == "profiles":
            from deep_research.model_profiles import load_profiles

            registry = load_profiles(args.profiles)
            result = {
                "default_profile": registry.default_profile,
                "profiles": registry.public_catalog(),
                "credentials_checked": False,
            }
        elif args.command == "demo-budget":
            from deep_research.budget_demo import run_budget_demo

            result = asyncio.run(
                run_budget_demo(db=args.db, output_dir=args.output_dir, scenario=args.scenario)
            )
        elif args.command == "smoke":
            result = asyncio.run(run_smoke(args.env_file, args.prompt))
        elif args.command == "run":
            result = asyncio.run(
                run_research(
                    args.question,
                    env_file=args.env_file,
                    language=args.language,
                    corpus=args.corpus,
                    db=args.db,
                    output_dir=args.output_dir,
                )
            )
        elif args.command == "demo":
            result = asyncio.run(run_demo(db=args.db, output_dir=args.output_dir))
        elif args.command == "run-dag":
            memory_options = (
                {"memory_scope": args.memory_scope, "memory_max_age_days": args.memory_max_age_days}
                if args.memory_scope is not None
                else {}
            )
            if args.memory_scope is None and args.memory_max_age_days != 30:
                raise ResearchError("memory_configuration", "请同时指定 --memory-scope。")
            if (
                args.profiles is not None
                or args.profile is not None
                or args.max_model_calls is not None
            ):
                memory_options.update(
                    profiles_file=args.profiles,
                    profile_id=args.profile,
                    max_model_calls=args.max_model_calls,
                )
            if (
                args.compression is not None
                or args.embedding_backend != "lexical"
                or args.embedding_model_path is not None
                or args.context_bytes != 8192
            ):
                if args.compression is None:
                    raise ResearchError(
                        "compression_configuration", "请同时指定 --compression hierarchical。"
                    )
                memory_options.update(
                    compression=args.compression,
                    embedding_backend=args.embedding_backend,
                    embedding_model_path=args.embedding_model_path,
                    context_bytes=args.context_bytes,
                )
            if (
                args.review is not None
                or args.review_rounds != 2
                or args.review_threshold != 80
                or args.review_format_repairs != 1
            ):
                build_review_options(
                    args.review,
                    review_rounds=args.review_rounds,
                    review_threshold=args.review_threshold,
                    review_format_repairs=args.review_format_repairs,
                )
                memory_options.update(
                    review=args.review,
                    review_rounds=args.review_rounds,
                    review_threshold=args.review_threshold,
                    review_format_repairs=args.review_format_repairs,
                )
            if args.evidence_selection != "coverage":
                memory_options["evidence_selection"] = args.evidence_selection
            if args.robust_contracts:
                memory_options["robust_contracts"] = True
            if args.optimize_research:
                memory_options["optimize_research"] = True
            result = asyncio.run(
                run_dag_research(
                    args.question,
                    env_file=args.env_file,
                    language=args.language,
                    corpus=args.corpus,
                    db=args.db,
                    output_dir=args.output_dir,
                    concurrency=args.concurrency,
                    task_timeout=args.task_timeout,
                    total_timeout=args.total_timeout,
                    max_replans=args.max_replans,
                    **memory_options,
                )
            )
        elif args.command == "demo-dag":
            result = asyncio.run(
                run_dag_demo(
                    db=args.db,
                    output_dir=args.output_dir,
                    scenario=args.scenario,
                    concurrency=args.concurrency,
                )
            )
        elif args.command == "show-dag":
            result = show_dag(args.run_id, args.db)
        elif args.command == "show-review":
            result = show_review(args.run_id, args.db)
        elif args.command == "demo-review":
            from deep_research.review_demo import run_review_demo

            result = asyncio.run(
                run_review_demo(db=args.db, output_dir=args.output_dir, scenario=args.scenario)
            )
        elif args.command == "demo-memory":
            from deep_research.memory_demo import run_memory_demo

            result = asyncio.run(
                run_memory_demo(
                    db=args.db,
                    output_dir=args.output_dir,
                    scenario=args.scenario,
                )
            )
        elif args.command == "demo-compression":
            from deep_research.compression_demo import run_compression_demo

            result = asyncio.run(
                run_compression_demo(
                    db=args.db,
                    output_dir=args.output_dir,
                    scenario=args.scenario,
                    embedding_backend=args.embedding_backend,
                    embedding_model_path=args.embedding_model_path,
                )
            )
        elif args.command == "embedding-prepare":
            from deep_research.embeddings import prepare_local_e5

            result = asyncio.run(prepare_local_e5(args.model_path))
        elif args.command == "embedding-check":
            from deep_research.compression_demo import check_local_embedding

            result = asyncio.run(check_local_embedding(args.model_path))
        elif args.command.startswith("memory-"):
            result = memory_command(args)
        else:
            result = show_research(args.run_id, args.db)
    except ResearchError as error:
        detail = {"run_id": error.run_id} if hasattr(error, "run_id") else {}
        if hasattr(error, "status_saved"):
            detail["status_saved"] = error.status_saved
        if hasattr(error, "audit_saved"):
            detail["audit_saved"] = error.audit_saved
        print(
            json.dumps(
                {
                    "status": "error",
                    "code": error.code,
                    "message": str(error),
                    "http_status": error.status_code,
                    **detail,
                },
                ensure_ascii=False,
            ),
            file=sys.stderr,
        )
        return 1
    except ValidationError:
        print("输入无效：请检查消息内容与输入大小。", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("任务已取消。", file=sys.stderr)
        return 130
    except OSError:
        print("无法访问研究目录或文件，请检查路径与权限。", file=sys.stderr)
        return 1
    except Exception:
        print(
            json.dumps(
                {
                    "status": "error",
                    "code": "unexpected",
                    "message": "研究程序发生内部错误，请检查任务记录并运行离线测试。",
                },
                ensure_ascii=False,
            ),
            file=sys.stderr,
        )
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
