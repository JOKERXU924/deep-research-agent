"""Bounded retrieval policy and durable, text-only audit contracts."""

from typing import Annotated, Literal

from pydantic import Field, StringConstraints

from deep_research.schemas import StrictModel


class RetrievalQualityOptions(StrictModel):
    optimize_research: bool = False
    cache_embeddings: bool = False
    rerank_evidence: bool = False
    check_support: bool = False
    robust_contracts: bool = False
    directed_gaps: bool = False
    adaptive_budget: bool = False
    route_gaps: bool = False
    evidence_selection: Literal["baseline", "coverage", "multilingual", "contextual"] = "baseline"
    filter_irrelevant: bool = True
    gap_searches: int = Field(default=2, ge=0, le=2)
    gap_fetches: int = Field(default=2, ge=0, le=2)
    gap_timeout_seconds: float = Field(default=30.0, gt=0, le=30, allow_inf_nan=False)


ShortCode = Annotated[str, StringConstraints(pattern=r"^[a-z][a-z0-9_.-]{0,63}$")]
AuditText = Annotated[str, StringConstraints(min_length=1, max_length=2048)]


class RetrievalEvent(StrictModel):
    phase: Literal["initial", "replan", "gap"]
    kind: Literal["search", "fetch", "filter", "gap"]
    status: Literal[
        "started",
        "completed",
        "failed",
        "cancelled",
        "skipped",
        "accepted",
        "rejected",
        "no_new_evidence",
        "timed_out",
    ]
    request_id: ShortCode | None = None
    query: Annotated[str, StringConstraints(min_length=1, max_length=120)] | None = None
    url: AuditText | None = None
    title: Annotated[str, StringConstraints(min_length=1, max_length=512)] | None = None
    reason: ShortCode | None = None
    error_code: ShortCode | None = None
    source_id: Annotated[str, StringConstraints(pattern=r"^S[1-9][0-9]{0,2}$")] | None = None
    evidence_ids: list[Annotated[str, StringConstraints(pattern=r"^E[1-9][0-9]{0,2}$")]] = Field(
        default_factory=list, max_length=32
    )
    topic_terms: int | None = Field(default=None, ge=0, le=10000)
    title_matches: int | None = Field(default=None, ge=0, le=10000)
    body_matches: int | None = Field(default=None, ge=0, le=10000)
    score: float | None = Field(default=None, ge=0, le=1, allow_inf_nan=False)


def validate_retrieval_event(value: object) -> dict:
    return RetrievalEvent.model_validate(value).model_dump(exclude_none=True)
