"""Clients select registered profile IDs, never paths, endpoints or secrets."""

from typing import Annotated, Literal, Self

from pydantic import Field, StringConstraints, model_validator

from deep_research.demo import DEMO_QUESTION
from deep_research.schemas import StrictModel


class WebRequest(StrictModel):
    mode: Literal["demo", "live"] = "demo"
    question: Annotated[
        str, StringConstraints(strip_whitespace=True, min_length=1, max_length=1000)
    ] = DEMO_QUESTION
    scenario: Literal[
        "passed", "repair", "oscillation", "stalled", "max-rounds", "format", "deadline", "slow"
    ] = "repair"
    review: bool = True
    compression: bool = False
    robust_contracts: bool = False
    rerank_evidence: bool = False
    check_support: bool = False
    optimize_research: bool = False
    memory_scope: Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9_-]{1,64}$")] | None = None
    confirm_live: bool = Field(default=False)
    profile_id: Annotated[str, StringConstraints(pattern=r"^[a-z][a-z0-9_-]{0,47}$")] | None = None

    @model_validator(mode="after")
    def validate_mode(self) -> Self:
        if self.mode == "live":
            if not self.confirm_live or self.scenario != "repair":
                raise ValueError("live research needs explicit confirmation and no demo scenario")
        elif (
            self.confirm_live
            or self.memory_scope is not None
            or self.profile_id is not None
            or self.robust_contracts
            or self.rerank_evidence
            or self.check_support
            or self.optimize_research
        ):
            raise ValueError("offline demos cannot confirm paid research or read shared memory")
        if self.mode == "demo" and self.question != DEMO_QUESTION:
            raise ValueError("offline demos only answer their fixed fixture question")
        if self.mode == "demo" and self.scenario != "repair" and not self.review:
            raise ValueError("this demo scenario needs review enabled")
        try:
            self.question.encode("utf-8")
        except UnicodeError:
            raise ValueError("question must contain valid UTF-8 text") from None
        return self
