"""M1 contracts: plans, evidence-linked claims, and bounded research output."""

from typing import Annotated, Literal, Self

from pydantic import Field, StringConstraints, model_validator

from deep_research.schemas import StrictModel

QuestionText = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=1000)
]
QueryText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=120)]
ShortText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=200)]
ClaimText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=700)]
EvidenceID = Annotated[str, StringConstraints(pattern=r"^E[1-9][0-9]{0,2}$")]


class ResearchInput(StrictModel):
    question: QuestionText
    language: Literal["zh", "en"] = "zh"


class ResearchPlan(StrictModel):
    queries: list[QueryText] = Field(min_length=1, max_length=3)

    @model_validator(mode="after")
    def unique_queries(self) -> Self:
        if len({query.casefold() for query in self.queries}) != len(self.queries):
            raise ValueError("search queries must be distinct")
        return self


class CitedClaim(StrictModel):
    text: ClaimText
    citations: list[EvidenceID] = Field(min_length=1, max_length=4)

    @model_validator(mode="after")
    def unique_citations(self) -> Self:
        if len(set(self.citations)) != len(self.citations):
            raise ValueError("duplicate evidence citation")
        return self


class ReportSection(StrictModel):
    heading: ShortText
    claims: list[CitedClaim] = Field(min_length=1, max_length=6)


class ResearchReport(StrictModel):
    title: ShortText
    sections: list[ReportSection] = Field(min_length=1, max_length=4)
    limitations: list[ClaimText] = Field(min_length=1, max_length=6)

    @model_validator(mode="after")
    def total_claim_limit(self) -> Self:
        if sum(len(section.claims) for section in self.sections) > 12:
            raise ValueError("a report may contain at most twelve claims")
        return self
