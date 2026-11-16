"""Frozen evaluation contracts: task-visible documents and evaluator-only gold."""

from __future__ import annotations

from typing import Annotated, Literal, Self
from urllib.parse import urlsplit

from pydantic import Field, StringConstraints, field_validator, model_validator

from deep_research.schemas import StrictModel

MAX_DOCUMENT_BYTES = 64 * 1024
BenchmarkID = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,95}$")]
Title = Annotated[str, StringConstraints(min_length=1, max_length=300)]
Text = Annotated[str, StringConstraints(min_length=1, max_length=1000)]


def _readable(value: str) -> str:
    if not value.strip() or any(ord(c) < 32 and c not in "\n\r\t" for c in value):
        raise ValueError("text must be nonblank and contain no unsafe control characters")
    # Reject unpaired surrogates before later UTF-8 serialization.
    value.encode("utf-8")
    return value


class BenchmarkDocument(StrictModel):
    id: BenchmarkID
    title: Title
    text: str = Field(min_length=1, max_length=MAX_DOCUMENT_BYTES)
    url: str = Field(min_length=1, max_length=2000)
    sentences: tuple[str, ...] | None = Field(default=None, min_length=1, max_length=1000)

    @field_validator("title", "text")
    @classmethod
    def readable_text(cls, value: str) -> str:
        return _readable(value)

    @field_validator("url")
    @classmethod
    def safe_url(cls, value: str) -> str:
        try:
            parsed = urlsplit(value)
            if (
                parsed.scheme not in {"http", "https"}
                or not parsed.hostname
                or parsed.username is not None
                or parsed.password is not None
                or "\\" in value
                or any(ord(char) <= 32 or ord(char) == 127 for char in value)
            ):
                raise ValueError("invalid document URL")
            _ = parsed.port
            value.encode("utf-8")
        except (ValueError, UnicodeError):
            raise ValueError("invalid document URL") from None
        return value

    @model_validator(mode="after")
    def bounded_document(self) -> Self:
        if len(self.text.encode("utf-8")) > MAX_DOCUMENT_BYTES:
            raise ValueError("document exceeds the 64 KiB UTF-8 limit")
        if self.sentences is not None:
            for sentence in self.sentences:
                if len(sentence.encode("utf-8")) > MAX_DOCUMENT_BYTES:
                    raise ValueError("sentence exceeds the document byte limit")
                if any(ord(c) < 32 and c not in "\n\r\t" for c in sentence):
                    raise ValueError("invalid sentence text")
            if self.text != "\n".join(self.sentences):
                raise ValueError("document text must preserve the original sentence sequence")
        return self


class GoldFact(StrictModel):
    id: BenchmarkID
    statement: Text
    aliases: tuple[Text, ...] = Field(min_length=1, max_length=8)
    source_ids: tuple[BenchmarkID, ...] = Field(min_length=1, max_length=10)

    @field_validator("statement")
    @classmethod
    def readable_statement(cls, value: str) -> str:
        return _readable(value)

    @model_validator(mode="after")
    def unique_values(self) -> Self:
        for alias in self.aliases:
            _readable(alias)
        if len(set(self.aliases)) != len(self.aliases):
            raise ValueError("duplicate fact aliases")
        if len(set(self.source_ids)) != len(self.source_ids):
            raise ValueError("duplicate gold source IDs")
        return self


class SupportingFact(StrictModel):
    title: Title
    sentence_index: int = Field(ge=0, le=999)

    @field_validator("title")
    @classmethod
    def readable_title(cls, value: str) -> str:
        return _readable(value)


class BenchmarkGold(StrictModel):
    answer_aliases: tuple[Text, ...] = Field(default=(), max_length=8)
    facts: tuple[GoldFact, ...] = Field(default=(), max_length=12)
    supporting_facts: tuple[SupportingFact, ...] = Field(default=(), max_length=100)

    @model_validator(mode="after")
    def unique_gold(self) -> Self:
        for alias in self.answer_aliases:
            _readable(alias)
        if len(set(self.answer_aliases)) != len(self.answer_aliases):
            raise ValueError("duplicate answer aliases")
        if len({fact.id for fact in self.facts}) != len(self.facts):
            raise ValueError("duplicate fact IDs")
        references = {(fact.title, fact.sentence_index) for fact in self.supporting_facts}
        if len(references) != len(self.supporting_facts):
            raise ValueError("duplicate supporting facts")
        if not (self.answer_aliases or self.facts or self.supporting_facts):
            raise ValueError("evaluation gold must not be empty")
        return self


class BenchmarkCase(StrictModel):
    id: BenchmarkID
    domain: Annotated[str, StringConstraints(pattern=r"^[a-z][a-z0-9_-]{0,47}$")]
    question: Text
    kind: Literal["research", "hotpotqa"]
    documents: tuple[BenchmarkDocument, ...] = Field(min_length=2, max_length=10)
    gold: BenchmarkGold

    @field_validator("question")
    @classmethod
    def readable_question(cls, value: str) -> str:
        return _readable(value)

    @model_validator(mode="after")
    def linked_gold(self) -> Self:
        for attribute in ("id", "title", "url"):
            if len({getattr(doc, attribute) for doc in self.documents}) != len(self.documents):
                raise ValueError(f"duplicate document {attribute}")
        ids = {doc.id for doc in self.documents}
        documents_by_title = {doc.title: doc for doc in self.documents}
        if any(not set(fact.source_ids) <= ids for fact in self.gold.facts):
            raise ValueError("gold fact references an unknown document")
        for fact in self.gold.supporting_facts:
            doc = documents_by_title.get(fact.title)
            if doc is None or doc.sentences is None or fact.sentence_index >= len(doc.sentences):
                raise ValueError("supporting fact references an unknown original sentence")
        if self.kind == "research" and not self.gold.facts:
            raise ValueError("research cases require explicit gold facts")
        if self.kind == "hotpotqa" and (
            not self.gold.answer_aliases or not self.gold.supporting_facts
        ):
            raise ValueError("HotpotQA cases require an answer and supporting facts")
        return self


class BenchmarkDataset(StrictModel):
    version: Literal[1] = 1
    id: BenchmarkID
    description: str = Field(min_length=1, max_length=2000)
    provenance: str = Field(min_length=1, max_length=2000)
    cases: tuple[BenchmarkCase, ...] = Field(min_length=1, max_length=100)
    source_sha256: Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")] | None = None
    split: Annotated[str, StringConstraints(min_length=1, max_length=100)] | None = None

    @field_validator("version", mode="before")
    @classmethod
    def strict_version(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("dataset version must be an integer")
        return value

    @field_validator("description", "provenance", "split")
    @classmethod
    def readable_metadata(cls, value: str | None) -> str | None:
        return None if value is None else _readable(value)

    @model_validator(mode="after")
    def unique_case_ids(self) -> Self:
        if len({case.id for case in self.cases}) != len(self.cases):
            raise ValueError("duplicate benchmark case IDs")
        return self
