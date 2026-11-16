"""M5 parser safety and explicit per-run audit budgets; no network."""

import json
import sqlite3

import pytest

from deep_research.dag_journal import DagJournal, DagJournalError
from deep_research.errors import StructuredOutputError
from deep_research.research_store import ResearchStore
from deep_research.review_json import parse_review_output
from deep_research.schemas import MAX_OUTPUT_BYTES, StrictModel
from deep_research.structured import parse_structured


class Answer(StrictModel):
    text: str
    count: int


VALID = '{"text":"中文 {literal} and escaped \\" quote", "count":2}'


@pytest.mark.parametrize("text", [VALID, f"```json\n{VALID}\n```", f" \n{VALID}\n"])
def test_strict_and_fenced_modes(text):
    data, mode = parse_review_output(text, Answer)
    assert mode == "strict"
    assert data.count == 2
    assert "{literal}" in data.text


@pytest.mark.parametrize(
    "wrapper", ["Here is the result: {} Done.", "说明\n{}\n结束", "x```json\n{}\n```y"]
)
def test_exactly_one_object_recovery(wrapper):
    text = wrapper.format(VALID)
    data, mode = parse_review_output(text, Answer)
    assert mode == "envelope" and data.count == 2
    with pytest.raises(StructuredOutputError):
        parse_structured(text, Answer)  # The M0 parser is intentionally unchanged.


@pytest.mark.parametrize(
    "text",
    [
        "",
        "plain text",
        '{"text":"unfinished',
        '{"text":"x","count":2',
        VALID + VALID,
        "a" + VALID + "b" + VALID,
        "}" + VALID,
        "{" + VALID,
        VALID + "}",
        "prefix [" + VALID + "] suffix",
        "prefix [] " + VALID,
        "a" * 1025 + VALID,
        "中" * 342 + VALID,
        'note {"text":"x","count":1,"count":2}',
        'note {"text":"x","count":NaN}',
        'note {"text":"x","count":Infinity}',
        'note {"text":"x","count":1e999}',
        'note {"text":"x","count":2,}',
        'note {text:"x","count":2}',
        'note {"text":"x",/*comment*/"count":2}',
        "\ud800",
        '"unterminated ' + VALID,
    ],
)
def test_recovery_does_not_guess_ambiguous_or_invalid_json(text):
    with pytest.raises(StructuredOutputError) as caught:
        parse_review_output(text, Answer)
    assert caught.value.code == "invalid_json"


@pytest.mark.parametrize(
    "text",
    [
        'note {"text":"x","count":"2"}',
        'note {"text":"x","count":2,"extra":true}',
        '{"text":"x"}',
        "[]",
        "null",
        "true",
        "[" + VALID + "]",
    ],
)
def test_envelope_never_bypasses_schema(text):
    with pytest.raises(StructuredOutputError) as caught:
        parse_review_output(text, Answer)
    assert caught.value.code == "schema_mismatch"


def test_exact_utf8_wrapper_budget():
    result, mode = parse_review_output("中" * 341 + "x" + VALID, Answer)
    assert mode == "envelope" and result.count == 2


def test_output_byte_budget_precedes_extraction():
    with pytest.raises(StructuredOutputError) as caught:
        parse_review_output("中" * (MAX_OUTPUT_BYTES // 3 + 1) + VALID, Answer)
    assert caught.value.code == "output_too_large"


@pytest.mark.parametrize("limit", [3, 4, 5, 6, 7, 8, 9])
def test_persisted_explicit_call_budget_survives_reopen(tmp_path, limit):
    path = tmp_path / "research.sqlite3"
    with ResearchStore(path) as store, DagJournal(path) as journal:
        run_id = store.create_run("budget example")
        journal.create_run(run_id, {"model_call_limit": limit})
        for index in range(limit):
            call_id = journal.start_model_call(run_id, f"review.red.{index}")
            journal.finish_model_call(run_id, call_id, "completed", {})
    with DagJournal(path) as journal:
        assert len(journal.snapshot(run_id)["model_calls"]) == limit
        with pytest.raises(DagJournalError):
            journal.start_model_call(run_id, "overflow")
        with pytest.raises(DagJournalError):
            journal.finish_model_call(run_id, limit, "failed", {})


@pytest.mark.parametrize("limit", [None, True, False, 0, 2, 10, 100, "9", 9.0, [], {}])
def test_invalid_budget_rejected_before_dag_creation(tmp_path, limit):
    path = tmp_path / "research.sqlite3"
    with ResearchStore(path) as store, DagJournal(path) as journal:
        run_id = store.create_run("budget example")
        with pytest.raises(DagJournalError):
            journal.create_run(run_id, {"model_call_limit": limit})
        assert journal.snapshot(run_id) is None


def test_original_three_call_budget_remains_default(tmp_path):
    path = tmp_path / "research.sqlite3"
    with ResearchStore(path) as store, DagJournal(path) as journal:
        run_id = store.create_run("old run")
        journal.create_run(run_id, {})
        for _ in range(3):
            journal.start_model_call(run_id, "old.stage")
        with pytest.raises(DagJournalError):
            journal.start_model_call(run_id, "fourth")
        with pytest.raises(DagJournalError):
            journal.finish_model_call(run_id, 4, "completed", {})


@pytest.mark.parametrize("config", ["{", "[]", "null", json.dumps({"model_call_limit": 99})])
def test_corrupted_budget_is_not_used_to_expand_calls(tmp_path, config):
    path = tmp_path / "research.sqlite3"
    with ResearchStore(path) as store, DagJournal(path) as journal:
        run_id = store.create_run("corruption example")
        journal.create_run(run_id, {})
        call_id = journal.start_model_call(run_id, "plan")
        with sqlite3.connect(path) as connection:
            connection.execute(
                "UPDATE dag_runs SET config_json = ? WHERE run_id = ?", (config, run_id)
            )
        with pytest.raises(DagJournalError):
            journal.start_model_call(run_id, "bad")
        with pytest.raises(DagJournalError):
            journal.finish_model_call(run_id, call_id, "completed", {})
