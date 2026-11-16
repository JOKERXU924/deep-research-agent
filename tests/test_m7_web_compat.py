"""M6 persisted requests keep their idempotency semantics after M7 defaults."""

import hashlib
import json
import sqlite3
from uuid import UUID, uuid4

import pytest

from deep_research import web_jobs
from deep_research.errors import ResearchError
from deep_research.web_jobs import WebJobStore
from deep_research.web_models import WebRequest


def legacy_job(store: WebJobStore, request: WebRequest) -> tuple[dict, str, str, str]:
    key = str(uuid4())
    job, created = store.submit(request, key, 8)
    assert created
    old_request = request.model_dump()
    del old_request["profile_id"]
    original = json.dumps(old_request, ensure_ascii=False, allow_nan=False, sort_keys=True)
    checksum = hashlib.sha256(original.encode("utf-8")).hexdigest()
    with sqlite3.connect(store.path) as connection:
        connection.execute(
            "UPDATE web_jobs SET request_json=?, request_hash=? WHERE id=?",
            (original, checksum, job["id"]),
        )
    return job, key, original, checksum


@pytest.mark.parametrize("live", [False, True])
def test_old_request_replays_after_reopening_without_rewriting_history(tmp_path, live):
    path = tmp_path / "jobs.sqlite3"
    request = WebRequest(mode="live", confirm_live=True) if live else WebRequest()
    with WebJobStore(path) as store:
        job, key, original, checksum = legacy_job(store, request)
        original_events = store.events(job["id"])
    with WebJobStore(path) as reopened:
        # The added optional field is populated on reads, not written into history.
        read = reopened.get(job["id"])
        assert read["request"]["profile_id"] is None
        assert reopened.list() == [read]
        replay, created = reopened.submit(request, key, 1)
        assert not created and replay == read
        assert reopened.events(job["id"]) == original_events
        with sqlite3.connect(path) as connection:
            persisted = connection.execute(
                "SELECT request_json, request_hash FROM web_jobs WHERE id=?", (job["id"],)
            ).fetchone()
        assert persisted == (original, checksum)


@pytest.mark.parametrize("profile", ["openai-local-name", "deepseek-default", "vllm"])
def test_old_same_key_with_explicit_profile_is_a_different_intent(tmp_path, profile):
    request = WebRequest(mode="live", confirm_live=True)
    with WebJobStore(tmp_path / "jobs.sqlite3") as store:
        job, key, _, _ = legacy_job(store, request)
        with pytest.raises(ResearchError) as caught:
            store.submit(request.model_copy(update={"profile_id": profile}), key, 8)
        assert caught.value.code == "web_conflict"
        assert len(store.list()) == 1
        assert len(store.events(job["id"])) == 1


def test_two_explicit_profiles_never_share_an_idempotent_request(tmp_path):
    request = WebRequest(mode="live", confirm_live=True, profile_id="openai-small")
    key = str(uuid4())
    with WebJobStore(tmp_path / "jobs.sqlite3") as store:
        original, _ = store.submit(request, key, 8)
        for changed in ("mimo-small", None):
            with pytest.raises(ResearchError) as caught:
                store.submit(request.model_copy(update={"profile_id": changed}), key, 8)
            assert caught.value.code == "web_conflict"
        assert store.submit(request, key, 8) == (original, False)


@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize("corruption", ["hash", "payload", "hash_matches_new_payload"])
def test_checksum_is_verified_before_normalizing_or_comparing(tmp_path, legacy, corruption):
    request = WebRequest()
    with WebJobStore(tmp_path / "jobs.sqlite3") as store:
        if legacy:
            job, key, original, _ = legacy_job(store, request)
        else:
            key = str(uuid4())
            job, _ = store.submit(request, key, 8)
            original = json.dumps(request.model_dump(), ensure_ascii=False, sort_keys=True)
        changed = request.model_copy(update={"review": False})
        changed_payload = json.dumps(changed.model_dump(), ensure_ascii=False, sort_keys=True)
        with sqlite3.connect(store.path) as connection:
            if corruption == "hash":
                connection.execute(
                    "UPDATE web_jobs SET request_hash=? WHERE id=?", ("0" * 64, job["id"])
                )
            elif corruption == "payload":
                connection.execute(
                    "UPDATE web_jobs SET request_json=? WHERE id=?",
                    (original + " ", job["id"]),
                )
            else:
                # A shortcut checking only incoming hash would wrongly accept it.
                connection.execute(
                    "UPDATE web_jobs SET request_hash=? WHERE id=?",
                    (hashlib.sha256(changed_payload.encode()).hexdigest(), job["id"]),
                )
        with pytest.raises(ResearchError) as caught:
            store.submit(changed if corruption == "hash_matches_new_payload" else request, key, 8)
        assert caught.value.code == "web_io"


def test_old_request_other_changed_fields_still_conflict(tmp_path):
    with WebJobStore(tmp_path / "jobs.sqlite3") as store:
        _, key, _, _ = legacy_job(store, WebRequest())
        with pytest.raises(ResearchError) as caught:
            store.submit(WebRequest(compression=True), key, 8)
        assert caught.value.code == "web_conflict"


@pytest.mark.parametrize("raw", ["not-json", '{"mode":"live"}', '{"secret":"no"}'])
def test_valid_checksum_does_not_bypass_request_schema(tmp_path, raw):
    with WebJobStore(tmp_path / "jobs.sqlite3") as store:
        job, key, _, _ = legacy_job(store, WebRequest())
        with sqlite3.connect(store.path) as connection:
            connection.execute(
                "UPDATE web_jobs SET request_json=?, request_hash=? WHERE id=?",
                (raw, hashlib.sha256(raw.encode()).hexdigest(), job["id"]),
            )
        with pytest.raises(ResearchError) as caught:
            store.submit(WebRequest(), key, 8)
        assert caught.value.code == "web_io"


def test_same_timestamp_jobs_keep_submission_order_not_random_uuid_order(tmp_path, monkeypatch):
    monkeypatch.setattr(web_jobs, "_now", lambda: "2026-10-16T00:00:00+00:00")
    ids = iter(
        [UUID("ffffffff-ffff-4fff-8fff-ffffffffffff"), UUID("00000000-0000-4000-8000-000000000001")]
    )
    monkeypatch.setattr(web_jobs, "uuid4", lambda: next(ids))
    with WebJobStore(tmp_path / "jobs.sqlite3") as store:
        first, _ = store.submit(WebRequest(), str(uuid4()), 8)
        second, _ = store.submit(WebRequest(), str(uuid4()), 8)
        assert first["created_at"] == second["created_at"]
        assert first["id"] > second["id"]
        assert store.begin_next()["id"] == first["id"]
        assert store.begin_next()["id"] == second["id"]
