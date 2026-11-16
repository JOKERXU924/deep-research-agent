import hashlib

from deep_research.evidence_relations import link, obligations
from deep_research.evidence_selection_v3 import prepare


def source(text):
    return dict(source_id="S1", text=text, sha256=hashlib.sha256(text.encode()).hexdigest())


async def test_api_reference_window_keeps_source_text_and_stops_at_next_class():
    text = (
        "package.snapshot()\nReturns a snapshot.\nAn example follows:\n"
        "The function has constant complexity.\nclass package.Other\nThe function fails."
    )
    linked = link(await prepare([source(text)], "snapshot complexity?"))
    window = next(
        c
        for c in linked.windows
        if "snapshot()" in c["quote"] and "constant complexity" in c["quote"]
    )
    assert window["quote"] == text[window["start"] : window["end"]]
    assert "Other" not in window["quote"] and len(window["quote"]) <= 600
    assert all(c["end"] <= text.index("class package.Other") for c in linked.windows)


async def test_condition_window_omits_long_irrelevant_leadin_and_keeps_following_predicate():
    text = (
        "package.search(items, value)\n"
        + "Background information about examples. " * 5
        + "If value is present, choose the left boundary. "
        "The return value is a valid position.\n"
        "The returned partition has left < value and right >= value."
    )
    linked = link(await prepare([source(text)], "search boundary?"))
    assert any(
        c["quote"].startswith("If value") and "right >= value" in c["quote"] for c in linked.windows
    )
    assert all(c["quote"] == text[c["start"] : c["end"]] for c in linked.windows)


async def test_plain_prose_does_not_create_api_windows():
    text = "An unrelated heading\nThe function has constant complexity."
    ranked = await prepare([source(text)], "complexity?")
    linked = link(ranked)
    assert linked.windows == [] and linked.audit["windows"] == []


async def test_only_explicit_api_constraint_requests_receive_dependency_candidates():
    text = "package.snapshot()\nReturns a snapshot.\nThe function has constant complexity."
    linked = link(await prepare([source(text)], "snapshot?"))
    result = obligations(linked, [dict(id="P1", question="snapshot 的复杂度？")], [])
    assert len(result.evidence) == 1 and "constant complexity" in result.evidence[0]["quote"]
    assert obligations(linked, [dict(id="P1", question="snapshot 的用途？")], []).evidence == []
    assert obligations(linked, [dict(id="P1", question="other 的复杂度？")], []).evidence == []


def test_overlong_union_can_append_only_missing_source_tail_without_growing_existing_span():
    from deep_research.compression_runtime import json_bytes
    from deep_research.evidence_selection import Selection
    from deep_research.rewrite_selection import protect

    text = "a" * 350 + "b" * 50 + "c" * 300
    original = [dict(evidence_id="E1", source_id="S1", start=0, end=400, quote=text[:400])]
    extra = [dict(evidence_id="E1", source_id="S1", start=350, end=700, quote=text[350:])]
    old = Selection(original, dict(context_bytes=json_bytes(original)))
    supplement = Selection(extra, dict(context_bytes=json_bytes(extra)))
    sources = [source(text)]
    declined = protect(old, supplement, sources, max_bytes=12000, max_items=2)
    assert declined.evidence == original  # frozen v2 defaults remain unchanged
    result = protect(old, supplement, sources, max_bytes=12000, max_items=2, allow_split=True)
    assert result.evidence[0] == original[0]
    assert result.evidence[1]["start"] == 400 and result.evidence[1]["end"] == 700
    assert result.evidence[1]["quote"] == text[400:700]
    denied = protect(
        old, supplement, sources, max_bytes=json_bytes(original), max_items=2, allow_split=True
    )
    assert denied.evidence == original
