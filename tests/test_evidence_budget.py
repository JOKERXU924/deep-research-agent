import hashlib

import numpy as np

from deep_research.compression_runtime import json_bytes
from deep_research.evidence_budget import allocate_marginal
from deep_research.evidence_selection import Selection


def test_compaction_and_marginal_fill_preserve_all_original_spans_and_real_budget():
    text = "Alpha supports reads. Beta supports writes. Gamma supports audit."
    source = dict(source_id="S1", text=text, sha256=hashlib.sha256(text.encode()).hexdigest())
    chunks = []
    for i, quote in enumerate(
        ["Alpha supports reads.", "Beta supports writes.", "Gamma supports audit."], 1
    ):
        start = text.index(quote)
        chunks.append(
            dict(
                evidence_id=f"E{i}",
                source_id="S1",
                start=start,
                end=start + len(quote),
                quote=quote,
            )
        )
    base = Selection(chunks[:2], dict(context_bytes=json_bytes(chunks[:2])))
    result = allocate_marginal(
        base, chunks, np.array([[1, 0.1, 0.8], [0.1, 1, 0.9]]), [source], max_bytes=4000
    )
    assert json_bytes(result.evidence) <= json_bytes(base.evidence)
    assert all(
        any(e["start"] <= c["start"] and e["end"] >= c["end"] for e in result.evidence)
        for c in base.evidence
    )
    assert "Gamma supports audit." in " ".join(e["quote"] for e in result.evidence)
    assert result.audit["adaptive_budget"]["adjacent_merges"] == 1
    assert len(result.audit["selected"]) == len(result.evidence)


def test_compaction_cannot_bridge_unselected_text_or_change_existing_context():
    text = "A. THIS TEXT WAS NOT SELECTED. B."
    chunks = [
        dict(evidence_id="E2", source_id="S1", start=0, end=2, quote="A."),
        dict(evidence_id="E3", source_id="S1", start=len(text) - 2, end=len(text), quote="B."),
    ]
    existing = [dict(evidence_id="E1", source_id="S0", start=0, end=1, quote="X")]
    before = [dict(e) for e in existing]
    result = allocate_marginal(
        Selection(chunks, {}),
        chunks,
        np.ones((1, 2)),
        [dict(source_id="S1", text=text)],
        max_bytes=4000,
        existing=existing,
        next_id=2,
    )
    assert existing == before and result.evidence == chunks
    assert result.audit["adaptive_budget"]["adjacent_merges"] == 0
