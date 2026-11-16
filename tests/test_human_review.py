import copy
import json

import pytest
from test_review import inputs, report

from deep_research.human_review import build_packet, score_packet


def packet(tmp_path):
    sources, evidence = inputs()
    out = tmp_path / "packet"
    build_packet(
        [
            dict(
                question="What is known?",
                variant="hidden",
                bundle=dict(report=report().model_dump(), sources=sources, evidence=evidence),
            )
        ],
        out,
    )
    return json.loads((out / "packet.json").read_text(encoding="utf-8")), json.loads(
        (out / "ratings.json").read_text(encoding="utf-8")
    )


def test_pending_human_work_never_becomes_an_accuracy_claim(tmp_path):
    p, r = packet(tmp_path)
    s = score_packet(p, r)
    assert s["supported_fact_pct"] is None and s["fully_answered_point_pct"] is None
    assert s["status"] == "pending_independent_human_review"


def test_only_complete_attested_labels_are_scored(tmp_path):
    p, r = packet(tmp_path)
    r["reviewer"] = "independent reviewer"
    r["independent_human_review"] = True
    r["cases"][0]["claims"][0].update(label="supported", rationale="E1 exact source")
    r["cases"][0]["requirements"][0].update(label="answered", rationale="C1 answers P1")
    assert score_packet(p, r)["supported_fact_pct"] == 100
    r["independent_human_review"] = False
    assert score_packet(p, r)["supported_fact_pct"] is None


def test_unknown_or_dropped_claims_and_changed_snapshots_are_rejected(tmp_path):
    p, r = packet(tmp_path)
    bad = copy.deepcopy(r)
    bad["cases"][0]["claims"] = []
    with pytest.raises(ValueError):
        score_packet(p, bad)
    p["cases"][0]["question"] = "edited question"
    with pytest.raises(ValueError):
        score_packet(p, r)
