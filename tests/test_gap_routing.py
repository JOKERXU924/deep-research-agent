from deep_research.gap_routing import plan_gap_actions, reported_conflict_ids


def data():
    publication = dict(
        points=[
            dict(
                id="P1",
                question="Reader concurrency?",
                status="partial",
                claim_ids=[],
                gap_query="reader concurrency",
            )
        ]
    )
    support = dict(
        claims=[dict(claim_id="C1", support="insufficient")],
        points=[dict(point_id="P1", claim_ids=["C1"])],
    )
    sources = [dict(source_id="S1", text="Reader concurrency is supported.")]
    return publication, support, sources


def test_omitted_context_and_absent_source_get_different_routes():
    p, s, docs = data()
    local = plan_gap_actions(p, s, docs, [])["targets"][0]
    remote = plan_gap_actions(p, s, [], [])["targets"][0]
    assert local["action"] == "restore_local" and local["closure"] == "unverified"
    assert remote["action"] == "search_new"


def test_claim_contradiction_is_not_mislabeled_as_conflicting_sources():
    p, s, docs = data()
    s["claims"][0]["support"] = "contradicted"
    assert plan_gap_actions(p, s, docs, [])["targets"][0]["action"] == "revise_existing"
    assert (
        plan_gap_actions(p, s, docs, [], conflict_claims=["C1"])["targets"][0]["action"]
        == "search_alternative"
    )


def test_untrusted_urls_cannot_be_turned_into_search_queries():
    p, s, docs = data()
    p["points"][0]["gap_query"] = "https://example.org/"
    assert plan_gap_actions(p, s, docs, [])["targets"] == []


def test_only_selected_matching_review_snapshot_can_route_a_source_conflict():
    review = dict(
        scope_review=dict(
            selected_round=0,
            snapshots=[
                dict(round=1, report_hash="new", source_conflicts=[dict(claim_id="C2")]),
                dict(round=0, report_hash="chosen", source_conflicts=[dict(claim_id="C1")]),
            ],
        )
    )
    assert reported_conflict_ids(review, "chosen") == {"C1"}
    assert reported_conflict_ids(review, "new") == set()
