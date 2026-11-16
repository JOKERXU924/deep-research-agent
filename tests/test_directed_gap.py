import json

import pytest
from test_dag_research import FakeRetriever
from test_research_map import bundle as map_bundle
from test_retrieval_quality_pipeline import GapProvider
from test_review_research import storage as review_storage

from deep_research.dag_research import DagOptions, run_dag_pipeline
from deep_research.directed_gap import plan_directed_gaps
from deep_research.gateway import ModelGateway
from deep_research.retrieval import SourceDocument
from deep_research.retrieval_quality import RetrievalQualityOptions
from deep_research.review_models import ReviewOptions
from deep_research.schemas import ModelResponse

bundle = map_bundle
storage = review_storage


def test_prioritizes_missing_requirements_and_binds_snapshot(bundle):
    plan = plan_directed_gaps(bundle, maximum=1)
    assert plan["targets"][0]["point_id"] == "P2"
    assert plan["targets"][0]["reason"] == "no_conclusion"
    assert len(plan["base_report_hash"]) == 64


def test_query_with_url_is_not_dispatched(bundle):
    bundle["question"] = "https://private.invalid/secret"
    assert plan_directed_gaps(bundle)["targets"] == []


def test_output_requirements_never_become_external_search_queries(bundle):
    bundle["question"] = "部署成本是多少？每点分别回答并引用证据。"
    targets = plan_directed_gaps(bundle)["targets"]
    assert len(targets) == 1 and targets[0]["point_id"] == "P1"


@pytest.mark.parametrize("duplicate_first_blue", [False, True])
async def test_targeted_gap_appends_evidence_and_rewrites_once_before_review(
    tmp_path, storage, duplicate_first_blue
):
    question = "光合作用提供什么？部署成本是多少？"
    original = SourceDocument("https://example.org/light", "光合作用", "光合作用提供能量。")
    extra = SourceDocument(
        "https://example.org/cost", "部署成本", "部署成本取决于机器规格和运行时间。"
    )

    class Provider(GapProvider):
        async def complete(self, request):
            context = json.loads(request.messages[-1].content)
            if "targets" in context:
                self.calls.append(("directed.plan", request))
                return ModelResponse(
                    provider="fixture",
                    model="fixture",
                    request_id="fixture",
                    content=json.dumps(
                        dict(
                            targets=[
                                dict(point_id=t["point_id"], query="部署成本是多少")
                                for t in context["targets"]
                            ]
                        ),
                        ensure_ascii=False,
                    ),
                    latency_ms=0.0,
                )
            response = await super().complete(request)
            if duplicate_first_blue and "red_review" in context and self.blue_calls == 1:
                data = json.loads(response.content)
                target = data["operations"][0]["target_id"]
                data["operations"].append(
                    dict(
                        op="VERIFY",
                        target_id=target,
                        verdict="supported",
                        evidence_ids=[context["evidence"][-1]["evidence_id"]],
                        rationale="Fixture duplicate target",
                    )
                )
                return response.model_copy(update={"content": json.dumps(data, ensure_ascii=False)})
            return response

    provider = Provider(scenario="repair" if duplicate_first_blue else "passed")
    result = await run_dag_pipeline(
        question,
        gateway=ModelGateway(provider),
        retriever=FakeRetriever({"initial": [original], "部署成本是多少": [extra]}),
        store=storage[0],
        journal=storage[1],
        review_journal=storage[2],
        output_dir=tmp_path / "reports",
        options=DagOptions(max_replans=0),
        quality_options=RetrievalQualityOptions(evidence_selection="coverage", directed_gaps=True),
        review_options=ReviewOptions(max_rounds=1),
        source_mode="local_corpus",
    )
    assert result["gap_retrieval"]["directed"]["revised"] is True
    assert result["gap_retrieval"]["added_evidence_ids"]
    assert [stage for stage, _ in provider.calls][:5] == [
        "plan",
        "report",
        "directed.plan",
        "report",
        "red",
    ]
    assert len(provider.calls) == (8 if duplicate_first_blue else 5)
    assert result["model_call_count"] <= 9
