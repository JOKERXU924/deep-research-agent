"""The production route retains foreign-language sources before context selection."""

import json

from test_dag_research import FakeRetriever
from test_evidence_selection_v2 import Encoder
from test_retrieval_quality_pipeline import GapProvider, events, execute
from test_review_research import storage as review_storage

from deep_research.retrieval import SourceDocument
from deep_research.retrieval_quality import RetrievalQualityOptions

storage = review_storage


async def test_cross_language_filter_rescue_is_audited_and_reaches_report(tmp_path, storage):
    provider = GapProvider(scenario="passed")
    document = SourceDocument(
        "https://example.org/light", "Light reaction", "Light supplies energy."
    )
    result = await execute(
        tmp_path,
        storage,
        provider=provider,
        retriever=FakeRetriever({"initial": [document]}),
        quality_options=RetrievalQualityOptions(evidence_selection="multilingual"),
        evidence_encoder=Encoder(),
    )
    bundle = storage[0].get_report(result["run_id"])
    assert bundle["sources"][0]["text"] == document.text
    assert bundle["retrieval_quality"]["selection"]["version"] == "multilingual-parent-v2"
    assert any(
        e.get("reason") == "semantic_cross_language" for e in events(storage, result["run_id"])
    )
    evidence = json.loads(provider.calls[1][1].messages[-1].content)["evidence"]
    assert evidence[0]["quote"] == document.text
