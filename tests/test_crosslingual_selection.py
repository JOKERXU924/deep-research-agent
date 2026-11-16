import hashlib

import numpy as np
import pytest

from deep_research.compression_runtime import json_bytes
from deep_research.crosslingual_selection import select_bilingual
from deep_research.errors import ResearchError


class Encoder:
    dimensions = 2
    encoder_id = "fixture"

    async def encode_queries(self, texts):
        return np.array(
            [[1, 0] if "读" in t or "Alpha" in t else [0, 1] for t in texts], dtype=float
        )

    async def encode_documents(self, texts):
        return np.array([[1, 0] if "Alpha" in t else [0, 1] for t in texts], dtype=float)


def inputs():
    text = "Existing stable fact.\nAlpha supports reads.\nBeta supports writes."
    source = dict(source_id="S1", text=text, sha256=hashlib.sha256(text.encode()).hexdigest())
    existing = [dict(evidence_id="E1", source_id="S1", start=0, end=21, quote=text[:21])]
    return [source], existing


async def test_bilingual_selection_preserves_existing_context_offsets_and_budget():
    sources, existing = inputs()
    before = [dict(e) for e in existing]
    result = await select_bilingual(
        sources,
        "读服务是什么？写服务是什么？",
        rewrites=[
            dict(point_id="P1", query="Alpha reads"),
            dict(point_id="P2", query="Beta writes"),
        ],
        encoder=Encoder(),
        existing=existing,
        next_id=2,
        max_bytes=800,
        max_items=3,
    )
    assert existing == before and json_bytes([*existing, *result.evidence]) <= 800
    assert len(result.evidence) + len(existing) <= 3
    assert all(e["quote"] == sources[0]["text"][e["start"] : e["end"]] for e in result.evidence)
    assert "Alpha supports reads." in " ".join(e["quote"] for e in result.evidence)
    assert "Beta supports writes." in " ".join(e["quote"] for e in result.evidence)


async def test_existing_over_budget_is_rejected_before_encoding():
    sources, existing = inputs()
    with pytest.raises(ResearchError):
        await select_bilingual(
            sources, "读服务？", rewrites=[], encoder=Encoder(), existing=existing, max_bytes=2
        )


async def test_invalid_embedding_is_not_silently_used():
    class Bad(Encoder):
        async def encode_queries(self, texts):
            return np.full((len(texts), 2), np.nan)

    sources, _ = inputs()
    with pytest.raises(ResearchError):
        await select_bilingual(sources, "读服务？", rewrites=[], encoder=Bad())
