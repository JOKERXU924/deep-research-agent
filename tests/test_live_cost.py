import asyncio
from types import SimpleNamespace

import pytest

from deep_research.errors import ResearchError
from deep_research.live_cost import CurrencyLedger, MeteredProvider
from deep_research.schemas import ChatMessage, ModelRequest, TokenUsage


def request():
    return ModelRequest(
        messages=(ChatMessage(role="user", content="Public test."),), max_output_tokens=100
    )


def test_persistent_cap_and_known_usage_accounting(tmp_path):
    path = tmp_path / "money.sqlite"
    ledger = CurrencyLedger(path, 20000)
    i = ledger.reserve(request(), "fixture")
    ledger.finish(i, TokenUsage(input_tokens=100, output_tokens=10, total_tokens=110))
    assert ledger.summary()["accounted_upper_cny"] == 0.00028
    assert CurrencyLedger(path, 20000).summary()["calls"] == 1
    with pytest.raises(ValueError):
        CurrencyLedger(path, 30000)


def test_unknown_and_unfinished_requests_keep_full_reservation(tmp_path):
    ledger = CurrencyLedger(tmp_path / "money.sqlite", 20000)
    i = ledger.reserve(request(), "fixture")
    ledger.finish(i, None, "timeout")
    row = ledger.summary()["attempts"][0]
    assert row["charged_micro"] == row["reserved_micro"]
    with pytest.raises(Exception, match="预算"):
        ledger.reserve(request(), "blocked")
    assert ledger.summary()["calls"] == 1


async def test_refuses_dispatch_before_reservation_and_drains_cancellation(tmp_path):
    class Provider:
        settings = SimpleNamespace(model="deepseek-flash")
        called = False

        async def complete(self, request):
            self.called = True
            raise asyncio.CancelledError

    provider = Provider()
    ledger = CurrencyLedger(tmp_path / "money.sqlite", 100)
    with pytest.raises(Exception, match="预算"):
        await MeteredProvider(provider, ledger, "test").complete(request())
    assert not provider.called
    ledger = CurrencyLedger(tmp_path / "other.sqlite")
    with pytest.raises(asyncio.CancelledError):
        await MeteredProvider(provider, ledger, "test").complete(request())
    assert ledger.summary()["unknown_calls"] == 1


def test_unknown_model_and_over_cap_rejected(tmp_path):
    with pytest.raises(ValueError):
        CurrencyLedger(tmp_path / "money.sqlite", 10_000_001)
    ledger = CurrencyLedger(tmp_path / "money.sqlite")
    with pytest.raises(ResearchError):
        MeteredProvider(SimpleNamespace(settings=SimpleNamespace(model="other")), ledger, "test")


def test_explicit_extension_keeps_history_and_blocks_stale_writers(tmp_path):
    path = tmp_path / "money.sqlite"
    ledger = CurrencyLedger(path)
    stale = CurrencyLedger(path)
    i = ledger.reserve(request(), "before")
    ledger.finish(i, TokenUsage(input_tokens=100, output_tokens=10, total_tokens=110))
    before = ledger.summary()["attempts"]
    ledger.extend_cap(
        10_000_000, expected_cap_micro=5_000_000, authorization="User raised total cap"
    )
    assert ledger.summary()["attempts"] == before
    assert CurrencyLedger(path).summary()["cap_cny"] == 10
    with pytest.raises(ValueError):
        CurrencyLedger(path, 5_000_000)
    with pytest.raises(ResearchError, match="上限已变更"):
        stale.reserve(request(), "stale")
    with pytest.raises(ValueError):
        ledger.extend_cap(10_000_000, expected_cap_micro=5_000_000, authorization="Duplicate")
    assert ledger.summary()["attempts"] == before
