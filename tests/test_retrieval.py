import asyncio
import json
from pathlib import Path

import httpx
import pytest

from deep_research import retrieval
from deep_research.retrieval import (
    MAX_CORPUS_BYTES,
    MAX_DOCUMENT_BYTES,
    MAX_HTTP_RESPONSE_BYTES,
    LocalCorpusRetriever,
    ResearchRetrievalError,
    SearchHit,
    WikipediaRetriever,
)


def corpus_file(tmp_path: Path, documents: list[dict] | None = None) -> Path:
    path = tmp_path / "corpus.json"
    if documents is None:
        documents = [
            {
                "url": "https://example.org/storage",
                "title": "会话存储",
                "text": "会话存储保存历史消息。",
            },
            {
                "url": "https://example.org/agent",
                "title": "Agent tools",
                "text": "Tools retrieve sources.",
            },
        ]
    path.write_text(json.dumps({"documents": documents}, ensure_ascii=False), encoding="utf-8")
    return path


async def test_local_corpus_search_fetch_and_no_match(tmp_path):
    retriever = LocalCorpusRetriever.from_file(corpus_file(tmp_path))
    hits = await retriever.search("会话存储是什么")
    assert hits == [SearchHit("https://example.org/storage", "会话存储")]
    document = await retriever.fetch(hits[0])
    assert document.text == "会话存储保存历史消息。"
    assert document.truncated is False
    assert (await retriever.search("TOOLS"))[0].title == "Agent tools"
    assert await retriever.search("quantum") == []
    assert await retriever.search("!@#$") == []


async def test_local_corpus_fetch_never_accepts_arbitrary_url(tmp_path):
    retriever = LocalCorpusRetriever.from_file(corpus_file(tmp_path))
    with pytest.raises(ResearchRetrievalError, match="不在"):
        await retriever.fetch(SearchHit("https://example.net/secret", "secret"))
    with pytest.raises(ResearchRetrievalError):
        await retriever.fetch(SearchHit("https://example.org/storage", "fake title"))


@pytest.mark.parametrize(
    "url",
    [
        "http://example.org/a",
        "https://localhost/a",
        "https://127.0.0.1/a",
        "https://[::1]/a",
        "https://192.168.1.1/a",
        "https://user:password@example.org/a",
        "https://host.local/a",
        "https://host.internal/a",
        "file:///tmp/a",
        "https://example.org:8443/a",
        "https://example.org/\nsecret",
        "https://example.org\\@127.0.0.1/a",
    ],
)
def test_local_corpus_rejects_nonpublic_https_url(tmp_path, url):
    path = corpus_file(tmp_path, [{"url": url, "title": "title", "text": "body"}])
    with pytest.raises(ResearchRetrievalError) as exc:
        LocalCorpusRetriever.from_file(path)
    assert exc.value.code == "retrieval_url"
    assert "password" not in str(exc.value)


@pytest.mark.parametrize(
    "raw",
    [
        "{}",
        '{"documents":[]}',
        '{"documents":[],"documents":[]}',
        '{"documents":NaN}',
        '{"documents":[{"url":"https://example.org","title":"title","text":"body","extra":1}]}',
        '{"documents":[{"url":"https://example.org","title":2,"text":"body"}]}',
        '{"documents":[{"url":"https://example.org","title":" ","text":"body"}]}',
        '{"documents":[{"url":"https://example.org","title":"title","text":" "}]}',
        "not json",
    ],
)
def test_local_corpus_strict_json_and_shape(tmp_path, raw):
    path = tmp_path / "bad.json"
    path.write_text(raw, encoding="utf-8")
    with pytest.raises(ResearchRetrievalError):
        LocalCorpusRetriever.from_file(path)


def test_local_corpus_duplicate_normalized_urls(tmp_path):
    path = corpus_file(
        tmp_path,
        [
            {"url": "https://EXAMPLE.org", "title": "a", "text": "a"},
            {"url": "https://example.org:443/", "title": "b", "text": "b"},
        ],
    )
    with pytest.raises(ResearchRetrievalError) as exc:
        LocalCorpusRetriever.from_file(path)
    assert exc.value.code == "corpus_duplicate"


def test_local_corpus_limits(tmp_path):
    path = corpus_file(
        tmp_path, [{"url": "https://example.org/a", "title": "a", "text": "a" * MAX_DOCUMENT_BYTES}]
    )
    LocalCorpusRetriever.from_file(path)
    path = corpus_file(
        tmp_path,
        [
            {
                "url": "https://example.org/a",
                "title": "a",
                "text": "中" * (MAX_DOCUMENT_BYTES // 3 + 1),
            }
        ],
    )
    with pytest.raises(ResearchRetrievalError) as exc:
        LocalCorpusRetriever.from_file(path)
    assert exc.value.code == "corpus_size"
    path.write_bytes(b" " * (MAX_CORPUS_BYTES + 1))
    with pytest.raises(ResearchRetrievalError) as exc:
        LocalCorpusRetriever.from_file(path)
    assert exc.value.code == "corpus_size"
    path = corpus_file(
        tmp_path,
        [{"url": f"https://example.org/{index}", "title": "a", "text": "a"} for index in range(51)],
    )
    with pytest.raises(ResearchRetrievalError):
        LocalCorpusRetriever.from_file(path)


def test_local_corpus_bad_utf8_and_missing_file(tmp_path):
    with pytest.raises(ResearchRetrievalError) as exc:
        LocalCorpusRetriever.from_file(tmp_path / "private-name.json")
    assert "private-name" not in str(exc.value)
    path = tmp_path / "bad.json"
    path.write_bytes(b"\xff")
    with pytest.raises(ResearchRetrievalError):
        LocalCorpusRetriever.from_file(path)


@pytest.mark.parametrize(
    "query,limit", [("", 3), ("中" * 342, 3), ("a", 0), ("a", 11), ("a", True)]
)
async def test_retrieval_input_invalid_before_network(tmp_path, query, limit):
    def unexpected(_: httpx.Request):
        pytest.fail("invalid request reached network")

    local = LocalCorpusRetriever.from_file(corpus_file(tmp_path))
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(unexpected), trust_env=False
    ) as client:
        wikipedia = WikipediaRetriever(client=client)
        for retriever in (local, wikipedia):
            with pytest.raises(ResearchRetrievalError) as exc:
                await retriever.search(query, limit)
            assert exc.value.code == "retrieval_input"


def search_body(title="Research"):
    return {"query": {"search": [{"title": title, "pageid": 1, "ns": 0}]}}


def extract_body(text="Research is systematic inquiry.", title="Research"):
    return {"query": {"pages": [{"title": title, "pageid": 1, "ns": 0, "extract": text}]}}


async def test_wikipedia_search_fetch_fixed_endpoint_and_encoded_title():
    requests = []

    def handler(request: httpx.Request):
        requests.append(request)
        assert request.url.scheme == "https"
        assert request.url.host == "zh.wikipedia.org"
        assert request.url.path == "/w/api.php"
        assert request.url.params["formatversion"] == "2"
        assert "DeepResearchAgent" in request.headers["User-Agent"]
        if request.url.params.get("list") == "search":
            assert request.url.params["srsearch"] == "研究方法"
            assert request.url.params["srlimit"] == "2"
            assert request.url.params["srnamespace"] == "0"
            return httpx.Response(200, json=search_body("研究 方法"))
        assert request.url.params["prop"] == "extracts"
        assert request.url.params["explaintext"] == "1"
        assert request.url.params["titles"] == "研究 方法"
        return httpx.Response(200, json=extract_body("研究方法需要证据。", "研究 方法"))

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler), trust_env=False) as client:
        async with WikipediaRetriever(client=client) as retriever:
            hits = await retriever.search("研究方法", 2)
            assert (
                hits[0].url == "https://zh.wikipedia.org/wiki/%E7%A0%94%E7%A9%B6_%E6%96%B9%E6%B3%95"
            )
            doc = await retriever.fetch(hits[0])
            assert doc.text == "研究方法需要证据。"
            assert not doc.truncated
        assert not client.is_closed  # injected clients belong to the caller
        with pytest.raises(ResearchRetrievalError) as exc:
            await retriever.search("Research")
        assert exc.value.code == "retrieval_closed"
        await retriever.aclose()
    assert len(requests) == 2


async def test_wikipedia_empty_search_is_not_failure():
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda _: httpx.Response(200, json={"query": {"search": []}})
        ),
        trust_env=False,
    ) as client:
        assert await WikipediaRetriever(client=client).search("no such page") == []


async def test_wikipedia_prioritizes_topic_titles_locally_without_extra_requests():
    requests = []

    def handler(request):
        requests.append(request)
        assert request.url.params["srsearch"] == "光合作用与碳固定"
        assert request.url.params["srlimit"] == "4"
        return httpx.Response(
            200,
            json={
                "query": {
                    "search": [
                        {"title": "FAIRY TAIL角色列表", "pageid": 1, "ns": 0},
                        {"title": "碳固定", "pageid": 2, "ns": 0},
                        {"title": "光合作用", "pageid": 3, "ns": 0},
                        {"title": "无标题匹配", "pageid": 4, "ns": 0},
                    ]
                }
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler), trust_env=False) as client:
        hits = await WikipediaRetriever(client=client).search("光合作用与碳固定", 4)
    assert [hit.title for hit in hits] == ["光合作用", "碳固定", "FAIRY TAIL角色列表", "无标题匹配"]
    assert len(requests) == 1


@pytest.mark.parametrize("status", [301, 302, 401, 403, 429, 500, 503])
async def test_wikipedia_http_error_does_not_follow_redirect_or_retry(status):
    requests = []
    response = httpx.Response(
        status, headers={"Location": "http://127.0.0.1/private"}, text="secret"
    )

    def handler(request):
        requests.append(request)
        return response

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler), follow_redirects=True, trust_env=False
    ) as client:
        with pytest.raises(ResearchRetrievalError) as exc:
            await WikipediaRetriever(client=client).search("Research")
        assert exc.value.code == "retrieval_http"
        assert exc.value.status_code == status
        assert "secret" not in str(exc.value)
    assert len(requests) == 1
    assert response.is_closed


@pytest.mark.parametrize(
    "url",
    [
        "https://en.wikipedia.org/wiki/Research",
        "https://zh.wikipedia.org.evil.example/wiki/Research",
        "https://zh.wikipedia.org/w/api.php",
        "https://zh.wikipedia.org/wiki/Research?redirect=http://127.0.0.1",
        "https://zh.wikipedia.org/wiki/Research#section",
        "https://zh.wikipedia.org/wiki/Research%7COther",
        "http://zh.wikipedia.org/wiki/Research",
    ],
)
async def test_wikipedia_fetch_rejects_urls_before_network(url):
    def unexpected(_):
        pytest.fail("forbidden URL reached network")

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(unexpected), trust_env=False
    ) as client:
        with pytest.raises(ResearchRetrievalError):
            await WikipediaRetriever(client=client).fetch(SearchHit(url, "Research"))


@pytest.mark.parametrize(
    "body",
    [b"not JSON", b"{}", b'{"error":{"code":"secret"}}', b'{"query":{"search":[{"title":1}]}}'],
)
async def test_wikipedia_invalid_search_protocol(body):
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, content=body)), trust_env=False
    ) as client:
        with pytest.raises(ResearchRetrievalError) as exc:
            await WikipediaRetriever(client=client).search("Research")
        assert exc.value.code == "retrieval_protocol"
        assert "secret" not in str(exc.value)


@pytest.mark.parametrize(
    "body", [{}, {"query": {"pages": []}}, extract_body(""), extract_body(" ")]
)
async def test_wikipedia_missing_or_empty_extract_rejected(body):
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, json=body)), trust_env=False
    ) as client:
        with pytest.raises(ResearchRetrievalError):
            await WikipediaRetriever("en", client=client).fetch(
                SearchHit("https://en.wikipedia.org/wiki/Research", "Research")
            )


async def test_wikipedia_document_utf8_truncation():
    body = extract_body("中" * (MAX_DOCUMENT_BYTES // 3 + 2))
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, json=body)), trust_env=False
    ) as client:
        doc = await WikipediaRetriever("en", client=client).fetch(
            SearchHit("https://en.wikipedia.org/wiki/Research", "Research")
        )
    assert doc.truncated
    assert len(doc.text.encode("utf-8")) <= MAX_DOCUMENT_BYTES
    assert "\ufffd" not in doc.text
    assert doc.text == "中" * (MAX_DOCUMENT_BYTES // 3)


async def test_wikipedia_response_size_limit():
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda _: httpx.Response(200, content=b"x" * (MAX_HTTP_RESPONSE_BYTES + 1))
        ),
        trust_env=False,
    ) as client:
        with pytest.raises(ResearchRetrievalError) as exc:
            await WikipediaRetriever(client=client).search("Research")
        assert exc.value.code == "retrieval_size"


@pytest.mark.parametrize(
    "exception,code",
    [(httpx.ReadTimeout, "retrieval_timeout"), (httpx.ConnectError, "retrieval_network")],
)
async def test_wikipedia_network_failures_sanitized(exception, code):
    def handler(_):
        raise exception("secret details")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler), trust_env=False) as client:
        with pytest.raises(ResearchRetrievalError) as exc:
            await WikipediaRetriever(client=client).search("Research")
        assert exc.value.code == code
        assert "secret" not in str(exc.value)


class BlockingStream(httpx.AsyncByteStream):
    def __init__(self):
        self.started = asyncio.Event()
        self.closed = False

    async def __aiter__(self):
        self.started.set()
        await asyncio.Event().wait()
        yield b"unreachable"

    async def aclose(self):
        self.closed = True


async def test_wikipedia_overall_timeout_closes_stream(monkeypatch):
    monkeypatch.setattr(retrieval, "REQUEST_TIMEOUT_SECONDS", 0.02)
    stream = BlockingStream()
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, stream=stream)), trust_env=False
    ) as client:
        with pytest.raises(ResearchRetrievalError) as exc:
            await WikipediaRetriever(client=client).search("Research")
        assert exc.value.code == "retrieval_timeout"
    assert stream.closed


async def test_wikipedia_cancel_propagates_and_closes_stream():
    stream = BlockingStream()
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, stream=stream)), trust_env=False
    ) as client:
        task = asyncio.create_task(WikipediaRetriever(client=client).search("Research"))
        await asyncio.wait_for(stream.started.wait(), timeout=1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert stream.closed


async def test_wikipedia_owned_client_closed_once(monkeypatch):
    owned = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(200)), trust_env=False
    )
    monkeypatch.setattr(retrieval.httpx, "AsyncClient", lambda **_: owned)
    retriever = WikipediaRetriever()
    await retriever.aclose()
    await retriever.aclose()
    assert owned.is_closed


def test_wikipedia_language_restricted():
    with pytest.raises(ResearchRetrievalError):
        WikipediaRetriever("evil.example")
