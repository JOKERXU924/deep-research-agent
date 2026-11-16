"""Bounded retrieval: an offline corpus or the fixed Wikipedia Action API.

Local corpus URLs are provenance labels, never network destinations. Wikipedia
extracts omit some original-page material and are not a general web search tool.
"""

import asyncio
import ipaddress
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol
from urllib.parse import quote, unquote, urlsplit, urlunsplit

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from deep_research.errors import ResearchError
from deep_research.relevance import title_relevance_score

MAX_DOCUMENT_BYTES = 64 * 1024
MAX_HTTP_RESPONSE_BYTES = 1024 * 1024
MAX_CORPUS_BYTES = 1024 * 1024
REQUEST_TIMEOUT_SECONDS = 15.0


class ResearchRetrievalError(ResearchError):
    """Stable retrieval errors without raw upstream bodies or local file paths."""


@dataclass(frozen=True)
class SearchHit:
    url: str
    title: str


@dataclass(frozen=True)
class SourceDocument:
    url: str
    title: str
    text: str
    truncated: bool = False


class ResearchRetriever(Protocol):
    async def search(self, query: str, limit: int = 3) -> list[SearchHit]: ...

    async def fetch(self, hit: SearchHit) -> SourceDocument: ...


def _fail(code: str, message: str) -> ResearchRetrievalError:
    return ResearchRetrievalError(code, message)


def _public_url(url: str) -> str:
    """Syntactic public-domain validation; no DNS lookup and no URL fetching."""
    try:
        if not isinstance(url, str) or len(url.encode("utf-8")) > 2048:
            raise ValueError
        if any(char.isspace() or ord(char) < 32 for char in url) or "\\" in url:
            raise ValueError
        parts = urlsplit(url)
        host = parts.hostname or ""
        if (
            parts.scheme != "https"
            or parts.username is not None
            or parts.password is not None
            or parts.port not in (None, 443)
            or not host
            or host.endswith(".")
            or "." not in host
            or host.endswith((".localhost", ".local", ".internal", ".lan", ".home"))
        ):
            raise ValueError
        try:
            ipaddress.ip_address(host)
        except ValueError:
            pass
        else:
            # Do not accept IP literals, including globally routed ones.
            raise ValueError
        ascii_host = host.encode("idna").decode("ascii").lower()
        if len(ascii_host) > 253 or not all(
            re.fullmatch(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?", label)
            for label in ascii_host.split(".")
        ):
            raise ValueError
        if not re.fullmatch(r"[a-z][a-z0-9-]+", ascii_host.rsplit(".", 1)[1]):
            raise ValueError
        return urlunsplit(("https", ascii_host, parts.path or "/", parts.query, parts.fragment))
    except (ValueError, UnicodeError):
        raise _fail("retrieval_url", "来源网址必须是无凭据的公共 HTTPS 域名网址。") from None


def _query_arguments(query: str, limit: int) -> str:
    try:
        valid = (
            isinstance(query, str)
            and bool(query.strip())
            and len(query.encode("utf-8")) <= 1024
            and not any(ord(char) < 32 for char in query)
            and type(limit) is int
            and 1 <= limit <= 10
        )
    except UnicodeError:
        valid = False
    if not valid:
        raise _fail("retrieval_input", "检索词不能为空或超过 1024 字节，条数须为 1 至 10。")
    return query.strip()


def _utf8_prefix(text: str, limit: int = MAX_DOCUMENT_BYTES) -> tuple[str, bool]:
    try:
        raw = text.encode("utf-8")
    except UnicodeError:
        raise _fail("retrieval_protocol", "来源正文包含无效 Unicode 字符。") from None
    return raw[:limit].decode("utf-8", errors="ignore"), len(raw) > limit


def _terms(text: str) -> set[str]:
    words = set(re.findall(r"[a-z0-9]+", text.casefold()))
    for segment in re.findall(r"[\u3400-\u9fff]+", text):
        words.update(segment[index : index + 2] for index in range(len(segment) - 1))
        if len(segment) == 1:
            words.add(segment)
    return words


class _CorpusDocument(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    url: str
    title: str = Field(min_length=1, max_length=512)
    text: str = Field(min_length=1)


class _Corpus(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    documents: list[_CorpusDocument] = Field(min_length=1, max_length=50)


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _no_constant(_: str) -> None:
    raise ValueError("invalid JSON constant")


class LocalCorpusRetriever:
    def __init__(self, documents: tuple[SourceDocument, ...]) -> None:
        self._documents = documents
        self._by_url = {document.url: document for document in documents}

    @classmethod
    def from_file(cls, path: Path) -> "LocalCorpusRetriever":
        try:
            with path.open("rb") as stream:
                raw = stream.read(MAX_CORPUS_BYTES + 1)
        except OSError:
            raise _fail("corpus_read", "无法读取本地资料文件。") from None
        if len(raw) > MAX_CORPUS_BYTES:
            raise _fail("corpus_size", "本地资料文件不能超过 1 MiB。")
        try:
            value = json.loads(
                raw.decode("utf-8"),
                object_pairs_hook=_unique_object,
                parse_constant=_no_constant,
            )
            corpus = _Corpus.model_validate(value)
        except (ValueError, UnicodeError, RecursionError):
            raise _fail("corpus_format", "本地资料须为符合约定字段的严格 UTF-8 JSON。") from None
        documents: list[SourceDocument] = []
        seen: set[str] = set()
        for item in corpus.documents:
            url = _public_url(item.url)
            if url in seen:
                raise _fail("corpus_duplicate", "本地资料包含重复来源网址。")
            if not item.title.strip() or not item.text.strip():
                raise _fail("corpus_format", "资料标题和正文不能为空。")
            _, truncated = _utf8_prefix(item.text)
            if truncated:
                raise _fail("corpus_size", "单篇本地资料不能超过 64 KiB。")
            seen.add(url)
            documents.append(SourceDocument(url=url, title=item.title.strip(), text=item.text))
        return cls(tuple(documents))

    async def search(self, query: str, limit: int = 3) -> list[SearchHit]:
        wanted = _terms(_query_arguments(query, limit))
        ranked = []
        for index, document in enumerate(self._documents):
            score = 3 * len(wanted & _terms(document.title)) + len(wanted & _terms(document.text))
            if score:
                ranked.append((-score, index, document))
        ranked.sort(key=lambda item: (item[0], item[1]))
        return [SearchHit(item.url, item.title) for _, _, item in ranked[:limit]]

    async def fetch(self, hit: SearchHit) -> SourceDocument:
        # Lookup only. An arbitrary URL can never initiate network traffic here.
        document = self._by_url.get(hit.url)
        if document is None or document.title != hit.title:
            raise _fail("source_missing", "该来源不在已加载的本地资料中。")
        return document


class _WireModel(BaseModel):
    model_config = ConfigDict(strict=True, extra="ignore")


class _SearchItem(_WireModel):
    ns: Literal[0]
    pageid: int = Field(gt=0)
    title: str = Field(min_length=1, max_length=512)


class _SearchQuery(_WireModel):
    search: list[_SearchItem] = Field(max_length=10)


class _SearchResponse(_WireModel):
    query: _SearchQuery


class _ExtractPage(_WireModel):
    ns: Literal[0]
    pageid: int = Field(gt=0)
    title: str = Field(min_length=1, max_length=512)
    extract: str = Field(min_length=1)
    missing: bool = False


class _ExtractQuery(_WireModel):
    pages: list[_ExtractPage] = Field(min_length=1, max_length=1)


class _ExtractResponse(_WireModel):
    query: _ExtractQuery


class WikipediaRetriever:
    def __init__(
        self,
        language: Literal["zh", "en"] = "zh",
        *,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        if language not in ("zh", "en"):
            raise _fail("retrieval_config", "Wikipedia 检索仅支持 zh 或 en。")
        self._host = f"{language}.wikipedia.org"
        self._endpoint = f"https://{self._host}/w/api.php"
        self._owns_client = client is None
        self._client = client if client is not None else httpx.AsyncClient(follow_redirects=False)
        self._closed = False

    async def __aenter__(self) -> "WikipediaRetriever":
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        if not self._closed:
            self._closed = True
            if self._owns_client:
                await self._client.aclose()

    def _article_url(self, title: str) -> str:
        if (
            not title.strip()
            or len(title.encode("utf-8")) > 1024
            or any(ord(char) < 32 for char in title)
            or any(char in title for char in "|#\\")
        ):
            raise _fail("retrieval_protocol", "来源服务返回了无效页面标题。")
        return f"https://{self._host}/wiki/" + quote(title.replace(" ", "_"), safe="")

    async def _request(self, parameters: dict[str, str | int]) -> bytes:
        if self._closed:
            raise _fail("retrieval_closed", "检索连接已关闭。")
        try:
            async with asyncio.timeout(REQUEST_TIMEOUT_SECONDS):
                async with self._client.stream(
                    "GET",
                    self._endpoint,
                    params={
                        "action": "query",
                        "format": "json",
                        "formatversion": "2",
                        **parameters,
                    },
                    headers={
                        "User-Agent": "DeepResearchAgent-M1/0.1 (local educational client)",
                        "Accept": "application/json",
                    },
                    timeout=REQUEST_TIMEOUT_SECONDS,
                    follow_redirects=False,
                ) as response:
                    if response.status_code != 200:
                        raise ResearchRetrievalError(
                            "retrieval_http", "资料服务拒绝请求。", status_code=response.status_code
                        )
                    body = bytearray()
                    async for chunk in response.aiter_bytes():
                        body.extend(chunk)
                        if len(body) > MAX_HTTP_RESPONSE_BYTES:
                            raise _fail("retrieval_size", "资料服务响应超过 1 MiB。")
                    return bytes(body)
        except (TimeoutError, httpx.TimeoutException):
            raise _fail("retrieval_timeout", "检索请求超时，本次未自动重试。") from None
        except httpx.RequestError:
            raise _fail("retrieval_network", "无法连接资料服务，请检查网络。") from None
        # Cancellation deliberately propagates and closes the response stream.

    async def search(self, query: str, limit: int = 3) -> list[SearchHit]:
        query = _query_arguments(query, limit)
        raw = await self._request(
            {
                "list": "search",
                "srsearch": query,
                "srlimit": limit,
                "srnamespace": "0",
                "srprop": "",
            }
        )
        try:
            result = _SearchResponse.model_validate_json(raw)
        except ValidationError:
            raise _fail("retrieval_protocol", "检索服务返回了无效的结果结构。") from None
        hits = []
        seen: set[str] = set()
        for item in result.query.search[:limit]:
            url = self._article_url(item.title)
            if url not in seen:
                hits.append(SearchHit(url=url, title=item.title))
                seen.add(url)
        # Prefer explicit topic titles among the results we already paid to
        # retrieve. Stable sorting preserves upstream order for tied scores.
        # This does not invent Wikipedia search syntax or add network requests.
        hits.sort(key=lambda hit: -title_relevance_score(query, hit.title))
        return hits

    async def fetch(self, hit: SearchHit) -> SourceDocument:
        try:
            parsed = urlsplit(_public_url(hit.url))
            if (
                parsed.hostname != self._host
                or parsed.query
                or parsed.fragment
                or not parsed.path.startswith("/wiki/")
            ):
                raise ValueError
            title = unquote(parsed.path.removeprefix("/wiki/"), errors="strict").replace("_", " ")
            if self._article_url(title) != hit.url or title != hit.title.replace("_", " "):
                raise ValueError
        except (ValueError, UnicodeError):
            raise _fail("retrieval_url", "只能读取本检索器返回的 Wikipedia 页面网址。") from None
        raw = await self._request(
            {"prop": "extracts", "titles": title, "explaintext": "1", "exsectionformat": "plain"}
        )
        try:
            page = _ExtractResponse.model_validate_json(raw).query.pages[0]
        except ValidationError:
            raise _fail("retrieval_protocol", "资料服务未返回有效正文，页面可能不存在。") from None
        if page.missing or not page.extract.strip():
            raise _fail("source_missing", "来源页面不存在或正文为空。")
        text, truncated = _utf8_prefix(page.extract)
        return SourceDocument(self._article_url(page.title), page.title, text, truncated)
