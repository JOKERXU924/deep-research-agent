"""Explicit local embeddings for compression, separate from the M3 memory index.

Only prepare_local_e5 may use the network. Encoding never downloads models or
falls back silently. A worker is drained on cancellation; Python cannot hard
kill an in-flight native CPU inference or download thread.

References: https://huggingface.co/intfloat/multilingual-e5-small
https://www.sbert.net/docs/package_reference/sentence_transformer/SentenceTransformer.html
"""

from __future__ import annotations

import asyncio
import hashlib
import importlib
import inspect
import json
import math
import re
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal, Protocol, runtime_checkable

import numpy as np

from deep_research.errors import ResearchError
from deep_research.memory_vector import LexicalEncoder

MAX_TEXTS = 256
MAX_TEXT_CHARS = 4000
MAX_TOTAL_CHARS = 256_000
MAX_MODEL_BYTES = 512 * 1024 * 1024
MAX_ASSET_BYTES = 32 * 1024 * 1024
MAX_CONFIG_BYTES = 64 * 1024
E5_REPOSITORY = "intfloat/multilingual-e5-small"
_HF_ENDPOINT = "https://huggingface.co"
MANIFEST_NAME = "embedding-manifest.json"
_REQUIRED_FILES = (
    "config.json",
    "modules.json",
    "sentence_bert_config.json",
    "1_Pooling/config.json",
    "tokenizer_config.json",
    "tokenizer.json",
    "special_tokens_map.json",
    "model.safetensors",
)
_OPTIONAL_FILES = ("sentencepiece.bpe.model", "config_sentence_transformers.json")
_MODULES = [
    {"idx": 0, "name": "0", "path": "", "type": "sentence_transformers.models.Transformer"},
    {"idx": 1, "name": "1", "path": "1_Pooling", "type": "sentence_transformers.models.Pooling"},
    {
        "idx": 2,
        "name": "2",
        "path": "2_Normalize",
        "type": "sentence_transformers.models.Normalize",
    },
]
_ARCHITECTURE = {
    "architectures": ["BertModel"],
    "model_type": "bert",
    "hidden_size": 384,
    "num_hidden_layers": 12,
    "num_attention_heads": 12,
    "intermediate_size": 1536,
    "max_position_embeddings": 512,
    "vocab_size": 250037,
    "tokenizer_class": "XLMRobertaTokenizer",
}


class EmbeddingError(ResearchError):
    """Stable embedding_* error without model paths, input text, or upstream bodies."""


@runtime_checkable
class EmbeddingEncoder(Protocol):
    encoder_id: str
    dimensions: int
    kind: Literal["semantic", "lexical", "fixture"]

    async def encode_queries(self, texts: list[str]) -> np.ndarray: ...

    async def encode_documents(self, texts: list[str]) -> np.ndarray: ...


def _error(code: str = "embedding_validation") -> EmbeddingError:
    messages = {
        "embedding_validation": "编码输入、配置或本地模型结构校验失败。",
        "embedding_model": "本地模型不完整、已变化或与支持的 E5 结构不兼容。",
        "embedding_dependency": "本地语义编码依赖未安装，请先显式安装 semantic 可选依赖。",
        "embedding_output": "编码器输出维度、数值或向量长度不合法。",
        "embedding_tokens": "输入超过本地模型的 token 上限，拒绝静默截断。",
        "embedding_timeout": "本地编码或准备超时；已等待当前工作线程清理。",
        "embedding_runtime": "本地编码失败；未切换到其他编码器。",
        "embedding_download": "模型准备失败，可能保留不完整目录；不会覆盖或自动重试。",
    }
    return EmbeddingError(code, messages[code])


def _texts(texts: list[str]) -> list[str]:
    if not isinstance(texts, list) or len(texts) > MAX_TEXTS:
        raise _error()
    total = 0
    for text in texts:
        if not isinstance(text, str) or not text.strip() or len(text) > MAX_TEXT_CHARS:
            raise _error()
        try:
            text.encode("utf-8")
        except UnicodeError:
            raise _error() from None
        total += len(text)
    if total > MAX_TOTAL_CHARS:
        raise _error()
    return list(texts)


def _normalized(
    value: object, rows: int, dimensions: int, *, allow_zero: bool = False
) -> np.ndarray:
    if (
        not isinstance(value, np.ndarray)
        or value.shape != (rows, dimensions)
        or value.dtype.kind not in "fiu"
        or not np.isfinite(value).all()
    ):
        raise _error("embedding_output")
    if not rows:
        return np.empty((0, dimensions), dtype=np.float32)
    data = value.astype(np.float64)
    scales = np.max(np.abs(data), axis=1, keepdims=True)
    if not allow_zero and (scales == 0).any():
        raise _error("embedding_output")
    np.divide(data, scales, out=data, where=scales != 0)
    lengths = np.linalg.norm(data, axis=1, keepdims=True)
    if not np.isfinite(lengths).all() or (not allow_zero and (lengths == 0).any()):
        raise _error("embedding_output")
    np.divide(data, lengths, out=data, where=lengths != 0)
    return data.astype(np.float32)


async def _offload[T](function: Callable[[threading.Event], T], *, budget_seconds: float) -> T:
    """Never detach a native worker, even when cancellation is requested twice."""
    await asyncio.sleep(0)  # Deliver cancellation before submitting new work.
    stop = threading.Event()
    worker = asyncio.create_task(asyncio.to_thread(function, stop))
    try:
        async with asyncio.timeout(budget_seconds):
            return await asyncio.shield(worker)
    except (asyncio.CancelledError, TimeoutError) as original:
        stop.set()
        cancelled = isinstance(original, asyncio.CancelledError)
        while not worker.done():
            try:
                await asyncio.shield(worker)
            except asyncio.CancelledError:
                cancelled = True
                continue
            except Exception:
                break
        if worker.done() and not worker.cancelled():
            worker.exception()  # Retrieve any secondary failure without replacing cancellation.
        if cancelled:
            if isinstance(original, asyncio.CancelledError):
                raise original
            raise asyncio.CancelledError from None
        raise _error("embedding_timeout") from None


def _check_stop(stop: threading.Event) -> None:
    if stop.is_set():
        raise _error("embedding_runtime")


class LexicalEmbedding:
    """Async interface for the unchanged M3 lexical encoder, not semantic E5."""

    kind: Literal["lexical"] = "lexical"
    encoder_id = LexicalEncoder.encoder_id
    dimensions = LexicalEncoder.dimensions

    def __init__(self) -> None:
        self._encoder = LexicalEncoder()

    def _encode_sync(self, values: list[str], stop: threading.Event) -> np.ndarray:
        rows = []
        for offset in range(0, len(values), 32):
            _check_stop(stop)
            batch = values[offset : offset + 32]
            rows.append(
                _normalized(
                    self._encoder.encode(batch), len(batch), self.dimensions, allow_zero=True
                )
            )
        _check_stop(stop)
        return np.concatenate(rows, axis=0)

    async def _encode(self, texts: list[str]) -> np.ndarray:
        values = _texts(texts)
        await asyncio.sleep(0)
        if not values:
            return np.empty((0, self.dimensions), dtype=np.float32)
        return await _offload(lambda stop: self._encode_sync(values, stop), budget_seconds=30)

    async def encode_queries(self, texts: list[str]) -> np.ndarray:
        return await self._encode(texts)

    async def encode_documents(self, texts: list[str]) -> np.ndarray:
        return await self._encode(texts)


def _json(path: Path) -> object:
    try:
        if not path.is_file() or path.stat().st_size > MAX_CONFIG_BYTES:
            raise _error("embedding_model")
        raw = path.read_bytes()
        if len(raw) > MAX_CONFIG_BYTES:
            raise _error("embedding_model")

        def reject(_: str) -> None:
            raise ValueError

        def finite(text: str) -> float:
            value = float(text)
            if not math.isfinite(value):
                raise ValueError
            return value

        result = json.loads(raw, parse_constant=reject, parse_float=finite)
    except (OSError, ValueError, TypeError, RecursionError):
        raise _error("embedding_model") from None
    return result


def _forbidden_keys(value: object) -> bool:
    if isinstance(value, dict):
        return any(
            key in {"auto_map", "custom_pipelines", "trust_remote_code", "code_revision"}
            or _forbidden_keys(item)
            for key, item in value.items()
        )
    return isinstance(value, list) and any(_forbidden_keys(item) for item in value)


def _safe_root(path: Path) -> Path:
    if not isinstance(path, Path) or str(path).startswith(("\\\\", "//")):
        raise _error("embedding_model")
    try:
        root = path.resolve(strict=True)
        if not root.is_dir():
            raise _error("embedding_model")
        return root
    except (OSError, RuntimeError):
        raise _error("embedding_model") from None


def _hash_file(path: Path, maximum: int, stop: threading.Event | None = None) -> str:
    try:
        size = path.stat().st_size
        if not path.is_file() or not 0 < size <= maximum:
            raise _error("embedding_model")
        digest = hashlib.sha256()
        read = 0
        with path.open("rb") as source:
            while data := source.read(1024 * 1024):
                if stop is not None:
                    _check_stop(stop)
                read += len(data)
                if read > maximum:
                    raise _error("embedding_model")
                digest.update(data)
        if read != size:
            raise _error("embedding_model")
        return digest.hexdigest()
    except OSError:
        raise _error("embedding_model") from None


def _inspect_model(path: Path, stop: threading.Event | None = None) -> dict:
    try:
        return _inspect_model_files(path, stop)
    except (OSError, RuntimeError, ValueError, RecursionError):
        raise _error("embedding_model") from None


def _inspect_model_files(path: Path, stop: threading.Event | None = None) -> dict:
    root = _safe_root(path)
    names = list(_REQUIRED_FILES)
    names.extend(name for name in _OPTIONAL_FILES if (root / name).exists())
    for name in names:
        target = root / name
        try:
            if not target.resolve(strict=True).is_relative_to(root):
                raise _error("embedding_model")
        except (OSError, RuntimeError):
            raise _error("embedding_model") from None
    # Do not let standard Transformers discover a local PEFT adapter or custom code.
    if (root / "adapter_config.json").exists():
        raise _error("embedding_model")
    modules = _json(root / "modules.json")
    config = _json(root / "config.json")
    transformer = _json(root / "sentence_bert_config.json")
    pooling = _json(root / "1_Pooling/config.json")
    tokenizer = _json(root / "tokenizer_config.json")
    if modules != _MODULES or not isinstance(config, dict):
        raise _error("embedding_model")
    if any(config.get(key) != value for key, value in _ARCHITECTURE.items()):
        raise _error("embedding_model")
    if transformer != {"max_seq_length": 512, "do_lower_case": False}:
        raise _error("embedding_model")
    if pooling != {
        "word_embedding_dimension": 384,
        "pooling_mode_cls_token": False,
        "pooling_mode_mean_tokens": True,
        "pooling_mode_max_tokens": False,
        "pooling_mode_mean_sqrt_len_tokens": False,
    }:
        raise _error("embedding_model")
    if (
        not isinstance(tokenizer, dict)
        or tokenizer.get("tokenizer_class")
        not in ("XLMRobertaTokenizer", "XLMRobertaTokenizerFast")
        or tokenizer.get("model_max_length") != 512
    ):
        raise _error("embedding_model")
    configs = [config, transformer, pooling, tokenizer]
    if "config_sentence_transformers.json" in names:
        sentence_config = _json(root / "config_sentence_transformers.json")
        if not isinstance(sentence_config, dict) or sentence_config.get("default_prompt_name"):
            raise _error("embedding_model")
        configs.append(sentence_config)
    if any(_forbidden_keys(value) for value in configs):
        raise _error("embedding_model")
    identities = {}
    total = 0
    for name in sorted(names):
        maximum = MAX_MODEL_BYTES if name == "model.safetensors" else MAX_ASSET_BYTES
        identities[name] = _hash_file(root / name, maximum, stop)
        total += (root / name).stat().st_size
    if total > MAX_MODEL_BYTES + 2 * MAX_ASSET_BYTES:
        raise _error("embedding_model")
    revision = None
    if (root / MANIFEST_NAME).exists():
        manifest = _json(root / MANIFEST_NAME)
        if (
            not isinstance(manifest, dict)
            or manifest.get("repo") != E5_REPOSITORY
            or not isinstance(manifest.get("revision"), str)
            or not re.fullmatch(r"[a-f0-9]{40}", manifest["revision"])
            or manifest.get("files") != identities
        ):
            raise _error("embedding_model")
        revision = manifest["revision"]
    fingerprint = hashlib.sha256(
        json.dumps(identities, sort_keys=True, separators=(",", ":")).encode("ascii")
    ).hexdigest()
    return {
        "encoder_id": f"local-e5-small-v1:384:{fingerprint}",
        "revision": revision,
        "files": identities,
        "bytes": total,
    }


class LocalE5Embedding:
    """CPU multilingual-e5-small from an explicitly trusted local directory.

    A content fingerprint detects drift; it is not a publisher signature. Local
    manifests and configs are data, never permission to execute custom modules.
    Construct outside the asynchronous research budget: initial bounded hashing
    is synchronous. Loading and inference are lazy, serialized, and offloaded.
    """

    kind: Literal["semantic"] = "semantic"
    dimensions = 384

    def __init__(
        self, model_path: Path, *, batch_size: int = 16, timeout_seconds: float = 120
    ) -> None:
        if type(batch_size) is not int or not 1 <= batch_size <= 32:
            raise _error()
        if type(timeout_seconds) not in (int, float) or not 0 < timeout_seconds <= 3600:
            raise _error()
        if not math.isfinite(timeout_seconds):
            raise _error()
        self._path = _safe_root(model_path)
        self._identity = _inspect_model(self._path)
        self.encoder_id = self._identity["encoder_id"]
        self._batch_size = batch_size
        self._timeout = timeout_seconds
        self._model: Any = None
        self._lock = asyncio.Lock()

    def _load(self, stop: threading.Event) -> None:
        _check_stop(stop)
        identity = _inspect_model(self._path, stop)
        if identity["encoder_id"] != self.encoder_id:
            raise _error("embedding_model")
        try:
            module = importlib.import_module("sentence_transformers")
        except ImportError:
            raise _error("embedding_dependency") from None
        _check_stop(stop)
        processor_key = (
            "processor_kwargs"
            if "processor_kwargs" in inspect.signature(module.SentenceTransformer).parameters
            else "tokenizer_kwargs"
        )
        model = module.SentenceTransformer(
            str(self._path),
            local_files_only=True,
            trust_remote_code=False,
            device="cpu",
            backend="torch",
            token=False,
            model_kwargs={"use_safetensors": True, "attn_implementation": "eager"},
            config_kwargs={"local_files_only": True, "trust_remote_code": False},
            **{processor_key: {"local_files_only": True, "trust_remote_code": False}},
        )
        dimension = getattr(model, "get_embedding_dimension", None)
        if dimension is None:
            dimension = model.get_sentence_embedding_dimension
        if dimension() != 384 or model.max_seq_length != 512:
            raise _error("embedding_model")
        # Detect files changed during load before exposing a usable encoder.
        if _inspect_model(self._path, stop)["encoder_id"] != self.encoder_id:
            raise _error("embedding_model")
        model.eval()
        self._model = model

    def _encode_sync(self, texts: list[str], prefix: str, stop: threading.Event) -> np.ndarray:
        try:
            if self._model is None:
                self._load(stop)
            _check_stop(stop)
            values = [prefix + value for value in texts]
            # Validate the entire input before encoding any batch. Include E5
            # prefix and tokenizer special tokens in the 512-token limit.
            for offset in range(0, len(values), self._batch_size):
                _check_stop(stop)
                batch = values[offset : offset + self._batch_size]
                encoded = self._model.tokenizer(
                    batch,
                    add_special_tokens=True,
                    truncation=False,
                    padding=False,
                    return_attention_mask=False,
                    return_token_type_ids=False,
                )
                ids = encoded.get("input_ids")
                if not isinstance(ids, list) or len(ids) != len(batch):
                    raise _error("embedding_tokens")
                if any(not isinstance(row, list) or not 0 < len(row) <= 512 for row in ids):
                    raise _error("embedding_tokens")
            rows = []
            for offset in range(0, len(values), self._batch_size):
                _check_stop(stop)
                batch = values[offset : offset + self._batch_size]
                value = self._model.encode(
                    batch,
                    batch_size=self._batch_size,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    precision="float32",
                    prompt="",
                )
                rows.append(_normalized(value, len(batch), self.dimensions))
            _check_stop(stop)
            return np.concatenate(rows, axis=0)
        except EmbeddingError:
            raise
        except Exception:
            raise _error("embedding_runtime") from None

    async def _encode(self, texts: list[str], prefix: str) -> np.ndarray:
        values = _texts(texts)
        await asyncio.sleep(0)
        if not values:
            return np.empty((0, self.dimensions), dtype=np.float32)
        async with self._lock:
            return await _offload(
                lambda stop: self._encode_sync(values, prefix, stop), budget_seconds=self._timeout
            )

    async def encode_queries(self, texts: list[str]) -> np.ndarray:
        return await self._encode(texts, "query: ")

    async def encode_documents(self, texts: list[str]) -> np.ndarray:
        return await self._encode(texts, "passage: ")


def build_encoder(kind: str, model_path: Path | None = None) -> EmbeddingEncoder:
    if kind == "lexical" and model_path is None:
        return LexicalEmbedding()
    if kind == "local-e5" and model_path is not None:
        return LocalE5Embedding(model_path)
    raise _error()


def _prepare_sync(destination: Path, stop: threading.Event) -> dict:
    try:
        _check_stop(stop)
        if not isinstance(destination, Path) or str(destination).startswith(("\\\\", "//")):
            raise _error("embedding_model")
        if destination.exists():
            identity = _inspect_model(destination, stop)
            return {"status": "existing", "model_path": str(destination.resolve()), **identity}
        try:
            hub = importlib.import_module("huggingface_hub")
        except ImportError:
            raise _error("embedding_dependency") from None
        # Explicit preparation is the only place that contacts Hugging Face.
        info = hub.HfApi(token=False, endpoint=_HF_ENDPOINT).model_info(
            E5_REPOSITORY, revision="main", timeout=15, files_metadata=True
        )
        revision = info.sha
        if not isinstance(revision, str) or not re.fullmatch(r"[a-f0-9]{40}", revision):
            raise _error("embedding_download")
        names = {item.rfilename for item in info.siblings}
        if not set(_REQUIRED_FILES) <= names:
            raise _error("embedding_download")
        selected = [*_REQUIRED_FILES, *(name for name in _OPTIONAL_FILES if name in names)]
        metadata = {item.rfilename: item for item in info.siblings}
        total = 0
        for name in selected:
            size = metadata[name].size
            maximum = MAX_MODEL_BYTES if name == "model.safetensors" else MAX_ASSET_BYTES
            if type(size) is not int or not 0 < size <= maximum:
                raise _error("embedding_download")
            total += size
        if total > MAX_MODEL_BYTES + 2 * MAX_ASSET_BYTES:
            raise _error("embedding_download")
        _check_stop(stop)
        destination.mkdir(parents=True, exist_ok=False)
        for name in selected:
            _check_stop(stop)
            hub.hf_hub_download(
                repo_id=E5_REPOSITORY,
                filename=name,
                revision=revision,
                local_dir=str(destination),
                token=False,
                etag_timeout=15,
                endpoint=_HF_ENDPOINT,
            )
        _check_stop(stop)
        identity = _inspect_model(destination, stop)
        manifest = {"repo": E5_REPOSITORY, "revision": revision, "files": identity["files"]}
        # Exclusive creation: a partial preparation is never silently replaced.
        with (destination / MANIFEST_NAME).open("x", encoding="utf-8") as output:
            json.dump(manifest, output, ensure_ascii=False, sort_keys=True, indent=2)
        identity["revision"] = revision
        return {"status": "prepared", "model_path": str(destination.resolve()), **identity}
    except EmbeddingError:
        raise
    except Exception:
        raise _error("embedding_download") from None


async def prepare_local_e5(destination: Path) -> dict:
    """Explicit network preparation; existing/partial directories are never overwritten."""
    return await _offload(lambda stop: _prepare_sync(destination, stop), budget_seconds=1800)
