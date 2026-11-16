"""Embedding contracts with tiny fake assets: no real weights, downloads, or inference."""

import asyncio
import json
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from deep_research import embeddings
from deep_research.embeddings import (
    EmbeddingEncoder,
    EmbeddingError,
    LexicalEmbedding,
    LocalE5Embedding,
    build_encoder,
    prepare_local_e5,
)
from deep_research.memory_vector import LexicalEncoder


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def model_dir(tmp_path: Path) -> Path:
    path = tmp_path / "local-model"
    path.mkdir()
    write_json(path / "modules.json", embeddings._MODULES)
    write_json(path / "config.json", embeddings._ARCHITECTURE)
    write_json(path / "sentence_bert_config.json", {"max_seq_length": 512, "do_lower_case": False})
    write_json(
        path / "1_Pooling/config.json",
        {
            "word_embedding_dimension": 384,
            "pooling_mode_cls_token": False,
            "pooling_mode_mean_tokens": True,
            "pooling_mode_max_tokens": False,
            "pooling_mode_mean_sqrt_len_tokens": False,
        },
    )
    write_json(
        path / "tokenizer_config.json",
        {
            "tokenizer_class": "XLMRobertaTokenizer",
            "model_max_length": 512,
        },
    )
    write_json(path / "tokenizer.json", {"fixture": True})
    write_json(path / "special_tokens_map.json", {"pad_token": "<pad>"})
    (path / "model.safetensors").write_bytes(b"NOT REAL WEIGHTS: mocked loader only")
    return path


class FakeModel:
    max_seq_length = 512

    def __init__(self) -> None:
        self.calls: list[tuple[list[str], dict]] = []
        self.token_calls: list[tuple[list[str], dict]] = []
        self.evaluated = False
        self.output = None

    def get_sentence_embedding_dimension(self) -> int:
        return 384

    def eval(self) -> None:
        self.evaluated = True

    def tokenizer(self, batch, **kwargs):
        self.token_calls.append((list(batch), kwargs))
        return {"input_ids": [[1] * (len(text.split()) + 2) for text in batch]}

    def encode(self, batch, **kwargs):
        self.calls.append((list(batch), kwargs))
        if self.output is not None:
            return self.output
        return np.ones((len(batch), 384), dtype=np.float32)


@pytest.fixture
def fake_loader(monkeypatch):
    model = FakeModel()
    loaded = []

    def load(path, **kwargs):
        loaded.append((path, kwargs))
        return model

    def import_module(name):
        assert name == "sentence_transformers", "unexpected optional dependency import"
        return SimpleNamespace(SentenceTransformer=load)

    monkeypatch.setattr(embeddings.importlib, "import_module", import_module)
    return model, loaded


async def wait_event(event: threading.Event) -> None:
    assert await asyncio.to_thread(event.wait, 3)


async def test_lexical_adapter_uses_unchanged_encoder() -> None:
    encoder = LexicalEmbedding()
    texts = ["Plants need sunlight.", "植物需要阳光。"]
    assert isinstance(encoder, EmbeddingEncoder)
    assert encoder.kind == "lexical"
    assert encoder.encoder_id == "lexical-hash-v1:1024"
    assert encoder.dimensions == 1024
    for method in (encoder.encode_queries, encoder.encode_documents):
        actual = await method(texts)
        assert actual.dtype == np.float32
        np.testing.assert_allclose(actual, LexicalEncoder().encode(texts), atol=1e-7)
        assert (await method([])).shape == (0, 1024)


@pytest.mark.parametrize(
    "values",
    [
        "not-a-list",
        [1],
        [""],
        ["   "],
        ["\ud800"],
        ["a" * 4001],
        ["x"] * 257,
        ["a" * 4000] * 65,
    ],
)
async def test_input_bounds_apply_before_optional_load(model_dir, fake_loader, values) -> None:
    encoder = LocalE5Embedding(model_dir)
    with pytest.raises(EmbeddingError) as error:
        await encoder.encode_queries(values)
    assert error.value.code == "embedding_validation"
    assert fake_loader[1] == []


async def test_lexical_no_tokens_remains_zero_without_pretending_to_be_semantic() -> None:
    encoder = LexicalEmbedding()
    result = await encoder.encode_documents(["!!!", "Plants need sunlight"])
    assert encoder.kind == "lexical"
    assert not result[0].any()
    assert np.linalg.norm(result[1]) == pytest.approx(1)


def test_factory_is_explicit_and_never_falls_back(model_dir) -> None:
    assert isinstance(build_encoder("lexical"), LexicalEmbedding)
    assert isinstance(build_encoder("local-e5", model_dir), LocalE5Embedding)
    for kind, path in [("local-e5", None), ("unknown", None), ("lexical", model_dir)]:
        with pytest.raises(EmbeddingError):
            build_encoder(kind, path)


async def test_e5_local_only_load_prefixes_normalization_and_batches(
    model_dir, fake_loader
) -> None:
    model, loaded = fake_loader
    encoder = LocalE5Embedding(model_dir, batch_size=2)
    assert isinstance(encoder, EmbeddingEncoder)
    assert encoder.kind == "semantic"
    assert encoder.encoder_id.startswith("local-e5-small-v1:384:")
    assert loaded == []  # Lazy import/load, not lazy path/config verification.
    assert (await encoder.encode_documents([])).shape == (0, 384)
    assert loaded == []
    vectors = await encoder.encode_queries(["问题一", "问题二", "问题三"])
    assert vectors.shape == (3, 384)
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-6)
    await encoder.encode_documents(["原文段落"])
    assert len(loaded) == 1
    settings = loaded[0][1]
    assert settings["local_files_only"] is True
    assert settings["trust_remote_code"] is False
    assert settings["token"] is False
    assert settings["device"] == "cpu"
    assert settings["backend"] == "torch"
    assert settings["model_kwargs"]["use_safetensors"] is True
    assert settings["tokenizer_kwargs"]["local_files_only"] is True
    assert model.evaluated
    assert [call[0] for call in model.calls] == [
        ["query: 问题一", "query: 问题二"],
        ["query: 问题三"],
        ["passage: 原文段落"],
    ]
    assert all(call[1]["prompt"] == "" for call in model.calls)
    assert all(call[1]["truncation"] is False for call in model.token_calls)


async def test_new_sentence_transformers_names_are_used_without_deprecated_aliases(
    model_dir, monkeypatch
) -> None:
    model = FakeModel()
    model.get_embedding_dimension = lambda: 384
    settings = []

    def load(path, *, processor_kwargs=None, **kwargs):
        settings.append((processor_kwargs, kwargs))
        return model

    monkeypatch.setattr(
        embeddings.importlib, "import_module", lambda _: SimpleNamespace(SentenceTransformer=load)
    )
    await LocalE5Embedding(model_dir).encode_queries(["test"])
    assert settings[0][0] == {"local_files_only": True, "trust_remote_code": False}
    assert "tokenizer_kwargs" not in settings[0][1]


def test_missing_or_partial_directory_errors_are_sanitized(tmp_path) -> None:
    for path in (tmp_path / "secret-missing", tmp_path):
        with pytest.raises(EmbeddingError) as error:
            LocalE5Embedding(path)
        assert error.value.code == "embedding_model"
        assert str(tmp_path) not in str(error.value)


@pytest.mark.parametrize(
    "filename,value",
    [
        ("modules.json", [{"idx": 0, "path": "../bad", "type": "evil.Payload", "name": "x"}]),
        ("config.json", {**embeddings._ARCHITECTURE, "architectures": ["CustomModel"]}),
        ("config.json", {**embeddings._ARCHITECTURE, "hidden_size": 768}),
        ("config.json", {**embeddings._ARCHITECTURE, "auto_map": {"AutoModel": "evil.Model"}}),
        ("sentence_bert_config.json", {"max_seq_length": 256, "do_lower_case": False}),
        ("tokenizer_config.json", {"tokenizer_class": "CustomTokenizer", "model_max_length": 512}),
        (
            "tokenizer_config.json",
            {"tokenizer_class": "XLMRobertaTokenizer", "model_max_length": 128},
        ),
        ("1_Pooling/config.json", {"word_embedding_dimension": 768}),
        ("config_sentence_transformers.json", {"default_prompt_name": "malicious"}),
    ],
)
def test_untrusted_or_incompatible_configuration_is_rejected_before_import(
    model_dir, fake_loader, filename, value
) -> None:
    write_json(model_dir / filename, value)
    with pytest.raises(EmbeddingError) as error:
        LocalE5Embedding(model_dir)
    assert error.value.code == "embedding_model"
    assert fake_loader[1] == []


def test_local_adapter_discovery_is_disallowed(model_dir) -> None:
    write_json(model_dir / "adapter_config.json", {"base_model_name_or_path": "remote"})
    with pytest.raises(EmbeddingError):
        LocalE5Embedding(model_dir)


def test_file_fingerprint_is_path_independent_and_detects_different_weights(
    model_dir, tmp_path
) -> None:
    first = LocalE5Embedding(model_dir)
    other = tmp_path / "different-path"
    for name in embeddings._REQUIRED_FILES:
        target = other / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((model_dir / name).read_bytes())
    assert LocalE5Embedding(other).encoder_id == first.encoder_id
    (other / "model.safetensors").write_bytes(b"different fake weights")
    assert LocalE5Embedding(other).encoder_id != first.encoder_id


async def test_weights_changed_between_construction_and_loading_fail_closed(
    model_dir, fake_loader
) -> None:
    encoder = LocalE5Embedding(model_dir)
    (model_dir / "model.safetensors").write_bytes(b"tampered")
    with pytest.raises(EmbeddingError) as error:
        await encoder.encode_documents(["test"])
    assert error.value.code == "embedding_model"
    assert fake_loader[1] == []


def test_manifest_hash_mismatch_is_rejected(model_dir) -> None:
    identity = embeddings._inspect_model(model_dir)
    write_json(
        model_dir / embeddings.MANIFEST_NAME,
        {
            "repo": embeddings.E5_REPOSITORY,
            "revision": "a" * 40,
            "files": identity["files"],
        },
    )
    assert LocalE5Embedding(model_dir).encoder_id == identity["encoder_id"]
    (model_dir / "model.safetensors").write_bytes(b"tampered")
    with pytest.raises(EmbeddingError):
        LocalE5Embedding(model_dir)


async def test_token_preflight_includes_prefix_and_checks_all_batches_first(
    model_dir, fake_loader
) -> None:
    encoder = LocalE5Embedding(model_dir, batch_size=1)
    # 509 words + prefix + two special tokens = 512, accepted exactly.
    await encoder.encode_queries(["x " * 509])
    fake_loader[0].calls.clear()
    with pytest.raises(EmbeddingError) as error:
        await encoder.encode_documents(["short", "x " * 510])
    assert error.value.code == "embedding_tokens"
    assert fake_loader[0].calls == []  # No partial successful batch precedes rejection.


@pytest.mark.parametrize(
    "output",
    [
        np.ones((1, 383)),
        np.ones((2, 384)),
        np.zeros((1, 384)),
        np.full((1, 384), np.nan),
        np.full((1, 384), np.inf),
        np.ones((1, 384), dtype=complex),
        np.ones((1, 384), dtype=bool),
        [[1.0] * 384],
    ],
)
async def test_invalid_model_output_is_rejected(model_dir, fake_loader, output) -> None:
    fake_loader[0].output = output
    with pytest.raises(EmbeddingError) as error:
        await LocalE5Embedding(model_dir).encode_documents(["test"])
    assert error.value.code == "embedding_output"


async def test_large_finite_values_normalize_without_overflow(model_dir, fake_loader) -> None:
    fake_loader[0].output = np.full((1, 384), 1e300)
    value = await LocalE5Embedding(model_dir).encode_documents(["test"])
    assert value.dtype == np.float32
    assert np.isfinite(value).all()
    assert np.linalg.norm(value[0]) == pytest.approx(1)


async def test_missing_optional_dependency_is_clear_and_does_not_fallback(
    model_dir, monkeypatch
) -> None:
    def missing(_):
        raise ImportError("secret filesystem detail")

    monkeypatch.setattr(embeddings.importlib, "import_module", missing)
    with pytest.raises(EmbeddingError) as error:
        await LocalE5Embedding(model_dir).encode_documents(["test"])
    assert error.value.code == "embedding_dependency"
    assert "secret" not in str(error.value)


async def test_runtime_errors_do_not_expose_text_or_paths(
    model_dir, fake_loader, monkeypatch
) -> None:
    def broken(*args, **kwargs):
        raise RuntimeError("secret key and user content and private path")

    monkeypatch.setattr(fake_loader[0], "encode", broken)
    with pytest.raises(EmbeddingError) as error:
        await LocalE5Embedding(model_dir).encode_documents(["private user input"])
    assert error.value.code == "embedding_runtime"
    assert "secret" not in str(error.value)
    assert "private" not in str(error.value)


async def test_cancel_drains_worker_and_never_launches_next_batch(
    model_dir, fake_loader, monkeypatch
) -> None:
    model = fake_loader[0]
    started, release = threading.Event(), threading.Event()
    calls = []

    def blocking(batch, **kwargs):
        calls.append(list(batch))
        started.set()
        assert release.wait(3)
        return np.ones((len(batch), 384), dtype=np.float32)

    monkeypatch.setattr(model, "encode", blocking)
    encoder = LocalE5Embedding(model_dir, batch_size=1)
    task = asyncio.create_task(encoder.encode_documents(["first", "second"]))
    await wait_event(started)
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert calls == [["passage: first"]]
    await encoder.encode_queries(["reusable"])
    assert calls[-1] == ["query: reusable"]


async def test_timeout_drains_worker_before_raising(model_dir, fake_loader, monkeypatch) -> None:
    started, release = threading.Event(), threading.Event()
    calls = []

    def blocking(batch, **kwargs):
        calls.append(list(batch))
        started.set()
        assert release.wait(3)
        return np.ones((len(batch), 384), dtype=np.float32)

    monkeypatch.setattr(fake_loader[0], "encode", blocking)
    encoder = LocalE5Embedding(model_dir, batch_size=1, timeout_seconds=0.03)
    encoder._model = fake_loader[0]  # Separate inference timeout from first-load filesystem timing.
    task = asyncio.create_task(encoder.encode_documents(["first", "second"]))
    await wait_event(started)
    await asyncio.sleep(0.05)
    assert not task.done()
    release.set()
    with pytest.raises(EmbeddingError) as error:
        await task
    assert error.value.code == "embedding_timeout"
    assert calls == [["passage: first"]]


async def test_cancel_before_start_does_not_load_model(model_dir, fake_loader) -> None:
    task = asyncio.create_task(LocalE5Embedding(model_dir).encode_documents(["test"]))
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert fake_loader[1] == []


async def test_cancel_waiting_for_serialization_does_not_start_second_worker(
    model_dir, fake_loader, monkeypatch
) -> None:
    started, release = threading.Event(), threading.Event()
    calls = []

    def blocking(batch, **kwargs):
        calls.append(batch)
        started.set()
        assert release.wait(3)
        return np.ones((len(batch), 384))

    monkeypatch.setattr(fake_loader[0], "encode", blocking)
    encoder = LocalE5Embedding(model_dir)
    first = asyncio.create_task(encoder.encode_queries(["one"]))
    await wait_event(started)
    second = asyncio.create_task(encoder.encode_queries(["two"]))
    await asyncio.sleep(0)
    second.cancel()
    with pytest.raises(asyncio.CancelledError):
        await second
    release.set()
    await first
    assert calls == [["query: one"]]


def fake_hub(model_dir: Path, monkeypatch, *, hook=None):
    requests = []
    revision = "a" * 40

    class Api:
        def __init__(self, **kwargs):
            assert kwargs == {"token": False, "endpoint": "https://huggingface.co"}

        def model_info(self, repo, **kwargs):
            assert repo == embeddings.E5_REPOSITORY
            assert kwargs == {"revision": "main", "timeout": 15, "files_metadata": True}
            return SimpleNamespace(
                sha=revision,
                siblings=[
                    SimpleNamespace(rfilename=name, size=(model_dir / name).stat().st_size)
                    for name in embeddings._REQUIRED_FILES
                ],
            )

    def download(**kwargs):
        requests.append(kwargs)
        assert kwargs["repo_id"] == embeddings.E5_REPOSITORY
        assert kwargs["revision"] == revision
        assert kwargs["token"] is False
        assert kwargs["endpoint"] == "https://huggingface.co"
        name = kwargs["filename"]
        assert name in embeddings._REQUIRED_FILES
        if hook:
            hook()
        target = Path(kwargs["local_dir"]) / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((model_dir / name).read_bytes())
        return str(target)

    def load(name):
        assert name == "huggingface_hub"
        return SimpleNamespace(HfApi=Api, hf_hub_download=download)

    monkeypatch.setattr(embeddings.importlib, "import_module", load)
    return requests


async def test_explicit_prepare_pins_revision_allowlists_files_and_writes_manifest(
    model_dir, tmp_path, monkeypatch
) -> None:
    requests = fake_hub(model_dir, monkeypatch)
    target = tmp_path / "prepared"
    result = await prepare_local_e5(target)
    assert result["status"] == "prepared"
    assert result["revision"] == "a" * 40
    assert result["model_path"] == str(target.resolve())
    assert len(requests) == len(embeddings._REQUIRED_FILES)
    manifest = json.loads((target / embeddings.MANIFEST_NAME).read_text())
    assert manifest["files"] == result["files"]
    assert LocalE5Embedding(target).encoder_id == result["encoder_id"]
    before = len(requests)
    existing = await prepare_local_e5(target)
    assert existing["status"] == "existing"
    assert len(requests) == before


async def test_partial_prepare_directory_is_not_overwritten_or_downloaded(
    tmp_path, monkeypatch
) -> None:
    target = tmp_path / "partial"
    target.mkdir()
    marker = target / "notes.txt"
    marker.write_text("user-owned")

    def forbid(_):
        pytest.fail("partial directory must be rejected before network dependency")

    monkeypatch.setattr(embeddings.importlib, "import_module", forbid)
    with pytest.raises(EmbeddingError) as error:
        await prepare_local_e5(target)
    assert error.value.code == "embedding_model"
    assert marker.read_text() == "user-owned"


async def test_cancel_preparation_drains_current_download_and_stops_remaining(
    model_dir, tmp_path, monkeypatch
) -> None:
    started, release = threading.Event(), threading.Event()

    def hook():
        started.set()
        assert release.wait(3)

    requests = fake_hub(model_dir, monkeypatch, hook=hook)
    target = tmp_path / "partial-preparation"
    task = asyncio.create_task(prepare_local_e5(target))
    await wait_event(started)
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert len(requests) == 1
    assert not (target / embeddings.MANIFEST_NAME).exists()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"batch_size": 0},
        {"batch_size": 33},
        {"batch_size": True},
        {"timeout_seconds": 0},
        {"timeout_seconds": float("nan")},
        {"timeout_seconds": True},
        {"timeout_seconds": 10**1000},
    ],
)
def test_constructor_limits(model_dir, kwargs) -> None:
    with pytest.raises(EmbeddingError):
        LocalE5Embedding(model_dir, **kwargs)
