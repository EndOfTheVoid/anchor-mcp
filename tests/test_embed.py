from unittest.mock import MagicMock

import pytest
from pinecone.exceptions import PineconeApiException
from pinecone.models.inference.embed import DenseEmbedding, SparseEmbedding

from anchor_mcp.chunk import Chunk
from anchor_mcp.embed import (
    HybridEmbedding,
    PineconeEmbedder,
    SparseValues,
    _is_rate_limit_or_transient,
)
from anchor_mcp.errors import BackendError


def _chunk(i: int) -> Chunk:
    return Chunk(
        id=f"id{i}",
        text=f"chunk text {i}",
        file_id="f1",
        file_name="doc.txt",
        chunk_index=i,
        token_count=3,
        modified_time="2024-01-01T00:00:00.000Z",
        source_url=None,
    )


# Use Pinecone's real response classes so the test fails if the API shape drifts
# (a MagicMock previously masked a sparse_indices/sparse_values attribute bug).
def _make_dense_response(n: int, dim: int = 1024) -> list[DenseEmbedding]:
    return [DenseEmbedding(values=[0.1] * dim) for _ in range(n)]


def _make_sparse_response(n: int) -> list[SparseEmbedding]:
    return [SparseEmbedding(sparse_values=[0.5, 0.3], sparse_indices=[1, 2]) for _ in range(n)]


def _make_pc(n_chunks: int) -> MagicMock:
    pc = MagicMock()
    pc.inference.embed.side_effect = [
        _make_dense_response(n_chunks),
        _make_sparse_response(n_chunks),
    ]
    return pc


def test_embed_query_returns_hybrid_embedding() -> None:
    pc = _make_pc(1)
    embedder = PineconeEmbedder(pc, "dense-model", "sparse-model")
    result = embedder.embed_query("test query")

    assert isinstance(result, HybridEmbedding)
    assert len(result.dense) == 1024
    assert isinstance(result.sparse, SparseValues)
    assert result.sparse.indices == [1, 2]
    assert result.sparse.values == [0.5, 0.3]


def test_embed_query_uses_query_input_type() -> None:
    pc = _make_pc(1)
    embedder = PineconeEmbedder(pc, "dense-model", "sparse-model")
    embedder.embed_query("hello")

    first_call = pc.inference.embed.call_args_list[0]
    params = first_call.kwargs.get("parameters") or first_call.args[2]
    assert params["input_type"] == "query"


def test_embed_chunks_uses_passage_input_type() -> None:
    pc = _make_pc(1)
    embedder = PineconeEmbedder(pc, "dense-model", "sparse-model")
    embedder.embed_chunks([_chunk(0)])

    first_call = pc.inference.embed.call_args_list[0]
    params = first_call.kwargs.get("parameters") or first_call.args[2]
    assert params["input_type"] == "passage"


def test_embed_chunks_returns_correct_count() -> None:
    pc = _make_pc(3)
    embedder = PineconeEmbedder(pc, "dense-model", "sparse-model")
    results = embedder.embed_chunks([_chunk(i) for i in range(3)])
    assert len(results) == 3


def test_embed_chunks_empty_returns_empty() -> None:
    pc = MagicMock()
    embedder = PineconeEmbedder(pc, "dense-model", "sparse-model")
    assert embedder.embed_chunks([]) == []
    pc.inference.embed.assert_not_called()


def test_embed_chunks_batches_at_96() -> None:
    pc = MagicMock()
    # 100 chunks → 2 batches: 96 + 4 → 4 total inference calls (dense+sparse per batch)
    pc.inference.embed.side_effect = [
        _make_dense_response(96),
        _make_sparse_response(96),
        _make_dense_response(4),
        _make_sparse_response(4),
    ]
    embedder = PineconeEmbedder(pc, "dense-model", "sparse-model")
    results = embedder.embed_chunks([_chunk(i) for i in range(100)])

    assert len(results) == 100
    assert pc.inference.embed.call_count == 4


# ── rate-limit retry ────────────────────────────────────────────────────────────


def test_is_rate_limit_or_transient_classifies_status_codes() -> None:
    assert _is_rate_limit_or_transient(PineconeApiException("rate limited", 429)) is True
    assert _is_rate_limit_or_transient(PineconeApiException("unavailable", 503)) is True
    assert _is_rate_limit_or_transient(PineconeApiException("bad request", 400)) is False
    assert _is_rate_limit_or_transient(ValueError("no status_code")) is False


def test_embed_retries_on_rate_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    # Don't actually sleep between retries.
    monkeypatch.setattr(PineconeEmbedder._embed_call.retry, "sleep", lambda _: None)

    pc = MagicMock()
    pc.inference.embed.side_effect = [
        PineconeApiException("rate limited", 429),  # dense attempt 1 → retry
        _make_dense_response(1),  # dense attempt 2 → ok
        _make_sparse_response(1),  # sparse → ok
    ]
    embedder = PineconeEmbedder(pc, "dense-model", "sparse-model")

    result = embedder.embed_query("test query")

    assert isinstance(result, HybridEmbedding)
    assert pc.inference.embed.call_count == 3


def test_embed_does_not_retry_client_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(PineconeEmbedder._embed_call.retry, "sleep", lambda _: None)

    pc = MagicMock()
    pc.inference.embed.side_effect = PineconeApiException("bad request", 400)
    embedder = PineconeEmbedder(pc, "dense-model", "sparse-model")

    with pytest.raises(BackendError):
        embedder.embed_query("test query")

    # 400 is a caller error — no retry, and it surfaces as a clean BackendError.
    assert pc.inference.embed.call_count == 1
