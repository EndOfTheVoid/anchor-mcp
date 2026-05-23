from unittest.mock import MagicMock

import pytest

from anchor_mcp.backends.pinecone_backend import PineconeBackend
from anchor_mcp.chunk import Chunk
from anchor_mcp.embed import HybridEmbedding, SparseValues


def _chunk(chunk_id: str, file_id: str = "f1", file_name: str = "doc.txt", idx: int = 0) -> Chunk:
    return Chunk(
        id=chunk_id,
        text=f"text for {chunk_id}",
        file_id=file_id,
        file_name=file_name,
        chunk_index=idx,
        token_count=4,
        modified_time="2024-01-01T00:00:00.000Z",
        source_url="https://example.com",
    )


def _hybrid(dense_val: float = 0.1, sparse_indices: list[int] | None = None) -> HybridEmbedding:
    return HybridEmbedding(
        dense=[dense_val] * 1024,
        sparse=SparseValues(
            indices=sparse_indices or [1, 2, 3],
            values=[0.5, 0.3, 0.2],
        ),
    )


def _make_backend() -> tuple[PineconeBackend, MagicMock]:
    pc = MagicMock()
    # Simulate an existing index so create_index is not called
    existing = MagicMock()
    existing.name = "anchor"
    pc.list_indexes.return_value = [existing]

    index = MagicMock()
    pc.Index.return_value = index

    stats = MagicMock()
    stats.dimension = 1024
    stats.total_vector_count = 0
    index.describe_index_stats.return_value = stats

    backend = PineconeBackend(pc, "anchor")
    return backend, index


# ── upsert ────────────────────────────────────────────────────────────────────


def test_upsert_includes_dense_and_sparse_values() -> None:
    backend, index = _make_backend()
    chunk = _chunk("c1")
    emb = _hybrid()
    backend.upsert([chunk], [emb])

    vectors = index.upsert.call_args.kwargs["vectors"]
    assert len(vectors) == 1
    v = vectors[0]
    assert v["values"] == emb.dense
    assert "sparse_values" in v
    assert v["sparse_values"]["indices"] == [1, 2, 3]
    assert v["sparse_values"]["values"] == [0.5, 0.3, 0.2]


def test_upsert_empty_is_noop() -> None:
    backend, index = _make_backend()
    backend.upsert([], [])
    index.upsert.assert_not_called()


def test_upsert_includes_metadata() -> None:
    backend, index = _make_backend()
    chunk = _chunk("c1", file_id="f1", file_name="myfile.txt")
    backend.upsert([chunk], [_hybrid()])

    vectors = index.upsert.call_args.kwargs["vectors"]
    meta = vectors[0]["metadata"]
    assert meta["file_id"] == "f1"
    assert meta["file_name"] == "myfile.txt"
    assert meta["text"] == "text for c1"


# ── query ─────────────────────────────────────────────────────────────────────


def _empty_query_response() -> MagicMock:
    resp = MagicMock()
    resp.matches = []
    return resp


def test_query_scales_dense_by_alpha() -> None:
    backend, index = _make_backend()
    index.query.return_value = _empty_query_response()
    emb = _hybrid(dense_val=1.0)
    backend.query(emb, top_k=5, alpha=0.7)

    kwargs = index.query.call_args.kwargs
    expected = [1.0 * 0.7] * 1024
    assert kwargs["vector"] == pytest.approx(expected)


def test_query_scales_sparse_by_one_minus_alpha() -> None:
    backend, index = _make_backend()
    index.query.return_value = _empty_query_response()
    emb = _hybrid()
    backend.query(emb, top_k=5, alpha=0.7)

    kwargs = index.query.call_args.kwargs
    expected_sparse_vals = [v * 0.3 for v in emb.sparse.values]
    assert kwargs["sparse_vector"]["values"] == pytest.approx(expected_sparse_vals)
    assert kwargs["sparse_vector"]["indices"] == [1, 2, 3]


def test_query_alpha_0_zeroes_dense() -> None:
    backend, index = _make_backend()
    index.query.return_value = _empty_query_response()
    backend.query(_hybrid(dense_val=1.0), top_k=5, alpha=0.0)

    kwargs = index.query.call_args.kwargs
    assert all(v == pytest.approx(0.0) for v in kwargs["vector"])


def test_query_alpha_1_zeroes_sparse() -> None:
    backend, index = _make_backend()
    index.query.return_value = _empty_query_response()
    backend.query(_hybrid(), top_k=5, alpha=1.0)

    kwargs = index.query.call_args.kwargs
    assert all(v == pytest.approx(0.0) for v in kwargs["sparse_vector"]["values"])


def test_query_with_file_name_filter() -> None:
    backend, index = _make_backend()
    index.query.return_value = _empty_query_response()
    backend.query(_hybrid(), top_k=3, file_name_filter="report.pdf")

    kwargs = index.query.call_args.kwargs
    assert kwargs["filter"] == {"file_name": {"$eq": "report.pdf"}}


def test_query_without_filter_has_no_filter_key() -> None:
    backend, index = _make_backend()
    index.query.return_value = _empty_query_response()
    backend.query(_hybrid(), top_k=3)

    kwargs = index.query.call_args.kwargs
    assert "filter" not in kwargs


# ── delete ────────────────────────────────────────────────────────────────────


def test_delete_calls_index_delete() -> None:
    backend, index = _make_backend()
    backend.delete(["c1", "c2"])
    index.delete.assert_called_once_with(ids=["c1", "c2"])


def test_delete_empty_list_is_noop() -> None:
    backend, index = _make_backend()
    backend.delete([])
    index.delete.assert_not_called()


# ── get_chunks_by_file ────────────────────────────────────────────────────────


def test_get_chunks_by_file_passes_file_id_filter() -> None:
    backend, index = _make_backend()
    index.query.return_value = _empty_query_response()
    backend.get_chunks_by_file("f1")

    kwargs = index.query.call_args.kwargs
    assert kwargs["filter"] == {"file_id": {"$eq": "f1"}}
    assert kwargs["top_k"] == 10_000


# ── factory ───────────────────────────────────────────────────────────────────


def test_get_backend_unknown_raises() -> None:
    import pytest

    from anchor_mcp.backends import get_backend
    from anchor_mcp.config import AnchorConfig
    from anchor_mcp.errors import BackendError

    # AnchorConfig only allows "pinecone" so we bypass validation
    cfg = AnchorConfig.model_construct(
        vector_backend="unknown", pinecone_index="x", drive_folder_id="y"
    )  # type: ignore[call-arg]
    with pytest.raises(BackendError, match="Unknown vector backend"):
        get_backend(cfg)
