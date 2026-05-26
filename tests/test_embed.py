from unittest.mock import MagicMock

from pinecone.models.inference.embed import DenseEmbedding, SparseEmbedding

from anchor_mcp.chunk import Chunk
from anchor_mcp.embed import HybridEmbedding, PineconeEmbedder, SparseValues


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
