import math
from unittest.mock import MagicMock, patch

import numpy as np

from anchor_mcp.chunk import Chunk
from anchor_mcp.embed import Embedder


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


def _mock_model(n: int, dim: int = 1024) -> MagicMock:
    model = MagicMock()
    arr = np.random.randn(n, dim).astype("float32")
    arr /= np.linalg.norm(arr, axis=1, keepdims=True)
    model.encode.return_value = arr
    return model


def test_embed_chunks_returns_correct_count() -> None:
    mock_model = _mock_model(3)
    with patch("sentence_transformers.SentenceTransformer", return_value=mock_model):
        embedder = Embedder()
        result = embedder.embed_chunks([_chunk(i) for i in range(3)])
    assert len(result) == 3


def test_embed_chunks_returns_correct_dim() -> None:
    mock_model = _mock_model(2)
    with patch("sentence_transformers.SentenceTransformer", return_value=mock_model):
        embedder = Embedder()
        result = embedder.embed_chunks([_chunk(0), _chunk(1)])
    assert len(result[0]) == 1024


def test_embed_chunks_empty_returns_empty() -> None:
    embedder = Embedder()
    assert embedder.embed_chunks([]) == []


def test_embed_chunks_calls_normalize() -> None:
    mock_model = _mock_model(1)
    with patch("sentence_transformers.SentenceTransformer", return_value=mock_model):
        embedder = Embedder()
        embedder.embed_chunks([_chunk(0)])
    _, kwargs = mock_model.encode.call_args
    assert kwargs.get("normalize_embeddings") is True


def test_embed_query_returns_vector() -> None:
    vec = np.random.randn(1024).astype("float32")
    vec /= np.linalg.norm(vec)
    mock_model = MagicMock()
    mock_model.encode.return_value = vec

    with patch("sentence_transformers.SentenceTransformer", return_value=mock_model):
        embedder = Embedder()
        result = embedder.embed_query("what is the project timeline?")

    assert len(result) == 1024
    assert abs(math.sqrt(sum(x * x for x in result)) - 1.0) < 1e-5


def test_embedder_lazy_loads_model() -> None:
    with patch("sentence_transformers.SentenceTransformer", return_value=_mock_model(1)) as mock_st:
        embedder = Embedder()
        mock_st.assert_not_called()
        embedder.embed_query("hello")
        mock_st.assert_called_once_with("BAAI/bge-m3")
