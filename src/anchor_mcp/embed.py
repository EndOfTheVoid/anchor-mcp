from typing import Any

from anchor_mcp.chunk import Chunk

EMBEDDING_DIM = 1024
_BATCH_SIZE = 32


class Embedder:
    """Wraps bge-m3 via sentence-transformers. Model is lazy-loaded on first use."""

    def __init__(self, model_name: str = "BAAI/bge-m3") -> None:
        self._model_name = model_name
        self._model: Any = None

    def _get_model(self) -> Any:
        if self._model is None:
            from sentence_transformers import SentenceTransformer  # type: ignore[import-untyped]

            self._model = SentenceTransformer(self._model_name)
        return self._model

    def embed_chunks(self, chunks: list[Chunk]) -> list[list[float]]:
        if not chunks:
            return []
        model = self._get_model()
        result: list[list[float]] = model.encode(
            [c.text for c in chunks],
            batch_size=_BATCH_SIZE,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).tolist()
        return result

    def embed_query(self, text: str) -> list[float]:
        model = self._get_model()
        result: list[float] = model.encode(
            text,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).tolist()
        return result
