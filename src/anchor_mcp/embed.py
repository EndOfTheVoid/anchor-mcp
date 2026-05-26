from typing import Any

from pydantic import BaseModel

from anchor_mcp.chunk import Chunk
from anchor_mcp.errors import BackendError

_BATCH_SIZE = 96


class SparseValues(BaseModel):
    indices: list[int]
    values: list[float]


class HybridEmbedding(BaseModel):
    dense: list[float]
    sparse: SparseValues


class PineconeEmbedder:
    def __init__(self, pc_client: Any, dense_model: str, sparse_model: str) -> None:
        self._pc = pc_client
        self._dense_model = dense_model
        self._sparse_model = sparse_model

    def embed_chunks(self, chunks: list[Chunk]) -> list[HybridEmbedding]:
        if not chunks:
            return []
        return self._embed_texts([c.text for c in chunks], input_type="passage")

    def embed_query(self, text: str) -> HybridEmbedding:
        return self._embed_texts([text], input_type="query")[0]

    def _embed_texts(self, texts: list[str], input_type: str) -> list[HybridEmbedding]:
        results: list[HybridEmbedding] = []
        for i in range(0, len(texts), _BATCH_SIZE):
            batch = texts[i : i + _BATCH_SIZE]
            results.extend(self._embed_batch(batch, input_type))
        return results

    def _embed_batch(self, texts: list[str], input_type: str) -> list[HybridEmbedding]:
        try:
            dense_response: Any = self._pc.inference.embed(
                model=self._dense_model,
                inputs=texts,
                parameters={"input_type": input_type, "truncate": "END"},
            )
            sparse_response: Any = self._pc.inference.embed(
                model=self._sparse_model,
                inputs=texts,
                parameters={"input_type": input_type},
            )
        except Exception as exc:
            raise BackendError(f"Pinecone inference API call failed: {exc}") from exc

        return [
            HybridEmbedding(
                dense=list(d.values),
                sparse=SparseValues(
                    indices=list(s.sparse_indices),
                    values=list(s.sparse_values),
                ),
            )
            for d, s in zip(dense_response, sparse_response, strict=True)
        ]
