import logging
from typing import Any

from pydantic import BaseModel
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_random_exponential,
)

from anchor_mcp.chunk import Chunk
from anchor_mcp.errors import BackendError

logger = logging.getLogger("anchor_mcp")

_BATCH_SIZE = 96
_MAX_ATTEMPTS = 6


def _is_rate_limit_or_transient(exc: BaseException) -> bool:
    """Retry Pinecone inference on 429 (rate limit) and 5xx (transient server) errors.

    Pinecone raises ApiError (aka PineconeApiException) with a ``status_code``
    attribute; anything 4xx other than 429 is a caller error and should not retry.
    """
    status = getattr(exc, "status_code", None)
    try:
        code = int(status) if status is not None else None
    except (TypeError, ValueError):
        code = None
    if code is None:
        return False
    return code == 429 or 500 <= code < 600


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

    @retry(
        retry=retry_if_exception(_is_rate_limit_or_transient),
        wait=wait_random_exponential(multiplier=1, max=30),
        stop=stop_after_attempt(_MAX_ATTEMPTS),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    def _embed_call(self, model: str, texts: list[str], parameters: dict[str, Any]) -> Any:
        return self._pc.inference.embed(model=model, inputs=texts, parameters=parameters)

    def _embed_batch(self, texts: list[str], input_type: str) -> list[HybridEmbedding]:
        try:
            dense_response: Any = self._embed_call(
                self._dense_model,
                texts,
                {"input_type": input_type, "truncate": "END"},
            )
            sparse_response: Any = self._embed_call(
                self._sparse_model,
                texts,
                {"input_type": input_type},
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
