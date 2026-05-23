from typing import Protocol

from pydantic import BaseModel

from anchor_mcp.chunk import Chunk
from anchor_mcp.embed import HybridEmbedding


class QueryResult(BaseModel):
    chunk: Chunk
    score: float  # cosine similarity; higher is more relevant


class VectorBackend(Protocol):
    def upsert(self, chunks: list[Chunk], embeddings: list[HybridEmbedding]) -> None: ...

    def query(
        self,
        embedding: HybridEmbedding,
        top_k: int,
        alpha: float = 0.7,
        file_name_filter: str | None = None,
    ) -> list[QueryResult]: ...

    def delete(self, chunk_ids: list[str]) -> None: ...

    def count(self) -> int: ...

    def get_chunks_by_file(self, file_id: str) -> list[Chunk]: ...
