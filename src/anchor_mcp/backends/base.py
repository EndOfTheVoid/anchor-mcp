from typing import Protocol

from pydantic import BaseModel

from anchor_mcp.chunk import Chunk


class QueryResult(BaseModel):
    chunk: Chunk
    score: float  # cosine similarity in [-1, 1]; higher is more relevant


class SourceInfo(BaseModel):
    file_id: str
    file_name: str
    chunk_count: int
    modified_time: str


class VectorBackend(Protocol):
    def upsert(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None: ...

    def query(
        self,
        embedding: list[float],
        top_k: int,
        file_name_filter: str | None = None,
    ) -> list[QueryResult]: ...

    def delete(self, chunk_ids: list[str]) -> None: ...

    def list_sources(self) -> list[SourceInfo]: ...

    def count(self) -> int: ...
