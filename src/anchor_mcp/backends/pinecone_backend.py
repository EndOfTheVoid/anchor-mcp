from collections import defaultdict
from typing import Any

from anchor_mcp import secrets
from anchor_mcp.backends.base import QueryResult, SourceInfo
from anchor_mcp.chunk import Chunk
from anchor_mcp.errors import BackendError

EMBEDDING_DIM = 1024


class PineconeBackend:
    def __init__(self) -> None:
        try:
            from pinecone import Pinecone  # type: ignore[import-untyped]
        except ImportError as exc:
            raise BackendError(
                "pinecone-client is not installed. "
                "Install with: pip install 'anchor-mcp[pinecone]'"
            ) from exc

        api_key = secrets.get_pinecone_api_key()
        if not api_key:
            raise BackendError("PINECONE_API_KEY is not set.")

        index_name = secrets.get_pinecone_index_name()
        pc: Any = Pinecone(api_key=api_key)
        self._index: Any = pc.Index(index_name)

        stats: Any = self._index.describe_index_stats()
        dim = getattr(stats, "dimension", None)
        if dim is not None and int(dim) != EMBEDDING_DIM:
            raise BackendError(
                f"Pinecone index dimension {dim} != expected {EMBEDDING_DIM}. "
                f"Re-create the index with dimension={EMBEDDING_DIM}."
            )

    def upsert(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if not chunks:
            return
        vectors = [
            {
                "id": c.id,
                "values": emb,
                "metadata": {
                    "file_id": c.file_id,
                    "file_name": c.file_name,
                    "chunk_index": c.chunk_index,
                    "token_count": c.token_count,
                    "modified_time": c.modified_time,
                    "source_url": c.source_url or "",
                    "text": c.text,
                },
            }
            for c, emb in zip(chunks, embeddings, strict=True)
        ]
        for i in range(0, len(vectors), 100):
            self._index.upsert(vectors=vectors[i : i + 100])

    def query(
        self,
        embedding: list[float],
        top_k: int,
        file_name_filter: str | None = None,
    ) -> list[QueryResult]:
        kwargs: dict[str, Any] = {
            "vector": embedding,
            "top_k": top_k,
            "include_metadata": True,
        }
        if file_name_filter is not None:
            kwargs["filter"] = {"file_name": {"$eq": file_name_filter}}

        response: Any = self._index.query(**kwargs)
        results: list[QueryResult] = []

        for match in response.matches:
            meta: dict[str, Any] = match.metadata or {}
            chunk = Chunk(
                id=match.id,
                text=str(meta.get("text", "")),
                file_id=str(meta.get("file_id", "")),
                file_name=str(meta.get("file_name", "")),
                chunk_index=int(meta.get("chunk_index", 0)),
                token_count=int(meta.get("token_count", 0)),
                modified_time=str(meta.get("modified_time", "")),
                source_url=str(meta.get("source_url")) or None,
            )
            results.append(QueryResult(chunk=chunk, score=float(match.score)))

        return results

    def delete(self, chunk_ids: list[str]) -> None:
        if chunk_ids:
            self._index.delete(ids=chunk_ids)

    def list_sources(self) -> list[SourceInfo]:
        counts: dict[str, int] = defaultdict(int)
        info: dict[str, tuple[str, str]] = {}

        for ids_page in self._index.list():
            if not ids_page:
                continue
            response: Any = self._index.fetch(ids=ids_page)
            for _vec_id, vec in (response.vectors or {}).items():
                meta: dict[str, Any] = vec.metadata or {}
                file_id = str(meta.get("file_id", ""))
                if not file_id:
                    continue
                counts[file_id] += 1
                if file_id not in info:
                    info[file_id] = (
                        str(meta.get("file_name", "")),
                        str(meta.get("modified_time", "")),
                    )

        return [
            SourceInfo(
                file_id=fid,
                file_name=info[fid][0],
                chunk_count=counts[fid],
                modified_time=info[fid][1],
            )
            for fid in counts
        ]

    def count(self) -> int:
        stats: Any = self._index.describe_index_stats()
        return int(stats.total_vector_count)

    def get_chunks_by_file(self, file_id: str) -> list[Chunk]:
        import math

        dim = EMBEDDING_DIM
        neutral = [1.0 / math.sqrt(dim)] * dim
        response: Any = self._index.query(
            vector=neutral,
            top_k=1000,
            filter={"file_id": {"$eq": file_id}},
            include_metadata=True,
        )
        chunks: list[Chunk] = []
        for match in response.matches:
            meta: dict[str, Any] = match.metadata or {}
            chunks.append(
                Chunk(
                    id=match.id,
                    text=str(meta.get("text", "")),
                    file_id=str(meta.get("file_id", "")),
                    file_name=str(meta.get("file_name", "")),
                    chunk_index=int(meta.get("chunk_index", 0)),
                    token_count=int(meta.get("token_count", 0)),
                    modified_time=str(meta.get("modified_time", "")),
                    source_url=str(meta.get("source_url")) or None,
                )
            )
        chunks.sort(key=lambda c: c.chunk_index)
        return chunks
