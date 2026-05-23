import math
import time
from typing import Any

from anchor_mcp.backends.base import QueryResult
from anchor_mcp.chunk import Chunk
from anchor_mcp.embed import HybridEmbedding
from anchor_mcp.errors import BackendError

EMBEDDING_DIM = 1024
_DEFAULT_CLOUD = "aws"
_DEFAULT_REGION = "us-east-1"


class PineconeBackend:
    def __init__(self, pc_client: Any, index_name: str = "anchor") -> None:
        existing = {idx.name for idx in pc_client.list_indexes()}
        if index_name not in existing:
            try:
                from pinecone import ServerlessSpec  # type: ignore[import-untyped]
            except ImportError as exc:
                raise BackendError(
                    "pinecone is not installed. Install with: pip install pinecone"
                ) from exc
            pc_client.create_index(
                name=index_name,
                dimension=EMBEDDING_DIM,
                metric="dotproduct",  # required for hybrid search
                spec=ServerlessSpec(cloud=_DEFAULT_CLOUD, region=_DEFAULT_REGION),
            )
            for _ in range(60):
                if pc_client.describe_index(index_name).status.get("ready", False):
                    break
                time.sleep(2)

        self._index: Any = pc_client.Index(index_name)

        stats: Any = self._index.describe_index_stats()
        dim = getattr(stats, "dimension", None)
        if dim is not None and int(dim) != EMBEDDING_DIM:
            raise BackendError(
                f"Pinecone index '{index_name}' has dimension {dim}, "
                f"but anchor requires {EMBEDDING_DIM}. "
                f"Delete the index and run 'anchor sync' to recreate it."
            )

    def upsert(self, chunks: list[Chunk], embeddings: list[HybridEmbedding]) -> None:
        if not chunks:
            return
        vectors = [
            {
                "id": c.id,
                "values": emb.dense,
                "sparse_values": {
                    "indices": emb.sparse.indices,
                    "values": emb.sparse.values,
                },
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
        embedding: HybridEmbedding,
        top_k: int,
        alpha: float = 0.7,
        file_name_filter: str | None = None,
    ) -> list[QueryResult]:
        # Scale dense by alpha, sparse by (1 - alpha) for hybrid blending
        scaled_dense = [v * alpha for v in embedding.dense]
        scaled_sparse = {
            "indices": embedding.sparse.indices,
            "values": [v * (1 - alpha) for v in embedding.sparse.values],
        }

        kwargs: dict[str, Any] = {
            "vector": scaled_dense,
            "sparse_vector": scaled_sparse,
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

    def count(self) -> int:
        stats: Any = self._index.describe_index_stats()
        return int(stats.total_vector_count)

    def get_chunks_by_file(self, file_id: str) -> list[Chunk]:
        # Neutral dense vector + empty sparse to satisfy Pinecone's query requirement.
        # The filter on file_id drives the result, not vector similarity.
        neutral_dense = [1.0 / math.sqrt(EMBEDDING_DIM)] * EMBEDDING_DIM
        neutral_sparse: dict[str, Any] = {"indices": [], "values": []}
        response: Any = self._index.query(
            vector=neutral_dense,
            sparse_vector=neutral_sparse,
            top_k=10_000,
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
