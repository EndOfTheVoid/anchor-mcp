from collections import defaultdict
from pathlib import Path
from typing import Any

import chromadb

from anchor_mcp.backends.base import QueryResult, SourceInfo
from anchor_mcp.chunk import Chunk

_COLLECTION = "anchor"


class ChromaBackend:
    def __init__(self, state_dir: Path) -> None:
        chroma_path = state_dir / "chroma"
        chroma_path.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(chroma_path))
        self._col = self._client.get_or_create_collection(
            _COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )

    def upsert(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if not chunks:
            return
        self._col.upsert(
            ids=[c.id for c in chunks],
            documents=[c.text for c in chunks],
            embeddings=embeddings,  # type: ignore[arg-type]
            metadatas=[
                {
                    "file_id": c.file_id,
                    "file_name": c.file_name,
                    "chunk_index": c.chunk_index,
                    "token_count": c.token_count,
                    "modified_time": c.modified_time,
                    "source_url": c.source_url or "",
                }
                for c in chunks
            ],
        )

    def query(
        self,
        embedding: list[float],
        top_k: int,
        file_name_filter: str | None = None,
    ) -> list[QueryResult]:
        total = self._col.count()
        if total == 0:
            return []

        kwargs: dict[str, Any] = {
            "query_embeddings": [embedding],
            "n_results": min(top_k, total),
            "include": ["documents", "metadatas", "distances"],
        }
        if file_name_filter is not None:
            kwargs["where"] = {"file_name": {"$eq": file_name_filter}}

        try:
            raw = self._col.query(**kwargs)
        except Exception:
            # ChromaDB raises when n_results > filtered result count
            return []

        ids: list[str] = raw["ids"][0]
        docs: list[str] = (raw["documents"] or [[]])[0]
        metas: list[Any] = (raw["metadatas"] or [[]])[0]
        dists: list[float] = (raw["distances"] or [[]])[0]

        results: list[QueryResult] = []
        for chunk_id, doc, meta, dist in zip(ids, docs, metas, dists, strict=True):
            if meta is None:
                continue
            chunk = Chunk(
                id=chunk_id,
                text=doc,
                file_id=str(meta["file_id"]),
                file_name=str(meta["file_name"]),
                chunk_index=int(meta["chunk_index"]),
                token_count=int(meta["token_count"]),
                modified_time=str(meta["modified_time"]),
                source_url=str(meta["source_url"]) or None,
            )
            results.append(QueryResult(chunk=chunk, score=1.0 - dist))

        return results

    def delete(self, chunk_ids: list[str]) -> None:
        if chunk_ids:
            self._col.delete(ids=chunk_ids)

    def list_sources(self) -> list[SourceInfo]:
        raw = self._col.get(include=["metadatas"])  # type: ignore[call-arg]
        metadatas: list[Any] = raw.get("metadatas") or []  # type: ignore[attr-defined]

        counts: dict[str, int] = defaultdict(int)
        info: dict[str, tuple[str, str]] = {}

        for meta in metadatas:
            if meta is None:
                continue
            file_id = str(meta["file_id"])
            counts[file_id] += 1
            if file_id not in info:
                info[file_id] = (str(meta["file_name"]), str(meta["modified_time"]))

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
        return self._col.count()

    def get_chunks_by_file(self, file_id: str) -> list[Chunk]:
        raw = self._col.get(  # type: ignore[call-arg]
            where={"file_id": {"$eq": file_id}},
            include=["documents", "metadatas"],
        )
        ids: list[str] = raw.get("ids") or []  # type: ignore[attr-defined]
        docs: list[Any] = raw.get("documents") or []  # type: ignore[attr-defined]
        metas: list[Any] = raw.get("metadatas") or []  # type: ignore[attr-defined]

        chunks: list[Chunk] = []
        for chunk_id, doc, meta in zip(ids, docs, metas, strict=True):
            if meta is None:
                continue
            chunks.append(
                Chunk(
                    id=chunk_id,
                    text=doc,
                    file_id=str(meta["file_id"]),
                    file_name=str(meta["file_name"]),
                    chunk_index=int(meta["chunk_index"]),
                    token_count=int(meta["token_count"]),
                    modified_time=str(meta["modified_time"]),
                    source_url=str(meta["source_url"]) or None,
                )
            )
        chunks.sort(key=lambda c: c.chunk_index)
        return chunks
