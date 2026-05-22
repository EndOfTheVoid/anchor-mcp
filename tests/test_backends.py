import math
import os
from pathlib import Path

import pytest

from anchor_mcp.backends import get_backend
from anchor_mcp.backends.chroma_backend import ChromaBackend
from anchor_mcp.chunk import Chunk
from anchor_mcp.config import AnchorConfig  # noqa: F401

# ── helpers ───────────────────────────────────────────────────────────────────

def _embed(seed: float, dim: int = 4) -> list[float]:
    """Deterministic unit vector for testing (no real embedder needed)."""
    vec = [seed + i for i in range(dim)]
    norm = math.sqrt(sum(x * x for x in vec))
    return [x / norm for x in vec]


def _chunk(chunk_id: str, file_id: str = "f1", file_name: str = "doc.txt", idx: int = 0) -> Chunk:
    return Chunk(
        id=chunk_id,
        text=f"text for {chunk_id}",
        file_id=file_id,
        file_name=file_name,
        chunk_index=idx,
        token_count=4,
        modified_time="2024-01-01T00:00:00.000Z",
        source_url="https://example.com",
    )


# ── ChromaBackend ─────────────────────────────────────────────────────────────

def test_chroma_upsert_and_count(tmp_path: Path) -> None:
    backend = ChromaBackend(tmp_path)
    chunks = [_chunk("a"), _chunk("b"), _chunk("c")]
    embeddings = [_embed(1.0), _embed(2.0), _embed(3.0)]
    backend.upsert(chunks, embeddings)
    assert backend.count() == 3


def test_chroma_upsert_empty_is_noop(tmp_path: Path) -> None:
    backend = ChromaBackend(tmp_path)
    backend.upsert([], [])
    assert backend.count() == 0


def test_chroma_query_returns_top_match(tmp_path: Path) -> None:
    backend = ChromaBackend(tmp_path)
    emb_a = _embed(1.0)
    emb_b = _embed(2.0)
    emb_c = _embed(100.0)
    backend.upsert([_chunk("a"), _chunk("b"), _chunk("c")], [emb_a, emb_b, emb_c])

    results = backend.query(emb_a, top_k=1)

    assert len(results) == 1
    assert results[0].chunk.id == "a"
    assert results[0].score > 0.99


def test_chroma_query_empty_collection_returns_empty(tmp_path: Path) -> None:
    backend = ChromaBackend(tmp_path)
    results = backend.query(_embed(1.0), top_k=5)
    assert results == []


def test_chroma_query_respects_top_k(tmp_path: Path) -> None:
    backend = ChromaBackend(tmp_path)
    chunks = [_chunk(f"c{i}") for i in range(10)]
    embeddings = [_embed(float(i)) for i in range(10)]
    backend.upsert(chunks, embeddings)

    results = backend.query(_embed(0.0), top_k=3)
    assert len(results) == 3


def test_chroma_delete_removes_chunks(tmp_path: Path) -> None:
    backend = ChromaBackend(tmp_path)
    backend.upsert([_chunk("a"), _chunk("b")], [_embed(1.0), _embed(2.0)])
    backend.delete(["a"])
    assert backend.count() == 1


def test_chroma_delete_empty_list_is_noop(tmp_path: Path) -> None:
    backend = ChromaBackend(tmp_path)
    backend.upsert([_chunk("a")], [_embed(1.0)])
    backend.delete([])
    assert backend.count() == 1


def test_chroma_list_sources_groups_by_file(tmp_path: Path) -> None:
    backend = ChromaBackend(tmp_path)
    chunks = [
        _chunk("a1", file_id="f1", file_name="file1.txt", idx=0),
        _chunk("a2", file_id="f1", file_name="file1.txt", idx=1),
        _chunk("b1", file_id="f2", file_name="file2.txt", idx=0),
    ]
    backend.upsert(chunks, [_embed(1.0), _embed(2.0), _embed(3.0)])

    sources = backend.list_sources()
    by_id = {s.file_id: s for s in sources}

    assert len(sources) == 2
    assert by_id["f1"].chunk_count == 2
    assert by_id["f2"].chunk_count == 1
    assert by_id["f1"].file_name == "file1.txt"


def test_chroma_list_sources_empty(tmp_path: Path) -> None:
    backend = ChromaBackend(tmp_path)
    assert backend.list_sources() == []


def test_chroma_source_url_roundtrip(tmp_path: Path) -> None:
    backend = ChromaBackend(tmp_path)
    chunk = Chunk(
        id="x",
        text="text",
        file_id="f1",
        file_name="doc.txt",
        chunk_index=0,
        token_count=1,
        modified_time="2024-01-01T00:00:00.000Z",
        source_url=None,
    )
    backend.upsert([chunk], [_embed(1.0)])
    results = backend.query(_embed(1.0), top_k=1)
    assert results[0].chunk.source_url is None


# ── factory ───────────────────────────────────────────────────────────────────

def test_get_backend_returns_chroma(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    cfg = AnchorConfig(drive_folder_id="x", vector_backend="chroma", state_dir=tmp_path)
    backend = get_backend(cfg)
    assert isinstance(backend, ChromaBackend)


# ── Pinecone (skipped without API key) ───────────────────────────────────────

_HAS_PINECONE = False
try:
    import pinecone as _pinecone_pkg  # noqa: F401

    _HAS_PINECONE = True
except ImportError:
    pass

@pytest.mark.skipif(
    not _HAS_PINECONE or not os.environ.get("PINECONE_API_KEY"),
    reason="pinecone-client not installed or PINECONE_API_KEY not set",
)
def test_pinecone_backend_roundtrip() -> None:
    from anchor_mcp.backends.pinecone_backend import PineconeBackend

    backend = PineconeBackend()
    chunk = _chunk("test-chunk-sprint4")
    embedding = _embed(1.0, dim=1024)
    backend.upsert([chunk], [embedding])
    results = backend.query(embedding, top_k=1)
    assert len(results) >= 1
    backend.delete(["test-chunk-sprint4"])
