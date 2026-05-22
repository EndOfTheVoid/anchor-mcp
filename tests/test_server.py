import math
from pathlib import Path
from unittest.mock import MagicMock

import pytest

import anchor_mcp.server as srv
from anchor_mcp.backends.base import SourceInfo
from anchor_mcp.backends.chroma_backend import ChromaBackend
from anchor_mcp.chunk import Chunk
from anchor_mcp.config import AnchorConfig
from anchor_mcp.embed import Embedder
from anchor_mcp.server import DocumentView, SearchResult

# ── fixtures ──────────────────────────────────────────────────────────────────

def _unit_vec(seed: float, dim: int = 4) -> list[float]:
    vec = [seed + i for i in range(dim)]
    norm = math.sqrt(sum(x * x for x in vec))
    return [x / norm for x in vec]


def _chunk(chunk_id: str, file_id: str = "f1", idx: int = 0, text: str = "chunk text") -> Chunk:
    return Chunk(
        id=chunk_id,
        text=text,
        file_id=file_id,
        file_name="doc.txt",
        chunk_index=idx,
        token_count=2,
        modified_time="2024-01-01T00:00:00.000Z",
        source_url="https://drive.google.com/file/d/f1/view",
    )


@pytest.fixture(autouse=True)
def reset_server_state(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reset module-level singletons between tests."""
    monkeypatch.setattr(srv, "_config", None)
    monkeypatch.setattr(srv, "_embedder", None)
    monkeypatch.setattr(srv, "_backend", None)


@pytest.fixture()
def injected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[AnchorConfig, MagicMock, ChromaBackend]:
    config = AnchorConfig(drive_folder_id="folder1", state_dir=tmp_path)
    mock_embedder = MagicMock(spec=Embedder)
    mock_embedder.embed_query.return_value = _unit_vec(1.0)
    backend = ChromaBackend(tmp_path)

    monkeypatch.setattr(srv, "_config", config)
    monkeypatch.setattr(srv, "_embedder", mock_embedder)
    monkeypatch.setattr(srv, "_backend", backend)

    return config, mock_embedder, backend


# ── search ────────────────────────────────────────────────────────────────────

def test_search_returns_search_results(
    injected: tuple[AnchorConfig, MagicMock, ChromaBackend],
) -> None:
    _, mock_embedder, backend = injected
    emb = _unit_vec(1.0)
    backend.upsert([_chunk("c1"), _chunk("c2", file_id="f2")], [emb, _unit_vec(2.0)])

    results = srv.search("my query", top_k=2)

    assert isinstance(results, list)
    assert all(isinstance(r, SearchResult) for r in results)
    assert len(results) <= 2
    mock_embedder.embed_query.assert_called_once_with("my query")


def test_search_top_k_limits_results(
    injected: tuple[AnchorConfig, MagicMock, ChromaBackend],
) -> None:
    _, _, backend = injected
    for i in range(5):
        backend.upsert([_chunk(f"c{i}")], [_unit_vec(float(i))])

    results = srv.search("query", top_k=2)
    assert len(results) == 2


def test_search_empty_backend_returns_empty(
    injected: tuple[AnchorConfig, MagicMock, ChromaBackend],
) -> None:
    results = srv.search("query")
    assert results == []


def test_search_result_fields(
    injected: tuple[AnchorConfig, MagicMock, ChromaBackend],
) -> None:
    _, _, backend = injected
    backend.upsert([_chunk("c1", text="interesting content")], [_unit_vec(1.0)])

    results = srv.search("query", top_k=1)
    r = results[0]

    assert r.file_id == "f1"
    assert r.file_name == "doc.txt"
    assert r.chunk_index == 0
    assert isinstance(r.relevance_score, float)
    assert r.source_url is not None


def test_search_file_name_filter(
    injected: tuple[AnchorConfig, MagicMock, ChromaBackend],
) -> None:
    _, _, backend = injected
    c1 = Chunk(id="c1", text="t", file_id="f1", file_name="report.txt",
               chunk_index=0, token_count=1, modified_time="2024-01-01T00:00:00.000Z",
               source_url=None)
    c2 = Chunk(id="c2", text="t", file_id="f2", file_name="notes.txt",
               chunk_index=0, token_count=1, modified_time="2024-01-01T00:00:00.000Z",
               source_url=None)
    backend.upsert([c1, c2], [_unit_vec(1.0), _unit_vec(2.0)])

    results = srv.search("query", top_k=5, file_name_filter="report.txt")
    assert all(r.file_name == "report.txt" for r in results)


# ── get_document ──────────────────────────────────────────────────────────────

def test_get_document_returns_document_view(
    injected: tuple[AnchorConfig, MagicMock, ChromaBackend],
) -> None:
    _, _, backend = injected
    chunks = [
        _chunk("c1", idx=0, text="First chunk."),
        _chunk("c2", idx=1, text="Second chunk."),
    ]
    backend.upsert(chunks, [_unit_vec(1.0), _unit_vec(2.0)])

    doc = srv.get_document("f1")

    assert isinstance(doc, DocumentView)
    assert doc.file_id == "f1"
    assert doc.chunk_count == 2
    assert "First chunk." in doc.text
    assert "Second chunk." in doc.text


def test_get_document_chunks_in_order(
    injected: tuple[AnchorConfig, MagicMock, ChromaBackend],
) -> None:
    _, _, backend = injected
    # Insert in reverse order to verify sort
    chunks = [_chunk("c2", idx=1, text="B"), _chunk("c1", idx=0, text="A")]
    backend.upsert(chunks, [_unit_vec(1.0), _unit_vec(2.0)])

    doc = srv.get_document("f1")
    assert doc.text.index("A") < doc.text.index("B")


def test_get_document_missing_file_raises(
    injected: tuple[AnchorConfig, MagicMock, ChromaBackend],
) -> None:
    from anchor_mcp.errors import BackendError

    with pytest.raises(BackendError):
        srv.get_document("nonexistent_file_id")


# ── list_sources ──────────────────────────────────────────────────────────────

def test_list_sources_returns_source_infos(
    injected: tuple[AnchorConfig, MagicMock, ChromaBackend],
) -> None:
    _, _, backend = injected
    c1 = Chunk(id="a1", text="t", file_id="f1", file_name="report.txt",
               chunk_index=0, token_count=1, modified_time="2024-01-01T00:00:00.000Z",
               source_url=None)
    c2 = Chunk(id="b1", text="t", file_id="f2", file_name="notes.txt",
               chunk_index=0, token_count=1, modified_time="2024-01-01T00:00:00.000Z",
               source_url=None)
    backend.upsert([c1, c2], [_unit_vec(1.0), _unit_vec(2.0)])

    sources = srv.list_sources()

    assert isinstance(sources, list)
    assert all(isinstance(s, SourceInfo) for s in sources)
    assert len(sources) == 2


def test_list_sources_name_filter(
    injected: tuple[AnchorConfig, MagicMock, ChromaBackend],
) -> None:
    _, _, backend = injected
    c1 = Chunk(id="a1", text="t", file_id="f1", file_name="ISB_Strategy.pdf",
               chunk_index=0, token_count=1, modified_time="2024-01-01T00:00:00.000Z",
               source_url=None)
    c2 = Chunk(id="b1", text="t", file_id="f2", file_name="random_notes.txt",
               chunk_index=0, token_count=1, modified_time="2024-01-01T00:00:00.000Z",
               source_url=None)
    backend.upsert([c1, c2], [_unit_vec(1.0), _unit_vec(2.0)])

    results = srv.list_sources(name_filter="ISB")
    assert len(results) == 1
    assert results[0].file_name == "ISB_Strategy.pdf"


def test_list_sources_empty(
    injected: tuple[AnchorConfig, MagicMock, ChromaBackend],
) -> None:
    assert srv.list_sources() == []


# ── lazy initialization ───────────────────────────────────────────────────────

def test_ensure_initialized_uses_injected_values(
    injected: tuple[AnchorConfig, MagicMock, ChromaBackend],
) -> None:
    config, embedder, backend = injected
    c, e, b = srv._ensure_initialized()
    assert c is config
    assert e is embedder
    assert b is backend
