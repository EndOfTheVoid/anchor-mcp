from pathlib import Path
from unittest.mock import MagicMock

import pytest

import anchor_mcp.server as srv
from anchor_mcp.backends.base import QueryResult
from anchor_mcp.chunk import Chunk
from anchor_mcp.config import AnchorConfig
from anchor_mcp.embed import HybridEmbedding, SparseValues
from anchor_mcp.server import DocumentView, SearchResult, SourceInfo
from anchor_mcp.state_store import LocalStateStore
from anchor_mcp.sync import FileRegistry, RegistryEntry

# ── helpers ───────────────────────────────────────────────────────────────────


def _hybrid() -> HybridEmbedding:
    return HybridEmbedding(
        dense=[0.1] * 1024,
        sparse=SparseValues(indices=[1, 2], values=[0.5, 0.3]),
    )


def _chunk(
    chunk_id: str = "c1",
    file_id: str = "f1",
    file_name: str = "doc.txt",
    idx: int = 0,
    text: str = "chunk text",
) -> Chunk:
    return Chunk(
        id=chunk_id,
        text=text,
        file_id=file_id,
        file_name=file_name,
        chunk_index=idx,
        token_count=2,
        modified_time="2024-01-01T00:00:00.000Z",
        source_url="https://drive.google.com/file/d/f1/view",
    )


def _query_result(chunk_id: str = "c1", score: float = 0.9, **kw: object) -> QueryResult:
    return QueryResult(chunk=_chunk(chunk_id, **kw), score=score)  # type: ignore[arg-type]


# ── fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def reset_server_state(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(srv, "_config", None)
    monkeypatch.setattr(srv, "_embedder", None)
    monkeypatch.setattr(srv, "_backend", None)
    monkeypatch.setattr(srv, "_state_store", None)


@pytest.fixture()
def injected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[AnchorConfig, MagicMock, MagicMock, LocalStateStore]:
    config = AnchorConfig(drive_folder_id="folder1", state_dir=tmp_path)

    mock_embedder = MagicMock()
    mock_embedder.embed_query.return_value = _hybrid()

    mock_backend = MagicMock()
    mock_backend.query.return_value = []
    mock_backend.get_chunks_by_file.return_value = []
    mock_backend.count.return_value = 0

    store = LocalStateStore(tmp_path / "cache")

    monkeypatch.setattr(srv, "_config", config)
    monkeypatch.setattr(srv, "_embedder", mock_embedder)
    monkeypatch.setattr(srv, "_backend", mock_backend)
    monkeypatch.setattr(srv, "_state_store", store)

    return config, mock_embedder, mock_backend, store


# ── search ────────────────────────────────────────────────────────────────────


def test_search_calls_embedder_and_backend(
    injected: tuple[AnchorConfig, MagicMock, MagicMock, LocalStateStore],
) -> None:
    config, mock_embedder, mock_backend, _ = injected
    mock_backend.query.return_value = [_query_result()]

    results = srv.search("my query", top_k=1)

    mock_embedder.embed_query.assert_called_once_with("my query")
    mock_backend.query.assert_called_once()
    assert len(results) == 1
    assert isinstance(results[0], SearchResult)


def test_search_uses_config_alpha_by_default(
    injected: tuple[AnchorConfig, MagicMock, MagicMock, LocalStateStore],
) -> None:
    config, _, mock_backend, _ = injected
    mock_backend.query.return_value = []
    config.search_alpha = 0.5

    srv.search("query")

    call_kwargs = mock_backend.query.call_args.kwargs
    assert call_kwargs["alpha"] == 0.5


def test_search_alpha_param_overrides_config(
    injected: tuple[AnchorConfig, MagicMock, MagicMock, LocalStateStore],
) -> None:
    _, _, mock_backend, _ = injected
    mock_backend.query.return_value = []

    srv.search("query", alpha=0.2)

    call_kwargs = mock_backend.query.call_args.kwargs
    assert call_kwargs["alpha"] == 0.2


def test_search_result_has_chunk_id(
    injected: tuple[AnchorConfig, MagicMock, MagicMock, LocalStateStore],
) -> None:
    _, _, mock_backend, _ = injected
    mock_backend.query.return_value = [_query_result(chunk_id="chunk-abc")]

    results = srv.search("query", top_k=1)
    assert results[0].chunk_id == "chunk-abc"


def test_search_forwards_file_name_filter(
    injected: tuple[AnchorConfig, MagicMock, MagicMock, LocalStateStore],
) -> None:
    _, _, mock_backend, _ = injected
    mock_backend.query.return_value = []

    srv.search("query", file_name_filter="report.pdf")

    call_kwargs = mock_backend.query.call_args.kwargs
    assert call_kwargs["file_name_filter"] == "report.pdf"


def test_search_empty_returns_empty_list(
    injected: tuple[AnchorConfig, MagicMock, MagicMock, LocalStateStore],
) -> None:
    _, _, mock_backend, _ = injected
    mock_backend.query.return_value = []
    assert srv.search("query") == []


# ── get_document ──────────────────────────────────────────────────────────────


def test_get_document_concatenates_chunks(
    injected: tuple[AnchorConfig, MagicMock, MagicMock, LocalStateStore],
) -> None:
    _, _, mock_backend, _ = injected
    mock_backend.get_chunks_by_file.return_value = [
        _chunk("c1", idx=0, text="First chunk."),
        _chunk("c2", idx=1, text="Second chunk."),
    ]

    doc = srv.get_document("f1")

    assert isinstance(doc, DocumentView)
    assert doc.file_id == "f1"
    assert doc.chunk_count == 2
    assert "First chunk." in doc.text
    assert "Second chunk." in doc.text


def test_get_document_missing_raises(
    injected: tuple[AnchorConfig, MagicMock, MagicMock, LocalStateStore],
) -> None:
    from anchor_mcp.errors import BackendError

    _, _, mock_backend, _ = injected
    mock_backend.get_chunks_by_file.return_value = []

    with pytest.raises(BackendError):
        srv.get_document("nonexistent")


# ── list_sources ──────────────────────────────────────────────────────────────


def _write_registry(store: LocalStateStore, entries: dict[str, RegistryEntry]) -> None:
    registry = FileRegistry(files=entries)
    registry.save(store)


def test_list_sources_returns_from_registry(
    injected: tuple[AnchorConfig, MagicMock, MagicMock, LocalStateStore],
) -> None:
    _, _, _, store = injected
    _write_registry(
        store,
        {
            "f1": RegistryEntry(file_name="report.pdf", chunk_count=3, modified_time="2024-01"),
            "f2": RegistryEntry(file_name="notes.txt", chunk_count=1, modified_time="2024-02"),
        },
    )

    sources = srv.list_sources()
    assert len(sources) == 2
    assert all(isinstance(s, SourceInfo) for s in sources)


def test_list_sources_name_filter(
    injected: tuple[AnchorConfig, MagicMock, MagicMock, LocalStateStore],
) -> None:
    _, _, _, store = injected
    _write_registry(
        store,
        {
            "f1": RegistryEntry(
                file_name="ISB_Strategy.pdf", chunk_count=2, modified_time="2024-01"
            ),
            "f2": RegistryEntry(
                file_name="random_notes.txt", chunk_count=1, modified_time="2024-01"
            ),
        },
    )

    results = srv.list_sources(name_filter="ISB")
    assert len(results) == 1
    assert results[0].file_name == "ISB_Strategy.pdf"


def test_list_sources_empty_registry(
    injected: tuple[AnchorConfig, MagicMock, MagicMock, LocalStateStore],
) -> None:
    assert srv.list_sources() == []
