import math
from pathlib import Path
from unittest.mock import MagicMock

from anchor_mcp.backends.chroma_backend import ChromaBackend
from anchor_mcp.drive import DriveFile
from anchor_mcp.sync import Syncer, SyncState

# ── helpers ───────────────────────────────────────────────────────────────────

def _file(
    file_id: str,
    name: str = "doc.txt",
    modified: str = "2024-01-01T00:00:00.000Z",
    mime: str = "text/plain",
) -> DriveFile:
    return DriveFile(
        id=file_id,
        name=name,
        mime_type=mime,
        modified_time=modified,
    )


def _embed(n: int, dim: int = 4) -> list[list[float]]:
    """Return n unit vectors for use as fake embeddings."""
    result: list[list[float]] = []
    for i in range(n):
        vec = [float(i + j + 1) for j in range(dim)]
        norm = math.sqrt(sum(x * x for x in vec))
        result.append([x / norm for x in vec])
    return result


def _make_syncer(
    tmp_path: Path,
    drive: MagicMock,
    content: bytes = b"some text content",
) -> tuple[Syncer, ChromaBackend, Path]:
    backend = ChromaBackend(tmp_path)
    embedder = MagicMock()
    embedder.embed_chunks.side_effect = lambda chunks: _embed(len(chunks))

    drive.download_file.return_value = content

    state_path = tmp_path / "cache" / "sync_state.json"
    state = SyncState.load(state_path)

    syncer = Syncer(
        drive=drive,
        embedder=embedder,
        backend=backend,
        state=state,
        state_path=state_path,
        chunk_size=800,
        chunk_overlap=100,
    )
    return syncer, backend, state_path


# ── cold sync ─────────────────────────────────────────────────────────────────

def test_cold_sync_adds_all_files(tmp_path: Path) -> None:
    drive = MagicMock()
    drive.list_files.return_value = [_file("f1", "a.txt"), _file("f2", "b.txt")]
    syncer, backend, _ = _make_syncer(tmp_path, drive)

    report = syncer.sync("folder123")

    assert report.added == 2
    assert report.updated == 0
    assert report.skipped == 0
    assert report.deleted == 0
    assert report.errors == []
    assert backend.count() == 2  # one chunk per file


def test_cold_sync_persists_state(tmp_path: Path) -> None:
    drive = MagicMock()
    drive.list_files.return_value = [_file("f1", "a.txt")]
    syncer, _, state_path = _make_syncer(tmp_path, drive)
    syncer.sync("folder123")

    loaded = SyncState.load(state_path)
    assert "f1" in loaded.files
    assert loaded.files["f1"].modified_time == "2024-01-01T00:00:00.000Z"
    assert len(loaded.files["f1"].chunk_ids) == 1


# ── no changes ────────────────────────────────────────────────────────────────

def test_second_sync_skips_unchanged_files(tmp_path: Path) -> None:
    drive = MagicMock()
    drive.list_files.return_value = [_file("f1"), _file("f2")]
    syncer, backend, _ = _make_syncer(tmp_path, drive)

    syncer.sync("folder123")
    report = syncer.sync("folder123")

    assert report.added == 0
    assert report.skipped == 2
    assert backend.count() == 2  # no duplicates added


# ── file modified ─────────────────────────────────────────────────────────────

def test_modified_file_replaces_old_chunks(tmp_path: Path) -> None:
    drive = MagicMock()
    file_v1 = _file("f1", modified="2024-01-01T00:00:00.000Z")
    file_v2 = _file("f1", modified="2024-06-01T00:00:00.000Z")

    drive.list_files.return_value = [file_v1]
    syncer, backend, state_path = _make_syncer(tmp_path, drive, b"version one")
    syncer.sync("folder123")

    old_chunk_ids = SyncState.load(state_path).files["f1"].chunk_ids
    assert backend.count() == 1

    drive.list_files.return_value = [file_v2]
    drive.download_file.return_value = b"version two with different content"
    syncer.sync("folder123")

    new_chunk_ids = SyncState.load(state_path).files["f1"].chunk_ids
    assert backend.count() == 1  # still one chunk, not two
    assert new_chunk_ids != old_chunk_ids  # new chunk ID because text changed

    # Old chunks are gone from the backend
    raw = backend._col.get(ids=old_chunk_ids)
    assert raw["ids"] == []


def test_modified_file_updates_state(tmp_path: Path) -> None:
    drive = MagicMock()
    drive.list_files.return_value = [_file("f1", modified="2024-01-01T00:00:00.000Z")]
    syncer, _, state_path = _make_syncer(tmp_path, drive)
    syncer.sync("folder123")

    drive.list_files.return_value = [_file("f1", modified="2024-06-01T00:00:00.000Z")]
    drive.download_file.return_value = b"new content"
    report = syncer.sync("folder123")

    assert report.updated == 1
    loaded = SyncState.load(state_path)
    assert loaded.files["f1"].modified_time == "2024-06-01T00:00:00.000Z"


# ── file deleted ──────────────────────────────────────────────────────────────

def test_deleted_file_removes_chunks(tmp_path: Path) -> None:
    drive = MagicMock()
    drive.list_files.return_value = [_file("f1"), _file("f2")]
    syncer, backend, state_path = _make_syncer(tmp_path, drive)
    syncer.sync("folder123")
    assert backend.count() == 2

    drive.list_files.return_value = [_file("f2")]  # f1 gone from Drive
    report = syncer.sync("folder123")

    assert report.deleted == 1
    assert report.skipped == 1
    assert backend.count() == 1
    assert "f1" not in SyncState.load(state_path).files


# ── error handling ────────────────────────────────────────────────────────────

def test_per_file_error_does_not_abort_sync(tmp_path: Path) -> None:
    drive = MagicMock()
    drive.list_files.return_value = [_file("f1", "good.txt"), _file("f2", "bad.txt")]

    def _download(file_id: str, mime_type: str) -> bytes:
        if file_id == "f2":
            raise RuntimeError("network error")
        return b"good content"

    drive.download_file.side_effect = _download

    backend = ChromaBackend(tmp_path)
    embedder = MagicMock()
    embedder.embed_chunks.side_effect = lambda chunks: _embed(len(chunks))

    state_path = tmp_path / "cache" / "sync_state.json"
    syncer = Syncer(
        drive=drive,
        embedder=embedder,
        backend=backend,
        state=SyncState.load(state_path),
        state_path=state_path,
    )

    report = syncer.sync("folder123")

    assert report.added == 1
    assert len(report.errors) == 1
    assert "bad.txt" in report.errors[0]
    assert backend.count() == 1  # good file was indexed


def test_failed_file_not_saved_to_state(tmp_path: Path) -> None:
    drive = MagicMock()
    drive.list_files.return_value = [_file("f1", "bad.txt")]
    drive.download_file.side_effect = RuntimeError("download failed")

    backend = ChromaBackend(tmp_path)
    embedder = MagicMock()
    state_path = tmp_path / "cache" / "sync_state.json"
    syncer = Syncer(
        drive=drive,
        embedder=embedder,
        backend=backend,
        state=SyncState.load(state_path),
        state_path=state_path,
    )

    syncer.sync("folder123")

    loaded = SyncState.load(state_path)
    assert "f1" not in loaded.files


# ── progress callback ─────────────────────────────────────────────────────────

def test_progress_callback_called_for_adds(tmp_path: Path) -> None:
    drive = MagicMock()
    drive.list_files.return_value = [_file("f1", "doc.txt")]
    syncer, _, _ = _make_syncer(tmp_path, drive)

    messages: list[str] = []
    syncer.sync("folder123", progress_cb=messages.append)

    assert any("doc.txt" in m for m in messages)


# ── SyncState persistence ─────────────────────────────────────────────────────

def test_sync_state_load_missing_returns_empty(tmp_path: Path) -> None:
    state = SyncState.load(tmp_path / "nonexistent.json")
    assert state.files == {}


def test_sync_state_save_load_roundtrip(tmp_path: Path) -> None:
    from anchor_mcp.sync import FileState

    path = tmp_path / "state.json"
    state = SyncState(
        files={"f1": FileState(modified_time="2024-01-01T00:00:00.000Z", chunk_ids=["c1", "c2"])}
    )
    state.save(path)
    loaded = SyncState.load(path)
    assert loaded.files["f1"].chunk_ids == ["c1", "c2"]
