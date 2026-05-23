from pathlib import Path
from unittest.mock import MagicMock

from anchor_mcp.drive import DriveFile
from anchor_mcp.embed import HybridEmbedding, SparseValues
from anchor_mcp.state_store import LocalStateStore
from anchor_mcp.sync import FileRegistry, FileState, RegistryEntry, Syncer, SyncState

# ── helpers ───────────────────────────────────────────────────────────────────


def _file(
    file_id: str,
    name: str = "doc.txt",
    modified: str = "2024-01-01T00:00:00.000Z",
    mime: str = "text/plain",
) -> DriveFile:
    return DriveFile(id=file_id, name=name, mime_type=mime, modified_time=modified)


def _hybrid_embeddings(n: int) -> list[HybridEmbedding]:
    return [
        HybridEmbedding(
            dense=[0.1] * 1024,
            sparse=SparseValues(indices=[i], values=[0.5]),
        )
        for i in range(n)
    ]


def _make_syncer(
    tmp_path: Path,
    drive: MagicMock,
    content: bytes = b"some text content",
) -> tuple[Syncer, MagicMock, LocalStateStore]:
    backend = MagicMock()
    backend.count.return_value = 0
    backend.query.return_value = []

    embedder = MagicMock()
    embedder.embed_chunks.side_effect = lambda chunks: _hybrid_embeddings(len(chunks))

    drive.download_file.return_value = content

    store = LocalStateStore(tmp_path / "cache")
    state = SyncState.load(store)

    syncer = Syncer(
        drive=drive,
        embedder=embedder,
        backend=backend,
        state=state,
        store=store,
        chunk_size=800,
        chunk_overlap=100,
    )
    return syncer, backend, store


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
    assert backend.upsert.call_count == 2


def test_cold_sync_persists_state(tmp_path: Path) -> None:
    drive = MagicMock()
    drive.list_files.return_value = [_file("f1", "a.txt")]
    syncer, _, store = _make_syncer(tmp_path, drive)
    syncer.sync("folder123")

    loaded = SyncState.load(store)
    assert "f1" in loaded.files
    assert loaded.files["f1"].modified_time == "2024-01-01T00:00:00.000Z"
    assert len(loaded.files["f1"].chunk_ids) >= 1


# ── no changes ────────────────────────────────────────────────────────────────


def test_second_sync_skips_unchanged_files(tmp_path: Path) -> None:
    drive = MagicMock()
    drive.list_files.return_value = [_file("f1"), _file("f2")]
    syncer, backend, _ = _make_syncer(tmp_path, drive)

    # After first sync, mark backend as having vectors
    backend.count.return_value = 2

    syncer.sync("folder123")
    report = syncer.sync("folder123")

    assert report.added == 0
    assert report.skipped == 2


# ── file modified ─────────────────────────────────────────────────────────────


def test_modified_file_replaces_old_chunks(tmp_path: Path) -> None:
    drive = MagicMock()
    file_v1 = _file("f1", modified="2024-01-01T00:00:00.000Z")
    file_v2 = _file("f1", modified="2024-06-01T00:00:00.000Z")

    drive.list_files.return_value = [file_v1]
    syncer, backend, store = _make_syncer(tmp_path, drive, b"version one")
    syncer.sync("folder123")

    backend.count.return_value = 1
    old_chunk_ids = SyncState.load(store).files["f1"].chunk_ids

    drive.list_files.return_value = [file_v2]
    drive.download_file.return_value = b"version two - completely different"
    syncer.sync("folder123")

    new_chunk_ids = SyncState.load(store).files["f1"].chunk_ids
    assert new_chunk_ids != old_chunk_ids
    backend.delete.assert_called_with(old_chunk_ids)


def test_modified_file_updates_state(tmp_path: Path) -> None:
    drive = MagicMock()
    drive.list_files.return_value = [_file("f1", modified="2024-01-01T00:00:00.000Z")]
    syncer, backend, store = _make_syncer(tmp_path, drive)
    syncer.sync("folder123")

    backend.count.return_value = 1
    drive.list_files.return_value = [_file("f1", modified="2024-06-01T00:00:00.000Z")]
    drive.download_file.return_value = b"new content"
    report = syncer.sync("folder123")

    assert report.updated == 1
    loaded = SyncState.load(store)
    assert loaded.files["f1"].modified_time == "2024-06-01T00:00:00.000Z"


# ── file deleted ──────────────────────────────────────────────────────────────


def test_deleted_file_removes_chunks(tmp_path: Path) -> None:
    drive = MagicMock()
    drive.list_files.return_value = [_file("f1"), _file("f2")]
    syncer, backend, store = _make_syncer(tmp_path, drive)
    syncer.sync("folder123")

    backend.count.return_value = 2
    drive.list_files.return_value = [_file("f2")]
    report = syncer.sync("folder123")

    assert report.deleted == 1
    assert report.skipped == 1
    assert "f1" not in SyncState.load(store).files


# ── error handling ────────────────────────────────────────────────────────────


def test_per_file_error_does_not_abort_sync(tmp_path: Path) -> None:
    drive = MagicMock()
    drive.list_files.return_value = [_file("f1", "good.txt"), _file("f2", "bad.txt")]

    def _download(file_id: str, mime_type: str) -> bytes:
        if file_id == "f2":
            raise RuntimeError("network error")
        return b"good content"

    drive.download_file.side_effect = _download
    syncer, backend, _ = _make_syncer(tmp_path, drive)

    report = syncer.sync("folder123")

    assert report.added == 1
    assert len(report.errors) == 1
    assert "bad.txt" in report.errors[0]


def test_failed_file_not_saved_to_state(tmp_path: Path) -> None:
    drive = MagicMock()
    drive.list_files.return_value = [_file("f1", "bad.txt")]
    drive.download_file.side_effect = RuntimeError("download failed")
    syncer, _, store = _make_syncer(tmp_path, drive)

    syncer.sync("folder123")

    loaded = SyncState.load(store)
    assert "f1" not in loaded.files


# ── progress callback ─────────────────────────────────────────────────────────


def test_progress_callback_called_for_adds(tmp_path: Path) -> None:
    drive = MagicMock()
    drive.list_files.return_value = [_file("f1", "doc.txt")]
    syncer, _, _ = _make_syncer(tmp_path, drive)

    messages: list[str] = []
    syncer.sync("folder123", progress_cb=messages.append)

    assert any("doc.txt" in m for m in messages)


# ── SyncState persistence via StateStore ──────────────────────────────────────


def test_sync_state_load_missing_returns_empty(tmp_path: Path) -> None:
    store = LocalStateStore(tmp_path)
    state = SyncState.load(store)
    assert state.files == {}


def test_sync_state_save_load_roundtrip(tmp_path: Path) -> None:
    store = LocalStateStore(tmp_path)
    state = SyncState(
        files={"f1": FileState(modified_time="2024-01-01T00:00:00.000Z", chunk_ids=["c1", "c2"])}
    )
    state.save(store)
    loaded = SyncState.load(store)
    assert loaded.files["f1"].chunk_ids == ["c1", "c2"]


# ── FileRegistry persistence via StateStore ───────────────────────────────────


def test_file_registry_load_missing_returns_empty(tmp_path: Path) -> None:
    store = LocalStateStore(tmp_path)
    registry = FileRegistry.load(store)
    assert registry.files == {}


def test_file_registry_save_load_roundtrip(tmp_path: Path) -> None:
    store = LocalStateStore(tmp_path)
    registry = FileRegistry(
        files={"f1": RegistryEntry(file_name="doc.txt", chunk_count=3, modified_time="2024-01-01")}
    )
    registry.save(store)
    loaded = FileRegistry.load(store)
    assert loaded.files["f1"].file_name == "doc.txt"
    assert loaded.files["f1"].chunk_count == 3


def test_to_registry_derives_from_sync_state() -> None:
    state = SyncState(
        files={
            "f1": FileState(file_name="a.txt", modified_time="2024-01", chunk_ids=["c1", "c2"]),
            "f2": FileState(file_name="b.txt", modified_time="2024-02", chunk_ids=["c3"]),
        }
    )
    registry = state.to_registry()
    assert registry.files["f1"].chunk_count == 2
    assert registry.files["f2"].chunk_count == 1
    assert registry.files["f1"].file_name == "a.txt"
