from pathlib import Path
from unittest.mock import MagicMock

import pytest

from anchor_mcp.state_store import GCSStateStore, LocalStateStore, get_state_store

# ── LocalStateStore ───────────────────────────────────────────────────────────


def test_local_read_nonexistent_returns_none(tmp_path: Path) -> None:
    store = LocalStateStore(tmp_path)
    assert store.read("missing.json") is None


def test_local_round_trip(tmp_path: Path) -> None:
    store = LocalStateStore(tmp_path)
    data = b'{"test": true}'
    store.write("state.json", data)
    assert store.read("state.json") == data


def test_local_write_creates_parent_dirs(tmp_path: Path) -> None:
    store = LocalStateStore(tmp_path / "nested" / "dir")
    store.write("file.json", b"data")
    assert (tmp_path / "nested" / "dir" / "file.json").exists()


def test_local_write_is_atomic(tmp_path: Path) -> None:
    store = LocalStateStore(tmp_path)
    store.write("state.json", b"v1")
    store.write("state.json", b"v2")
    assert store.read("state.json") == b"v2"
    assert not list(tmp_path.glob("*.tmp"))


# ── GCSStateStore (bypass __init__ to avoid google-cloud-storage import) ─────


def _make_gcs_store(bucket: MagicMock) -> GCSStateStore:
    """Build a GCSStateStore with _bucket injected, skipping __init__."""
    store = object.__new__(GCSStateStore)
    store._bucket = bucket  # type: ignore[attr-defined]
    return store


def test_gcs_read_existing_blob() -> None:
    mock_blob = MagicMock()
    mock_blob.exists.return_value = True
    mock_blob.download_as_bytes.return_value = b'{"key": "val"}'
    mock_bucket = MagicMock()
    mock_bucket.blob.return_value = mock_blob

    store = _make_gcs_store(mock_bucket)
    result = store.read("file_registry.json")

    mock_bucket.blob.assert_called_once_with("file_registry.json")
    assert result == b'{"key": "val"}'


def test_gcs_read_nonexistent_blob_returns_none() -> None:
    mock_blob = MagicMock()
    mock_blob.exists.return_value = False
    mock_bucket = MagicMock()
    mock_bucket.blob.return_value = mock_blob

    store = _make_gcs_store(mock_bucket)
    assert store.read("missing.json") is None


def test_gcs_write_uploads_bytes() -> None:
    mock_blob = MagicMock()
    mock_bucket = MagicMock()
    mock_bucket.blob.return_value = mock_blob

    store = _make_gcs_store(mock_bucket)
    store.write("sync_state.json", b'{"files": {}}')

    mock_bucket.blob.assert_called_once_with("sync_state.json")
    mock_blob.upload_from_string.assert_called_once_with(b'{"files": {}}')


# ── get_state_store factory ───────────────────────────────────────────────────


def test_get_state_store_returns_local_when_no_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("GCS_BUCKET", raising=False)
    from anchor_mcp.config import AnchorConfig

    cfg = AnchorConfig(drive_folder_id="x", state_dir=tmp_path)
    store = get_state_store(cfg)
    assert isinstance(store, LocalStateStore)
    assert store._base == tmp_path / "cache"  # type: ignore[attr-defined]
