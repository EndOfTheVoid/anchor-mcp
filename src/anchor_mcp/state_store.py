import os
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from anchor_mcp.config import AnchorConfig


@runtime_checkable
class StateStore(Protocol):
    def read(self, key: str) -> bytes | None: ...
    def write(self, key: str, data: bytes) -> None: ...


class LocalStateStore:
    def __init__(self, base_dir: Path) -> None:
        self._base = base_dir

    def read(self, key: str) -> bytes | None:
        path = self._base / key
        if not path.exists():
            return None
        return path.read_bytes()

    def write(self, key: str, data: bytes) -> None:
        path = self._base / key
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_bytes(data)
        tmp.replace(path)


class GCSStateStore:
    def __init__(self, bucket_name: str) -> None:
        try:
            from google.cloud import storage  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "google-cloud-storage is not installed. "
                "Install with: pip install google-cloud-storage"
            ) from exc
        client = storage.Client()
        self._bucket = client.bucket(bucket_name)

    def read(self, key: str) -> bytes | None:
        blob = self._bucket.blob(key)
        if not blob.exists():
            return None
        return blob.download_as_bytes()  # type: ignore[no-any-return]

    def write(self, key: str, data: bytes) -> None:
        blob = self._bucket.blob(key)
        blob.upload_from_string(data)


def get_state_store(config: "AnchorConfig") -> StateStore:
    bucket = os.environ.get("GCS_BUCKET")
    if bucket:
        return GCSStateStore(bucket)
    return LocalStateStore(config.state_dir / "cache")
