import json
from collections.abc import Callable

from pydantic import BaseModel, Field
from tqdm.auto import tqdm

from anchor_mcp.backends.base import VectorBackend
from anchor_mcp.chunk import chunk_text
from anchor_mcp.drive import DriveClient, DriveFile
from anchor_mcp.embed import PineconeEmbedder
from anchor_mcp.extract import extract_text
from anchor_mcp.state_store import StateStore

# ── file registry (sidecar) ───────────────────────────────────────────────────


class RegistryEntry(BaseModel):
    file_name: str
    chunk_count: int
    modified_time: str


class FileRegistry(BaseModel):
    """Portable sidecar that lists every indexed file without touching the vector backend."""

    files: dict[str, RegistryEntry] = Field(default_factory=dict)

    @classmethod
    def load(cls, store: StateStore) -> "FileRegistry":
        data = store.read("file_registry.json")
        if data is None:
            return cls()
        raw: object = json.loads(data)
        return cls.model_validate(raw)

    def save(self, store: StateStore) -> None:
        store.write("file_registry.json", self.model_dump_json(indent=2).encode())


# ── sync state ────────────────────────────────────────────────────────────────


class FileState(BaseModel):
    file_name: str = ""
    modified_time: str
    md5_checksum: str | None = None
    chunk_ids: list[str] = Field(default_factory=list)


class SyncState(BaseModel):
    files: dict[str, FileState] = Field(default_factory=dict)

    @classmethod
    def load(cls, store: StateStore) -> "SyncState":
        data = store.read("sync_state.json")
        if data is None:
            return cls()
        raw: object = json.loads(data)
        return cls.model_validate(raw)

    def save(self, store: StateStore) -> None:
        store.write("sync_state.json", self.model_dump_json(indent=2).encode())

    def to_registry(self) -> FileRegistry:
        """Derive a FileRegistry from the current sync state (for migration)."""
        return FileRegistry(
            files={
                fid: RegistryEntry(
                    file_name=fs.file_name,
                    chunk_count=len(fs.chunk_ids),
                    modified_time=fs.modified_time,
                )
                for fid, fs in self.files.items()
                if fs.file_name
            }
        )


# ── sync report ───────────────────────────────────────────────────────────────


class SyncReport(BaseModel):
    added: int = 0
    updated: int = 0
    deleted: int = 0
    skipped: int = 0
    errors: list[str] = Field(default_factory=list)


# ── syncer ────────────────────────────────────────────────────────────────────


class Syncer:
    def __init__(
        self,
        drive: DriveClient,
        embedder: PineconeEmbedder,
        backend: VectorBackend,
        state: SyncState,
        store: StateStore,
        chunk_size: int = 800,
        chunk_overlap: int = 100,
    ) -> None:
        self._drive = drive
        self._embedder = embedder
        self._backend = backend
        self._state = state
        self._store = store
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

    def sync(
        self,
        folder_id: str,
        progress_cb: Callable[[str], None] | None = None,
        show_progress: bool = False,
    ) -> SyncReport:
        report = SyncReport()

        # If the backend is empty but sync state has entries, the user likely switched
        # backends or wiped Pinecone. Force a full re-sync.
        if self._state.files and self._backend.count() == 0:
            self._state = SyncState()

        drive_files = self._drive.list_files(folder_id)
        drive_ids = {f.id for f in drive_files}

        pbar = tqdm(
            drive_files,
            desc="Files",
            unit="file",
            disable=not show_progress,
            dynamic_ncols=True,
        )

        for file in pbar:
            if show_progress:
                pbar.set_postfix_str(file.name[:50])
            try:
                if file.id not in self._state.files:
                    self._add_file(file, progress_cb)
                    report.added += 1
                elif self._state.files[file.id].modified_time != file.modified_time:
                    self._update_file(file, progress_cb)
                    report.updated += 1
                else:
                    report.skipped += 1
            except Exception as exc:
                report.errors.append(f"{file.name}: {exc}")

        for file_id in list(self._state.files):
            if file_id not in drive_ids:
                try:
                    self._delete_file(file_id)
                    report.deleted += 1
                except Exception as exc:
                    report.errors.append(f"delete {file_id}: {exc}")

        self._state.to_registry().save(self._store)

        return report

    # ── private ───────────────────────────────────────────────────────────────

    def _add_file(self, file: DriveFile, progress_cb: Callable[[str], None] | None) -> None:
        if progress_cb:
            progress_cb(f"Adding {file.name}")
        raw = self._drive.download_file(file.id, file.mime_type)
        text = extract_text(file, raw)
        chunks = chunk_text(text, file, self._chunk_size, self._chunk_overlap)
        embeddings = self._embedder.embed_chunks(chunks)
        self._backend.upsert(chunks, embeddings)
        self._state.files[file.id] = FileState(
            file_name=file.name,
            modified_time=file.modified_time,
            md5_checksum=file.md5_checksum,
            chunk_ids=[c.id for c in chunks],
        )
        self._state.save(self._store)

    def _update_file(self, file: DriveFile, progress_cb: Callable[[str], None] | None) -> None:
        if progress_cb:
            progress_cb(f"Updating {file.name}")
        raw = self._drive.download_file(file.id, file.mime_type)
        text = extract_text(file, raw)
        new_chunks = chunk_text(text, file, self._chunk_size, self._chunk_overlap)
        new_embeddings = self._embedder.embed_chunks(new_chunks)
        self._backend.delete(self._state.files[file.id].chunk_ids)
        self._backend.upsert(new_chunks, new_embeddings)
        self._state.files[file.id] = FileState(
            file_name=file.name,
            modified_time=file.modified_time,
            md5_checksum=file.md5_checksum,
            chunk_ids=[c.id for c in new_chunks],
        )
        self._state.save(self._store)

    def _delete_file(self, file_id: str) -> None:
        self._backend.delete(self._state.files[file_id].chunk_ids)
        del self._state.files[file_id]
        self._state.save(self._store)
