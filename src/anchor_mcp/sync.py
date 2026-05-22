import json
from collections.abc import Callable
from pathlib import Path

from pydantic import BaseModel, Field
from tqdm.auto import tqdm

from anchor_mcp.backends.base import VectorBackend
from anchor_mcp.chunk import chunk_text
from anchor_mcp.drive import DriveClient, DriveFile
from anchor_mcp.embed import Embedder
from anchor_mcp.extract import extract_text


class FileState(BaseModel):
    modified_time: str
    md5_checksum: str | None = None
    chunk_ids: list[str] = Field(default_factory=list)


class SyncState(BaseModel):
    files: dict[str, FileState] = Field(default_factory=dict)

    @classmethod
    def load(cls, path: Path) -> "SyncState":
        if not path.exists():
            return cls()
        raw: object = json.loads(path.read_text(encoding="utf-8"))
        return cls.model_validate(raw)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(self.model_dump_json(indent=2), encoding="utf-8")
        tmp.replace(path)


class SyncReport(BaseModel):
    added: int = 0
    updated: int = 0
    deleted: int = 0
    skipped: int = 0
    errors: list[str] = Field(default_factory=list)


class Syncer:
    def __init__(
        self,
        drive: DriveClient,
        embedder: Embedder,
        backend: VectorBackend,
        state: SyncState,
        state_path: Path,
        chunk_size: int = 800,
        chunk_overlap: int = 100,
    ) -> None:
        self._drive = drive
        self._embedder = embedder
        self._backend = backend
        self._state = state
        self._state_path = state_path
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

    def sync(
        self,
        folder_id: str,
        progress_cb: Callable[[str], None] | None = None,
        show_progress: bool = False,
    ) -> SyncReport:
        report = SyncReport()
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
            modified_time=file.modified_time,
            md5_checksum=file.md5_checksum,
            chunk_ids=[c.id for c in chunks],
        )
        self._state.save(self._state_path)

    def _update_file(self, file: DriveFile, progress_cb: Callable[[str], None] | None) -> None:
        if progress_cb:
            progress_cb(f"Updating {file.name}")
        # Compute new content first — safe to fail here, nothing changed yet
        raw = self._drive.download_file(file.id, file.mime_type)
        text = extract_text(file, raw)
        new_chunks = chunk_text(text, file, self._chunk_size, self._chunk_overlap)
        new_embeddings = self._embedder.embed_chunks(new_chunks)
        # Replace old with new
        self._backend.delete(self._state.files[file.id].chunk_ids)
        self._backend.upsert(new_chunks, new_embeddings)
        self._state.files[file.id] = FileState(
            modified_time=file.modified_time,
            md5_checksum=file.md5_checksum,
            chunk_ids=[c.id for c in new_chunks],
        )
        self._state.save(self._state_path)

    def _delete_file(self, file_id: str) -> None:
        self._backend.delete(self._state.files[file_id].chunk_ids)
        del self._state.files[file_id]
        self._state.save(self._state_path)
