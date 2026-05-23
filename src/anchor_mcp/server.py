import logging
import sys
import threading
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel

from anchor_mcp.backends.base import VectorBackend
from anchor_mcp.config import AnchorConfig, load_config
from anchor_mcp.embed import PineconeEmbedder
from anchor_mcp.state_store import StateStore

mcp = FastMCP(
    "anchor",
    instructions=(
        "Anchor gives you access to the user's Google Drive knowledge base. "
        "Always cite sources when using information from search results."
    ),
)

# ── singletons ────────────────────────────────────────────────────────────────

_init_lock = threading.Lock()
_config: AnchorConfig | None = None
_embedder: PineconeEmbedder | None = None
_backend: VectorBackend | None = None
_state_store: StateStore | None = None

logger = logging.getLogger("anchor_mcp")


def _setup_logging(log_dir: Path) -> None:
    if logger.handlers:
        return
    log_dir.mkdir(parents=True, exist_ok=True)
    logger.setLevel(logging.DEBUG)

    fh = RotatingFileHandler(str(log_dir / "anchor.log"), maxBytes=5 * 1024 * 1024, backupCount=3)
    fh.setFormatter(logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s"))
    logger.addHandler(fh)

    sh = logging.StreamHandler(sys.stderr)
    sh.setLevel(logging.WARNING)
    logger.addHandler(sh)


def _init_config_unlocked() -> None:
    """Initialize config and state store. Must be called with _init_lock held."""
    global _config, _state_store
    from anchor_mcp.state_store import get_state_store

    config = load_config()
    _config = config
    _state_store = get_state_store(config)
    _setup_logging(config.state_dir / "logs")


def _ensure_config() -> AnchorConfig:
    """Lightweight init — just config + state store. Used by list_sources."""
    global _config
    if _config is not None:
        return _config
    with _init_lock:
        if _config is None:
            _init_config_unlocked()
    return _config  # type: ignore[return-value]


def _ensure_initialized() -> tuple[AnchorConfig, PineconeEmbedder, VectorBackend]:
    """Full init — config + state store + embedder + Pinecone backend."""
    global _embedder, _backend

    if _backend is not None:
        return _config, _embedder, _backend  # type: ignore[return-value]

    with _init_lock:
        if _backend is not None:
            return _config, _embedder, _backend  # type: ignore[return-value]

        if _config is None:
            _init_config_unlocked()

        config = _config
        assert config is not None

        from pinecone import Pinecone  # type: ignore[import-untyped]

        from anchor_mcp import secrets
        from anchor_mcp.backends.pinecone_backend import PineconeBackend
        from anchor_mcp.errors import BackendError

        api_key = secrets.get_pinecone_api_key()
        if not api_key:
            raise BackendError("PINECONE_API_KEY environment variable is not set.")

        pc: Any = Pinecone(api_key=api_key)

        embedder = PineconeEmbedder(pc, config.pinecone_dense_model, config.pinecone_sparse_model)
        _embedder = embedder

        backend = PineconeBackend(pc, config.pinecone_index)
        _backend = backend

    return _config, _embedder, _backend  # type: ignore[return-value]


def start_background_init() -> None:
    """Warm up the Pinecone connection at server startup so the first tool call is fast."""
    t = threading.Thread(target=_ensure_initialized, daemon=True, name="anchor-eager-init")
    t.start()


# ── response models ───────────────────────────────────────────────────────────


class SearchResult(BaseModel):
    chunk_id: str
    text: str
    file_name: str
    file_id: str
    chunk_index: int
    source_url: str | None
    relevance_score: float
    modified_time: str


class DocumentView(BaseModel):
    file_id: str
    file_name: str
    text: str
    chunk_count: int
    modified_time: str
    source_url: str | None


class SourceInfo(BaseModel):
    file_id: str
    file_name: str
    chunk_count: int
    modified_time: str


# ── tools ─────────────────────────────────────────────────────────────────────


@mcp.tool(
    description=(
        "Search the user's Google Drive knowledge base by hybrid semantic + keyword similarity. "
        "Returns up to top_k chunks of text with full source metadata. "
        "alpha controls the blend: 0.0 = keyword-only, 1.0 = semantic-only, default 0.7. "
        "You MUST cite every result you use with the format [file_name, chunk N](source_url). "
        "If results have low relevance scores (below 0.5), tell the user the answer "
        "may not be in their indexed documents."
    )
)
def search(
    query: str,
    top_k: int = 5,
    alpha: float | None = None,
    file_name_filter: str | None = None,
) -> list[SearchResult]:
    config, embedder, backend = _ensure_initialized()
    effective_alpha = alpha if alpha is not None else config.search_alpha
    logger.info(
        "search query=%r top_k=%d alpha=%r filter=%r",
        query,
        top_k,
        effective_alpha,
        file_name_filter,
    )

    embedding = embedder.embed_query(query)
    results = backend.query(
        embedding, top_k=top_k, alpha=effective_alpha, file_name_filter=file_name_filter
    )

    return [
        SearchResult(
            chunk_id=r.chunk.id,
            text=r.chunk.text,
            file_name=r.chunk.file_name,
            file_id=r.chunk.file_id,
            chunk_index=r.chunk.chunk_index,
            source_url=r.chunk.source_url,
            relevance_score=round(r.score, 4),
            modified_time=r.chunk.modified_time,
        )
        for r in results
    ]


@mcp.tool(
    description=(
        "Retrieve the full reconstructed text of a single document by its file_id. "
        "Use this when search returned a useful snippet and you need broader context "
        "from the same file. The file_id comes from search result metadata."
    )
)
def get_document(file_id: str) -> DocumentView:
    _, _, backend = _ensure_initialized()
    logger.info("get_document file_id=%r", file_id)

    chunks = backend.get_chunks_by_file(file_id)
    if not chunks:
        from anchor_mcp.errors import BackendError

        raise BackendError(f"No indexed content found for file_id={file_id!r}.")

    full_text = "\n\n".join(c.text for c in chunks)
    first = chunks[0]
    return DocumentView(
        file_id=file_id,
        file_name=first.file_name,
        text=full_text,
        chunk_count=len(chunks),
        modified_time=first.modified_time,
        source_url=first.source_url,
    )


@mcp.tool(
    description=(
        "List all documents available in the user's indexed Google Drive folder. "
        "Use this when the user asks 'what do you have access to?', wants to know "
        "if a specific document is indexed, or needs to browse available sources. "
        "Optionally filter by file name substring. "
        "This reads a GCS/local sidecar — it is instant and requires no Pinecone call."
    )
)
def list_sources(name_filter: str | None = None) -> list[SourceInfo]:
    from anchor_mcp.sync import FileRegistry

    _ensure_config()
    assert _state_store is not None
    logger.info("list_sources filter=%r", name_filter)

    registry = FileRegistry.load(_state_store)
    return [
        SourceInfo(
            file_id=fid,
            file_name=entry.file_name,
            chunk_count=entry.chunk_count,
            modified_time=entry.modified_time,
        )
        for fid, entry in registry.files.items()
        if not name_filter or name_filter.lower() in entry.file_name.lower()
    ]


@mcp.tool(
    description=(
        "Re-sync the user's Google Drive folder, picking up new, modified, and deleted files. "
        "Only call this when the user explicitly asks to refresh, sync, update, or re-index "
        "their documents. This may take several minutes for large folders. "
        "Returns a summary of what changed."
    )
)
def sync_drive() -> dict[str, object]:
    from anchor_mcp.auth import load_credentials
    from anchor_mcp.drive import DriveClient
    from anchor_mcp.sync import Syncer, SyncState

    config, embedder, backend = _ensure_initialized()
    assert _state_store is not None
    logger.info("sync_drive folder_id=%r", config.drive_folder_id)

    creds = load_credentials(config.state_dir)
    state = SyncState.load(_state_store)

    syncer = Syncer(
        drive=DriveClient(creds),
        embedder=embedder,
        backend=backend,
        state=state,
        store=_state_store,
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
    )
    report = syncer.sync(config.drive_folder_id)
    logger.info("sync_drive complete: %s", report.model_dump())

    return {
        "added": report.added,
        "updated": report.updated,
        "deleted": report.deleted,
        "skipped": report.skipped,
        "errors": report.errors,
    }
