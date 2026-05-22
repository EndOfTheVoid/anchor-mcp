import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel

from anchor_mcp.backends.base import SourceInfo, VectorBackend
from anchor_mcp.config import AnchorConfig, load_config
from anchor_mcp.embed import Embedder

mcp = FastMCP(
    "anchor",
    instructions=(
        "Anchor gives you access to the user's Google Drive knowledge base. "
        "Always cite sources when using information from search results."
    ),
)

# ── lazy singletons ───────────────────────────────────────────────────────────

_config: AnchorConfig | None = None
_embedder: Embedder | None = None
_backend: VectorBackend | None = None

logger = logging.getLogger("anchor_mcp")


def _setup_logging(log_dir: Path) -> None:
    if logger.handlers:
        return
    log_dir.mkdir(parents=True, exist_ok=True)
    logger.setLevel(logging.DEBUG)

    fh = RotatingFileHandler(
        str(log_dir / "anchor.log"), maxBytes=5 * 1024 * 1024, backupCount=3
    )
    fh.setFormatter(logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s"))
    logger.addHandler(fh)

    sh = logging.StreamHandler(sys.stderr)
    sh.setLevel(logging.WARNING)
    logger.addHandler(sh)


def _ensure_initialized() -> tuple[AnchorConfig, Embedder, VectorBackend]:
    global _config, _embedder, _backend

    config = _config
    if config is None:
        from anchor_mcp.backends import get_backend

        config = load_config()
        _config = config
        _setup_logging(config.state_dir / "logs")

    embedder = _embedder
    if embedder is None:
        embedder = Embedder(config.embedding_model)
        _embedder = embedder

    backend = _backend
    if backend is None:
        from anchor_mcp.backends import get_backend

        backend = get_backend(config)
        _backend = backend

    return config, embedder, backend


# ── response models ───────────────────────────────────────────────────────────

class SearchResult(BaseModel):
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


# ── tools ─────────────────────────────────────────────────────────────────────

@mcp.tool(
    description=(
        "Search the user's Google Drive knowledge base by semantic similarity. "
        "Returns up to top_k chunks of text with full source metadata. "
        "You MUST cite every result you use with the format [file_name, chunk N](source_url). "
        "If results have low relevance scores (below 0.5), tell the user the answer "
        "may not be in their indexed documents."
    )
)
def search(
    query: str,
    top_k: int = 5,
    file_name_filter: str | None = None,
) -> list[SearchResult]:
    config, embedder, backend = _ensure_initialized()
    del config
    logger.info("search query=%r top_k=%d filter=%r", query, top_k, file_name_filter)

    embedding = embedder.embed_query(query)
    results = backend.query(embedding, top_k=top_k, file_name_filter=file_name_filter)

    return [
        SearchResult(
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
        "Optionally filter by file name substring."
    )
)
def list_sources(name_filter: str | None = None) -> list[SourceInfo]:
    _, _, backend = _ensure_initialized()
    logger.info("list_sources filter=%r", name_filter)

    sources = backend.list_sources()
    if name_filter:
        sources = [s for s in sources if name_filter.lower() in s.file_name.lower()]
    return sources


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
    logger.info("sync_drive folder_id=%r", config.drive_folder_id)

    creds = load_credentials(config.state_dir)
    state_path = config.state_dir / "cache" / "sync_state.json"
    state = SyncState.load(state_path)

    syncer = Syncer(
        drive=DriveClient(creds),
        embedder=embedder,
        backend=backend,
        state=state,
        state_path=state_path,
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
