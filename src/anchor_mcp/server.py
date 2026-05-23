import json
import logging
import os
import secrets as _secrets
import sys
import threading
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any
from urllib.parse import quote

import httpx
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field
from starlette.requests import Request
from starlette.responses import JSONResponse, RedirectResponse, Response

from anchor_mcp.auth_middleware import require_role
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


def setup_http_auth(server_url: str) -> None:
    """Configure JWT bearer auth + host/port for HTTP (Cloud Run) mode.

    Must be called before mcp.run(transport="streamable-http").
    """
    from mcp.server.auth.settings import AuthSettings

    from anchor_mcp.auth_middleware import AnchorTokenVerifier

    server_url = server_url.rstrip("/")
    mcp.settings.auth = AuthSettings(
        issuer_url=server_url,  # type: ignore[arg-type]
        resource_server_url=None,
    )
    mcp._token_verifier = AnchorTokenVerifier()  # type: ignore[attr-defined]
    mcp.settings.host = "0.0.0.0"
    mcp.settings.port = 8080
    # Cloud Run is not localhost; disable localhost-only DNS rebinding protection.
    mcp.settings.transport_security = None


# ── OAuth in-memory state ─────────────────────────────────────────────────────

# google_state → {client_id, client_state, redirect_uri, code_challenge, expires_at}
_oauth_pending: dict[str, dict[str, Any]] = {}

# server_code → {email, role, code_challenge, redirect_uri, client_id, expires_at}
_auth_codes: dict[str, dict[str, Any]] = {}


def _clean_expired(store: dict[str, dict[str, Any]]) -> None:
    now = int(time.time())
    stale = [k for k, v in store.items() if int(v.get("expires_at", 0)) < now]
    for k in stale:
        del store[k]


def _server_url() -> str:
    return os.environ.get("SERVER_URL", "http://localhost:8080").rstrip("/")


# ── Allowlist ─────────────────────────────────────────────────────────────────


class Allowlist(BaseModel):
    readers: list[str] = Field(default_factory=list)
    admins: list[str] = Field(default_factory=list)


def _load_allowlist() -> Allowlist:
    _ensure_config()
    assert _state_store is not None
    data = _state_store.read("allowlist.json")
    if data is None:
        return Allowlist()
    return Allowlist.model_validate(json.loads(data))


def _save_allowlist(allowlist: Allowlist) -> None:
    assert _state_store is not None
    _state_store.write("allowlist.json", allowlist.model_dump_json(indent=2).encode())


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
        "Returns a summary of what changed. Admin role required."
    )
)
@require_role("admin")
def sync_drive() -> dict[str, object]:
    from anchor_mcp.drive import DriveClient
    from anchor_mcp.sync import Syncer, SyncState

    config, embedder, backend = _ensure_initialized()
    assert _state_store is not None
    logger.info("sync_drive folder_id=%r", config.drive_folder_id)

    if os.environ.get("GOOGLE_SERVICE_ACCOUNT_KEY"):
        from anchor_mcp.auth import load_service_account_credentials

        creds = load_service_account_credentials()
    else:
        from anchor_mcp.auth import load_credentials

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


# ── custom HTTP routes ────────────────────────────────────────────────────────


@mcp.custom_route("/health", methods=["GET"])
async def health(request: Request) -> Response:
    return JSONResponse({"status": "ok"})


@mcp.custom_route("/.well-known/oauth-authorization-server", methods=["GET"])
async def oauth_discovery(request: Request) -> Response:
    base = _server_url()
    return JSONResponse(
        {
            "issuer": base,
            "authorization_endpoint": f"{base}/oauth/authorize",
            "token_endpoint": f"{base}/oauth/token",
            "response_types_supported": ["code"],
            "grant_types_supported": ["authorization_code"],
        }
    )


@mcp.custom_route("/oauth/authorize", methods=["GET"])
async def oauth_authorize(request: Request) -> Response:
    client_id = request.query_params.get("client_id", "")
    redirect_uri = request.query_params.get("redirect_uri", "")
    state = request.query_params.get("state", "")
    code_challenge = request.query_params.get("code_challenge", "")
    code_challenge_method = request.query_params.get("code_challenge_method", "")

    if not redirect_uri or not code_challenge:
        return JSONResponse({"error": "invalid_request"}, status_code=400)
    if code_challenge_method != "S256":
        return JSONResponse(
            {"error": "invalid_request", "error_description": "Only S256 PKCE is supported"},
            status_code=400,
        )

    google_state = _secrets.token_urlsafe(32)
    now = int(time.time())
    _oauth_pending[google_state] = {
        "client_id": client_id,
        "client_state": state,
        "redirect_uri": redirect_uri,
        "code_challenge": code_challenge,
        "expires_at": now + 600,
    }
    _clean_expired(_oauth_pending)

    base = _server_url()
    google_client_id = os.environ.get("GOOGLE_OAUTH_CLIENT_ID", "")
    google_redirect = f"{base}/oauth/callback"

    google_url = (
        "https://accounts.google.com/o/oauth2/auth"
        f"?client_id={quote(google_client_id)}"
        f"&redirect_uri={quote(google_redirect)}"
        f"&state={quote(google_state)}"
        "&scope=email+profile"
        "&response_type=code"
        "&access_type=online"
    )
    return RedirectResponse(url=google_url, status_code=302)


@mcp.custom_route("/oauth/callback", methods=["GET"])
async def oauth_callback(request: Request) -> Response:
    code = request.query_params.get("code", "")
    google_state = request.query_params.get("state", "")

    if not code or google_state not in _oauth_pending:
        return JSONResponse({"error": "invalid_state"}, status_code=400)

    pending = _oauth_pending.pop(google_state)
    if int(time.time()) > int(pending["expires_at"]):
        return JSONResponse({"error": "state_expired"}, status_code=400)

    base = _server_url()
    google_redirect = f"{base}/oauth/callback"

    async with httpx.AsyncClient() as client:
        token_resp = await client.post(
            "https://oauth2.googleapis.com/token",
            data={
                "code": code,
                "client_id": os.environ.get("GOOGLE_OAUTH_CLIENT_ID", ""),
                "client_secret": os.environ.get("GOOGLE_OAUTH_CLIENT_SECRET", ""),
                "redirect_uri": google_redirect,
                "grant_type": "authorization_code",
            },
        )

    if token_resp.status_code != 200:
        logger.warning("Google token exchange failed: %s", token_resp.text)
        return JSONResponse({"error": "token_exchange_failed"}, status_code=502)

    google_access_token: str = token_resp.json().get("access_token", "")

    async with httpx.AsyncClient() as client:
        userinfo_resp = await client.get(
            "https://www.googleapis.com/oauth2/v3/userinfo",
            headers={"Authorization": f"Bearer {google_access_token}"},
        )

    if userinfo_resp.status_code != 200:
        return JSONResponse({"error": "userinfo_failed"}, status_code=502)

    email: str = userinfo_resp.json().get("email", "")
    if not email:
        return JSONResponse({"error": "no_email"}, status_code=400)

    allowlist = _load_allowlist()
    role: str | None = None
    if email in allowlist.admins:
        role = "admin"
    elif email in allowlist.readers:
        role = "reader"

    if role is None:
        logger.info("OAuth access denied for %s — not in allowlist", email)
        return JSONResponse(
            {"error": "access_denied", "error_description": "Email not in allowlist"},
            status_code=403,
        )

    now = int(time.time())
    server_code = _secrets.token_urlsafe(32)
    _auth_codes[server_code] = {
        "email": email,
        "role": role,
        "code_challenge": pending["code_challenge"],
        "redirect_uri": pending["redirect_uri"],
        "client_id": pending["client_id"],
        "expires_at": now + 600,
    }
    _clean_expired(_auth_codes)

    redirect_uri = str(pending["redirect_uri"])
    sep = "&" if "?" in redirect_uri else "?"
    client_state = str(pending["client_state"])
    final_url = f"{redirect_uri}{sep}code={server_code}&state={quote(client_state)}"
    return RedirectResponse(url=final_url, status_code=302)


@mcp.custom_route("/oauth/token", methods=["POST"])
async def oauth_token(request: Request) -> Response:
    from anchor_mcp.auth_middleware import issue_jwt, verify_pkce

    try:
        body = await request.form()
    except Exception:
        return JSONResponse({"error": "invalid_request"}, status_code=400)

    grant_type = body.get("grant_type", "")
    if grant_type != "authorization_code":
        return JSONResponse({"error": "unsupported_grant_type"}, status_code=400)

    code = str(body.get("code", ""))
    code_verifier = str(body.get("code_verifier", ""))

    if code not in _auth_codes:
        return JSONResponse({"error": "invalid_grant"}, status_code=400)

    stored = _auth_codes.pop(code)
    if int(time.time()) > int(stored["expires_at"]):
        return JSONResponse(
            {"error": "invalid_grant", "error_description": "Code expired"}, status_code=400
        )

    if not verify_pkce(code_verifier, str(stored["code_challenge"])):
        return JSONResponse(
            {"error": "invalid_grant", "error_description": "PKCE verification failed"},
            status_code=400,
        )

    token = issue_jwt(str(stored["email"]), str(stored["role"]))  # type: ignore[arg-type]
    return JSONResponse({"access_token": token, "token_type": "Bearer", "expires_in": 86_400})


def _require_admin(request: Request) -> Response | None:
    """Check admin role from auth context. Returns error Response or None if OK."""
    from mcp.server.auth.middleware.auth_context import get_access_token

    access_token = get_access_token()
    if access_token is None or "admin" not in access_token.scopes:
        return JSONResponse({"error": "forbidden"}, status_code=403)
    return None


@mcp.custom_route("/admin/users", methods=["POST"])
async def admin_add_user(request: Request) -> Response:
    if (err := _require_admin(request)) is not None:
        return err

    try:
        body: dict[str, str] = await request.json()
        email = body["email"]
        role = body["role"]
    except Exception:
        return JSONResponse({"error": "invalid_request"}, status_code=400)

    if role not in ("reader", "admin"):
        return JSONResponse({"error": "invalid_role"}, status_code=400)

    allowlist = _load_allowlist()
    if role == "admin":
        allowlist.readers = [e for e in allowlist.readers if e != email]
        if email not in allowlist.admins:
            allowlist.admins.append(email)
    else:
        allowlist.admins = [e for e in allowlist.admins if e != email]
        if email not in allowlist.readers:
            allowlist.readers.append(email)
    _save_allowlist(allowlist)

    return JSONResponse({"ok": True})


@mcp.custom_route("/admin/users/{email}", methods=["DELETE"])
async def admin_remove_user(request: Request) -> Response:
    if (err := _require_admin(request)) is not None:
        return err

    email = request.path_params["email"]
    allowlist = _load_allowlist()
    allowlist.readers = [e for e in allowlist.readers if e != email]
    allowlist.admins = [e for e in allowlist.admins if e != email]
    _save_allowlist(allowlist)

    return JSONResponse({"ok": True})
