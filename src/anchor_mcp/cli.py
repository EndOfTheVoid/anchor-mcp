import os
from pathlib import Path
from typing import Any

import click
from dotenv import load_dotenv

from anchor_mcp import secrets
from anchor_mcp.config import AnchorConfig, get_state_dir, load_config, save_config
from anchor_mcp.errors import AuthError, ConfigNotFoundError

load_dotenv()


@click.group()
def cli() -> None:
    """Anchor — ground your LLM in Google Drive."""


@cli.command()
def init() -> None:
    """Bootstrap the Anchor state directory with an initial config."""
    folder_id: str = click.prompt("Google Drive folder ID")
    pinecone_index: str = click.prompt("Pinecone index name", default="anchor")

    state_dir = get_state_dir()
    for subdir in ("cache", "notes", "logs"):
        (state_dir / subdir).mkdir(parents=True, exist_ok=True)

    cfg = AnchorConfig(
        drive_folder_id=folder_id,
        pinecone_index=pinecone_index,
        state_dir=state_dir,
    )
    save_config(cfg)
    click.echo(f"Initialized Anchor at {state_dir}")
    click.echo(
        "\nNext steps:\n"
        "  1. anchor auth login --credentials <path-to-oauth-credentials.json>\n"
        "  2. Set PINECONE_API_KEY in your environment\n"
        "  3. anchor sync"
    )


@cli.group("config")
def config_group() -> None:
    """Manage Anchor configuration."""


@config_group.command("show")
def config_show() -> None:
    """Print the current configuration (secrets redacted)."""
    try:
        cfg = load_config()
    except ConfigNotFoundError as exc:
        raise click.ClickException(str(exc)) from exc

    for key, value in cfg.model_dump().items():
        click.echo(f"{key}: {value}")


@config_group.command("set")
@click.argument("key")
@click.argument("value")
def config_set(key: str, value: str) -> None:
    """Update a single configuration field and re-validate."""
    try:
        cfg = load_config()
    except ConfigNotFoundError as exc:
        raise click.ClickException(str(exc)) from exc

    data = cfg.model_dump()
    if key not in data:
        valid = ", ".join(data.keys())
        raise click.ClickException(f"Unknown key {key!r}. Valid keys: {valid}")

    data[key] = value
    try:
        updated = AnchorConfig.model_validate(data)
    except Exception as exc:
        raise click.ClickException(f"Invalid value for {key!r}: {exc}") from exc

    save_config(updated)
    click.echo(f"Set {key} = {value}")


@cli.command()
def doctor() -> None:
    """Run sanity checks on the Anchor installation."""
    all_ok = True

    state_dir = get_state_dir()
    if state_dir.exists():
        _report(True, "State directory", str(state_dir))
    else:
        _report(False, "State directory", "Missing — run `anchor init`")
        all_ok = False

    try:
        load_config()
        _report(True, "Config", "Valid")
    except ConfigNotFoundError as exc:
        _report(False, "Config", str(exc))
        all_ok = False
    except Exception as exc:
        _report(False, "Config", f"Malformed — {exc}")
        all_ok = False

    if secrets.get_pinecone_api_key():
        _report(True, "PINECONE_API_KEY", "Set")
    else:
        _report(False, "PINECONE_API_KEY", "Not set — anchor sync will fail")
        all_ok = False

    token_path = state_dir / "oauth_token.json"
    if token_path.exists():
        _report(True, "OAuth token", "Present")
    else:
        _report(
            False,
            "OAuth token",
            "Not found — run `anchor auth login --credentials <path>`",
        )
        all_ok = False

    if not all_ok:
        raise SystemExit(1)


def _report(ok: bool, label: str, detail: str) -> None:
    icon = "✓" if ok else "✗"
    click.echo(f"  {icon} {label}: {detail}")


# ── auth ──────────────────────────────────────────────────────────────────────


@cli.group("auth")
def auth_group() -> None:
    """Manage Google Drive authentication."""


@auth_group.command("login")
@click.option(
    "--credentials",
    "credentials_path",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Path to OAuth 2.0 Desktop client credentials JSON from Google Cloud Console.",
)
def auth_login(credentials_path: Path) -> None:
    """Run the OAuth consent flow and save the encrypted token."""
    from anchor_mcp.auth import run_oauth_flow

    try:
        run_oauth_flow(credentials_path)
        click.echo("Authentication successful.")
    except AuthError as exc:
        raise click.ClickException(str(exc)) from exc


@auth_group.command("status")
def auth_status() -> None:
    """Report whether a valid OAuth token is present."""
    from anchor_mcp.auth import is_authenticated

    if is_authenticated():
        click.echo("Authenticated.")
    else:
        click.echo("Not authenticated. Run `anchor auth login --credentials <path>`.")
        raise SystemExit(1)


# ── sync ──────────────────────────────────────────────────────────────────────


@cli.command()
def sync() -> None:
    """Sync the configured Google Drive folder into Pinecone."""
    from pinecone import Pinecone  # type: ignore[import-untyped]

    from anchor_mcp.auth import load_credentials
    from anchor_mcp.backends.pinecone_backend import PineconeBackend
    from anchor_mcp.drive import DriveClient
    from anchor_mcp.embed import PineconeEmbedder
    from anchor_mcp.state_store import get_state_store
    from anchor_mcp.sync import Syncer, SyncState

    try:
        cfg = load_config()
    except ConfigNotFoundError as exc:
        raise click.ClickException(str(exc)) from exc

    try:
        creds = load_credentials()
    except AuthError as exc:
        raise click.ClickException(str(exc)) from exc

    api_key = secrets.get_pinecone_api_key()
    if not api_key:
        raise click.ClickException(
            "PINECONE_API_KEY is not set. Get your key at https://app.pinecone.io → API Keys."
        )

    pc: Any = Pinecone(api_key=api_key)
    embedder = PineconeEmbedder(pc, cfg.pinecone_dense_model, cfg.pinecone_sparse_model)
    backend = PineconeBackend(pc, cfg.pinecone_index)
    store = get_state_store(cfg)
    state = SyncState.load(store)

    syncer = Syncer(
        drive=DriveClient(creds),
        embedder=embedder,
        backend=backend,
        state=state,
        store=store,
        chunk_size=cfg.chunk_size,
        chunk_overlap=cfg.chunk_overlap,
    )

    click.echo("Syncing…")
    report = syncer.sync(cfg.drive_folder_id, show_progress=True)

    click.echo(
        f"\nDone — added {report.added}, updated {report.updated}, "
        f"deleted {report.deleted}, skipped {report.skipped}."
    )
    if report.errors:
        click.echo(f"{len(report.errors)} error(s):", err=True)
        for err in report.errors:
            click.echo(f"  {err}", err=True)


# ── serve ─────────────────────────────────────────────────────────────────────


@cli.command()
@click.option(
    "--local",
    "local_mode",
    is_flag=True,
    default=False,
    help="Use stdio transport for local dev (default: streamable-http for Cloud Run).",
)
def serve(local_mode: bool) -> None:
    """Start the Anchor MCP server."""
    if local_mode:
        _serve_stdio()
    else:
        _serve_http()


def _serve_stdio() -> None:
    """stdio transport: protect stdout from library noise then run."""
    import io
    import sys
    from typing import Any

    # MCP stdio transport requires stdout to carry only JSON-RPC frames.
    # Any library that prints to stdout corrupts the protocol.
    _real_stdout_buffer = sys.stdout.buffer
    _stderr_encoding: str = sys.stderr.encoding or "utf-8"

    class _StdoutToStderr(io.TextIOBase):
        encoding: str = _stderr_encoding

        @property
        def buffer(self) -> Any:
            return _real_stdout_buffer

        def write(self, s: str) -> int:
            return sys.stderr.write(s)

        def flush(self) -> None:
            sys.stderr.flush()

        def isatty(self) -> bool:
            return False

    sys.stdout = _StdoutToStderr()

    from anchor_mcp.server import mcp, start_background_init

    start_background_init()
    mcp.run(transport="stdio")


def _serve_http() -> None:
    """StreamableHTTP transport for Cloud Run."""
    server_url = os.environ.get("SERVER_URL") or "http://localhost:8080"
    from anchor_mcp.server import mcp, setup_http_auth, start_background_init

    setup_http_auth(server_url)
    start_background_init()
    mcp.run(transport="streamable-http")
