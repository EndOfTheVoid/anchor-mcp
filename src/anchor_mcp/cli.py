from pathlib import Path
from typing import Literal, cast

import click

from anchor_mcp import secrets
from anchor_mcp.config import AnchorConfig, get_state_dir, load_config, save_config
from anchor_mcp.errors import AuthError, ConfigNotFoundError


@click.group()
def cli() -> None:
    """Anchor — ground your LLM in Google Drive."""


@cli.command()
def init() -> None:
    """Bootstrap the Anchor state directory with an initial config."""
    folder_id: str = click.prompt("Google Drive folder ID")
    backend_raw: str = click.prompt(
        "Vector backend",
        default="chroma",
        type=click.Choice(["chroma", "pinecone"]),
    )
    backend = cast(Literal["chroma", "pinecone"], backend_raw)

    if not secrets.get_openrouter_api_key():
        click.echo(
            "Warning: OPENROUTER_API_KEY is not set. "
            "The verify_claim tool will be disabled.",
            err=True,
        )

    state_dir = get_state_dir()
    for subdir in ("cache", "notes", "logs", "chroma"):
        (state_dir / subdir).mkdir(parents=True, exist_ok=True)

    cfg = AnchorConfig(drive_folder_id=folder_id, vector_backend=backend, state_dir=state_dir)
    save_config(cfg)
    click.echo(f"Initialized Anchor at {state_dir}")


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

    if secrets.get_openrouter_api_key():
        _report(True, "OPENROUTER_API_KEY", "Set")
    else:
        _report(False, "OPENROUTER_API_KEY", "Not set — verify_claim tool will be disabled")

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
        click.echo(
            "Not authenticated. Run `anchor auth login --credentials <path>`."
        )
        raise SystemExit(1)
