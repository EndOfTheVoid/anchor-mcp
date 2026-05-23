import contextlib
import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, cast

from cryptography.fernet import Fernet
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow

from anchor_mcp.config import get_state_dir
from anchor_mcp.errors import AuthError

if TYPE_CHECKING:
    import google.oauth2.service_account

SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]
_KEY_FILE = ".key"
_TOKEN_FILE = "oauth_token.json"


def _token_path(state_dir: Path) -> Path:
    return state_dir / _TOKEN_FILE


def _get_or_create_key(state_dir: Path) -> Fernet:
    key_path = state_dir / _KEY_FILE
    if key_path.exists():
        raw_key = key_path.read_bytes()
    else:
        raw_key = Fernet.generate_key()
        key_path.write_bytes(raw_key)
        with contextlib.suppress(OSError, NotImplementedError):
            os.chmod(key_path, 0o600)
    return Fernet(raw_key)


def run_oauth_flow(credentials_path: Path, state_dir: Path | None = None) -> None:
    if state_dir is None:
        state_dir = get_state_dir()

    if not credentials_path.exists():
        raise AuthError(f"Credentials file not found: {credentials_path}")

    try:
        flow = InstalledAppFlow.from_client_secrets_file(str(credentials_path), SCOPES)
        creds = cast(Credentials, flow.run_local_server(port=0))
    except AuthError:
        raise
    except Exception as exc:
        raise AuthError(f"OAuth flow failed: {exc}") from exc

    fernet = _get_or_create_key(state_dir)
    _token_path(state_dir).write_bytes(fernet.encrypt(creds.to_json().encode()))


def load_credentials(state_dir: Path | None = None) -> Credentials:
    if state_dir is None:
        state_dir = get_state_dir()

    token_path = _token_path(state_dir)
    if not token_path.exists():
        raise AuthError("No OAuth token found. Run `anchor auth login --credentials <path>`.")

    try:
        fernet = _get_or_create_key(state_dir)
        decrypted = fernet.decrypt(token_path.read_bytes())
        info: object = json.loads(decrypted.decode())
        creds = Credentials.from_authorized_user_info(info, SCOPES)  # type: ignore[arg-type]
    except AuthError:
        raise
    except Exception as exc:
        raise AuthError(f"Failed to load credentials: {exc}") from exc

    if creds.expired and creds.refresh_token:
        try:
            creds.refresh(Request())
            token_path.write_bytes(fernet.encrypt(creds.to_json().encode()))
        except Exception as exc:
            raise AuthError(f"Failed to refresh token: {exc}") from exc

    return creds


def is_authenticated(state_dir: Path | None = None) -> bool:
    if state_dir is None:
        state_dir = get_state_dir()
    return _token_path(state_dir).exists()


def load_service_account_credentials() -> "google.oauth2.service_account.Credentials":
    """Load Drive credentials from GOOGLE_SERVICE_ACCOUNT_KEY env var (JSON string)."""
    import google.oauth2.service_account as _sa

    key_json = os.environ.get("GOOGLE_SERVICE_ACCOUNT_KEY")
    if not key_json:
        raise AuthError("GOOGLE_SERVICE_ACCOUNT_KEY environment variable is not set")
    try:
        key_data: object = json.loads(key_json)
        creds = _sa.Credentials.from_service_account_info(
            key_data,  # type: ignore[arg-type]
            scopes=SCOPES,
        )
    except AuthError:
        raise
    except Exception as exc:
        raise AuthError(f"Failed to load service account credentials: {exc}") from exc
    return creds
