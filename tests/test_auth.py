from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from cryptography.fernet import Fernet

from anchor_mcp.auth import (
    _get_or_create_key,
    _token_path,
    is_authenticated,
    load_credentials,
    run_oauth_flow,
)
from anchor_mcp.errors import AuthError


def test_is_authenticated_false_when_no_token(tmp_path: Path) -> None:
    assert is_authenticated(state_dir=tmp_path) is False


def test_is_authenticated_true_when_token_exists(tmp_path: Path) -> None:
    _token_path(tmp_path).write_bytes(b"dummy")
    assert is_authenticated(state_dir=tmp_path) is True


def test_get_or_create_key_creates_key_file(tmp_path: Path) -> None:
    fernet = _get_or_create_key(tmp_path)
    assert (tmp_path / ".key").exists()
    assert isinstance(fernet, Fernet)


def test_get_or_create_key_reuses_existing(tmp_path: Path) -> None:
    f1 = _get_or_create_key(tmp_path)
    f2 = _get_or_create_key(tmp_path)
    # Both should decrypt the same payload
    token = f1.encrypt(b"hello")
    assert f2.decrypt(token) == b"hello"


def test_run_oauth_flow_missing_credentials_raises(tmp_path: Path) -> None:
    with pytest.raises(AuthError, match="not found"):
        run_oauth_flow(tmp_path / "nonexistent.json", state_dir=tmp_path)


def test_run_oauth_flow_saves_encrypted_token(tmp_path: Path) -> None:
    mock_creds = MagicMock()
    mock_creds.to_json.return_value = '{"token": "fake"}'

    mock_flow = MagicMock()
    mock_flow.run_local_server.return_value = mock_creds

    creds_file = tmp_path / "credentials.json"
    creds_file.write_text("{}", encoding="utf-8")

    with patch(
        "anchor_mcp.auth.InstalledAppFlow.from_client_secrets_file",
        return_value=mock_flow,
    ):
        run_oauth_flow(creds_file, state_dir=tmp_path)

    token_path = _token_path(tmp_path)
    assert token_path.exists()

    # Verify the content is encrypted (not raw JSON)
    raw = token_path.read_bytes()
    assert raw != b'{"token": "fake"}'

    # Verify it decrypts correctly
    fernet = _get_or_create_key(tmp_path)
    assert fernet.decrypt(raw) == b'{"token": "fake"}'


def test_load_credentials_missing_token_raises(tmp_path: Path) -> None:
    with pytest.raises(AuthError, match="No OAuth token"):
        load_credentials(state_dir=tmp_path)


def test_load_credentials_decrypts_and_returns(tmp_path: Path) -> None:
    token_data = '{"client_id": "x", "client_secret": "y", "refresh_token": "z", "token_uri": "https://oauth2.googleapis.com/token"}'

    fernet = _get_or_create_key(tmp_path)
    _token_path(tmp_path).write_bytes(fernet.encrypt(token_data.encode()))

    mock_creds = MagicMock()
    mock_creds.expired = False

    with patch(
        "anchor_mcp.auth.Credentials.from_authorized_user_info",
        return_value=mock_creds,
    ):
        result = load_credentials(state_dir=tmp_path)

    assert result is mock_creds


def test_load_credentials_refreshes_when_expired(tmp_path: Path) -> None:
    token_data = '{"client_id": "x", "client_secret": "y", "refresh_token": "z", "token_uri": "https://oauth2.googleapis.com/token"}'

    fernet = _get_or_create_key(tmp_path)
    _token_path(tmp_path).write_bytes(fernet.encrypt(token_data.encode()))

    mock_creds = MagicMock()
    mock_creds.expired = True
    mock_creds.refresh_token = "z"
    mock_creds.to_json.return_value = token_data

    with (
        patch("anchor_mcp.auth.Credentials.from_authorized_user_info", return_value=mock_creds),
        patch("anchor_mcp.auth.Request"),
    ):
        load_credentials(state_dir=tmp_path)

    mock_creds.refresh.assert_called_once()
