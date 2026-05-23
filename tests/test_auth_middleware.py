import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import jwt
import pytest

from anchor_mcp.auth_middleware import (
    AnchorTokenVerifier,
    decode_jwt,
    issue_jwt,
    require_role,
    verify_pkce,
)
from anchor_mcp.errors import AuthError

SECRET = "test-jwt-secret-abc123-padding-for-32-bytes"


@pytest.fixture(autouse=True)
def jwt_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("JWT_SECRET", SECRET)


# ── decode_jwt ────────────────────────────────────────────────────────────────


def test_decode_jwt_roundtrip_reader() -> None:
    token = issue_jwt("user@test.com", "reader")
    claims = decode_jwt(token)
    assert claims.sub == "user@test.com"
    assert claims.role == "reader"
    assert claims.exp > int(time.time())


def test_decode_jwt_roundtrip_admin() -> None:
    token = issue_jwt("admin@test.com", "admin")
    claims = decode_jwt(token)
    assert claims.sub == "admin@test.com"
    assert claims.role == "admin"


def test_decode_jwt_rejects_expired() -> None:
    now = int(time.time())
    payload = {"sub": "u@t.com", "role": "reader", "iat": now - 90_000, "exp": now - 86_400}
    token = jwt.encode(payload, SECRET, algorithm="HS256")
    with pytest.raises(AuthError, match="expired"):
        decode_jwt(token)


def test_decode_jwt_rejects_wrong_signature() -> None:
    payload = {
        "sub": "u@t.com",
        "role": "reader",
        "iat": int(time.time()),
        "exp": int(time.time()) + 3600,
    }
    token = jwt.encode(payload, "wrong-secret-padding-for-32-bytes-min", algorithm="HS256")
    with pytest.raises(AuthError):
        decode_jwt(token)  # expects SECRET, got wrong-secret


def test_decode_jwt_rejects_missing_jwt_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("JWT_SECRET")
    with pytest.raises(AuthError, match="JWT_SECRET"):
        decode_jwt("any.token.here")


# ── verify_pkce ───────────────────────────────────────────────────────────────


def test_verify_pkce_valid() -> None:
    import base64
    import hashlib

    verifier = "dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk"
    challenge = (
        base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
    )
    assert verify_pkce(verifier, challenge) is True


def test_verify_pkce_invalid() -> None:
    assert verify_pkce("wrong-verifier", "some-challenge") is False


# ── require_role ──────────────────────────────────────────────────────────────


def _set_auth_context(role: str):
    from mcp.server.auth.middleware.auth_context import auth_context_var
    from mcp.server.auth.middleware.bearer_auth import AuthenticatedUser
    from mcp.server.auth.provider import AccessToken

    access_token = AccessToken(
        token="dummy",
        client_id="user@test.com",
        scopes=[role],
        expires_at=int(time.time()) + 3600,
    )
    user = AuthenticatedUser(access_token)
    return auth_context_var.set(user)


def test_require_role_local_mode_allows_all() -> None:
    @require_role("admin")
    def admin_fn() -> str:
        return "ok"

    # No auth context (contextvar not set) → local stdio mode → allow all
    assert admin_fn() == "ok"


def test_require_role_allows_admin() -> None:
    from mcp.server.auth.middleware.auth_context import auth_context_var

    token = _set_auth_context("admin")
    try:

        @require_role("admin")
        def admin_fn() -> str:
            return "ok"

        assert admin_fn() == "ok"
    finally:
        auth_context_var.reset(token)


def test_require_role_admin_blocks_reader() -> None:
    from mcp.server.auth.middleware.auth_context import auth_context_var

    token = _set_auth_context("reader")
    try:

        @require_role("admin")
        def admin_fn() -> str:
            return "ok"

        with pytest.raises(AuthError, match="Admin role required"):
            admin_fn()
    finally:
        auth_context_var.reset(token)


def test_require_role_reader_allows_reader() -> None:
    from mcp.server.auth.middleware.auth_context import auth_context_var

    token = _set_auth_context("reader")
    try:

        @require_role("reader")
        def reader_fn() -> str:
            return "ok"

        assert reader_fn() == "ok"
    finally:
        auth_context_var.reset(token)


def test_require_role_reader_allows_admin() -> None:
    from mcp.server.auth.middleware.auth_context import auth_context_var

    token = _set_auth_context("admin")
    try:

        @require_role("reader")
        def reader_fn() -> str:
            return "ok"

        assert reader_fn() == "ok"
    finally:
        auth_context_var.reset(token)


def test_require_role_preserves_function_name() -> None:
    @require_role("admin")
    def my_tool() -> None:
        pass

    assert my_tool.__name__ == "my_tool"


# ── AnchorTokenVerifier ───────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_anchor_token_verifier_valid() -> None:
    token = issue_jwt("user@test.com", "reader")
    verifier = AnchorTokenVerifier()
    result = await verifier.verify_token(token)
    assert result is not None
    assert result.client_id == "user@test.com"
    assert result.scopes == ["reader"]


@pytest.mark.asyncio
async def test_anchor_token_verifier_admin() -> None:
    token = issue_jwt("admin@test.com", "admin")
    verifier = AnchorTokenVerifier()
    result = await verifier.verify_token(token)
    assert result is not None
    assert result.scopes == ["admin"]


@pytest.mark.asyncio
async def test_anchor_token_verifier_invalid_returns_none() -> None:
    verifier = AnchorTokenVerifier()
    result = await verifier.verify_token("not.a.valid.jwt")
    assert result is None


@pytest.mark.asyncio
async def test_anchor_token_verifier_expired_returns_none() -> None:
    now = int(time.time())
    payload = {"sub": "u@t.com", "role": "reader", "iat": now - 90_000, "exp": now - 86_400}
    token = jwt.encode(payload, SECRET, algorithm="HS256")
    verifier = AnchorTokenVerifier()
    result = await verifier.verify_token(token)
    assert result is None


# ── OAuth callback (unit-level) ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_oauth_callback_issues_code_for_allowlisted_email(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mock Google endpoints; verify callback stores a server auth code."""
    import anchor_mcp.server as srv
    from anchor_mcp.state_store import LocalStateStore

    store = LocalStateStore(tmp_path / "cache")
    monkeypatch.setattr(srv, "_state_store", store)
    monkeypatch.setattr(srv, "_config", MagicMock(state_dir=tmp_path))

    # Pre-populate allowlist
    from anchor_mcp.server import Allowlist, _save_allowlist

    _save_allowlist(Allowlist(readers=["reader@test.com"], admins=[]))

    # Plant a pending OAuth state
    google_state = "test-google-state-abc"
    now = int(time.time())
    srv._oauth_pending[google_state] = {
        "client_id": "mcp-client",
        "client_state": "orig-state",
        "redirect_uri": "https://client.example.com/callback",
        "code_challenge": "E9Melhoa2OwvFrEMTJguCHaoeK1t8URWbuGJSstw-cM",
        "expires_at": now + 600,
    }

    # Mock httpx calls
    mock_token_resp = MagicMock()
    mock_token_resp.status_code = 200
    mock_token_resp.json.return_value = {"access_token": "google-at"}

    mock_userinfo_resp = MagicMock()
    mock_userinfo_resp.status_code = 200
    mock_userinfo_resp.json.return_value = {"email": "reader@test.com"}

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    mock_client.post = AsyncMock(return_value=mock_token_resp)
    mock_client.get = AsyncMock(return_value=mock_userinfo_resp)

    with patch("anchor_mcp.server.httpx") as mock_httpx:
        mock_httpx.AsyncClient.return_value = mock_client

        from starlette.applications import Starlette
        from starlette.routing import Route
        from starlette.testclient import TestClient

        app = Starlette(
            routes=[Route("/oauth/callback", endpoint=srv.oauth_callback, methods=["GET"])]
        )
        client = TestClient(app, follow_redirects=False)

        resp = client.get(f"/oauth/callback?code=google-code&state={google_state}")

    assert resp.status_code == 302
    location = resp.headers["location"]
    assert "code=" in location
    assert "state=orig-state" in location
    assert google_state not in srv._oauth_pending


@pytest.mark.asyncio
async def test_oauth_callback_denies_non_allowlisted_email(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import anchor_mcp.server as srv
    from anchor_mcp.server import Allowlist, _save_allowlist
    from anchor_mcp.state_store import LocalStateStore

    store = LocalStateStore(tmp_path / "cache")
    monkeypatch.setattr(srv, "_state_store", store)
    monkeypatch.setattr(srv, "_config", MagicMock(state_dir=tmp_path))

    _save_allowlist(Allowlist(readers=[], admins=[]))

    google_state = "test-state-xyz"
    now = int(time.time())
    srv._oauth_pending[google_state] = {
        "client_id": "c",
        "client_state": "s",
        "redirect_uri": "https://client.example.com/cb",
        "code_challenge": "challenge",
        "expires_at": now + 600,
    }

    mock_token_resp = MagicMock(status_code=200)
    mock_token_resp.json.return_value = {"access_token": "at"}
    mock_userinfo_resp = MagicMock(status_code=200)
    mock_userinfo_resp.json.return_value = {"email": "stranger@notallowed.com"}

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    mock_client.post = AsyncMock(return_value=mock_token_resp)
    mock_client.get = AsyncMock(return_value=mock_userinfo_resp)

    with patch("anchor_mcp.server.httpx") as mock_httpx:
        mock_httpx.AsyncClient.return_value = mock_client

        from starlette.applications import Starlette
        from starlette.routing import Route
        from starlette.testclient import TestClient

        app = Starlette(
            routes=[Route("/oauth/callback", endpoint=srv.oauth_callback, methods=["GET"])]
        )
        client = TestClient(app)
        resp = client.get(f"/oauth/callback?code=gc&state={google_state}")

    assert resp.status_code == 403
    assert resp.json()["error"] == "access_denied"


# ── health endpoint ───────────────────────────────────────────────────────────


def test_health_endpoint() -> None:
    from starlette.applications import Starlette
    from starlette.routing import Route
    from starlette.testclient import TestClient

    import anchor_mcp.server as srv

    app = Starlette(routes=[Route("/health", endpoint=srv.health, methods=["GET"])])
    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}
