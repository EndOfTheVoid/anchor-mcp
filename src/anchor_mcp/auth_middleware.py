import base64
import hashlib
import os
import time
from collections.abc import Callable
from functools import wraps
from typing import Any, Literal, TypeVar

import jwt
from pydantic import BaseModel

from anchor_mcp.errors import AuthError

_F = TypeVar("_F", bound=Callable[..., Any])


class JWTClaims(BaseModel):
    sub: str
    role: Literal["reader", "admin"]
    exp: int
    iat: int = 0


def _jwt_secret() -> str:
    secret = os.environ.get("JWT_SECRET")
    if not secret:
        raise AuthError("JWT_SECRET environment variable is not set")
    return secret


def decode_jwt(token: str) -> JWTClaims:
    """Validate signature and expiry. Raises AuthError on failure."""
    secret = _jwt_secret()
    try:
        payload: object = jwt.decode(token, secret, algorithms=["HS256"])
    except jwt.ExpiredSignatureError as exc:
        raise AuthError("Token has expired") from exc
    except jwt.InvalidTokenError as exc:
        raise AuthError(f"Invalid token: {exc}") from exc
    return JWTClaims.model_validate(payload)


def issue_jwt(email: str, role: Literal["reader", "admin"]) -> str:
    """Sign and return a new server JWT (24 h TTL)."""
    secret = _jwt_secret()
    now = int(time.time())
    payload = {"sub": email, "role": role, "iat": now, "exp": now + 86_400}
    token: str = jwt.encode(payload, secret, algorithm="HS256")
    return token


def verify_pkce(code_verifier: str, code_challenge: str) -> bool:
    digest = hashlib.sha256(code_verifier.encode()).digest()
    computed = base64.urlsafe_b64encode(digest).rstrip(b"=").decode()
    return computed == code_challenge


def require_role(role: Literal["reader", "admin"]) -> Callable[[_F], _F]:
    """Decorator that enforces a minimum role on FastMCP tool functions.

    In local (stdio) mode get_access_token() returns None — all calls are
    permitted. In HTTP mode the bearer middleware guarantees the token is
    already verified, so we only check the role claim here.
    """

    def decorator(fn: _F) -> _F:
        @wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            from mcp.server.auth.middleware.auth_context import get_access_token

            access_token = get_access_token()
            if access_token is None:
                # No auth context → local stdio mode, allow all.
                return fn(*args, **kwargs)
            if role == "admin" and "admin" not in access_token.scopes:
                raise AuthError("Admin role required for this operation")
            return fn(*args, **kwargs)

        return wrapper  # type: ignore[return-value]

    return decorator


class AnchorTokenVerifier:
    """FastMCP TokenVerifier: validates server-issued JWTs."""

    async def verify_token(self, token: str) -> Any:
        from mcp.server.auth.provider import AccessToken as MCPAccessToken

        try:
            claims = decode_jwt(token)
        except AuthError:
            return None
        return MCPAccessToken(
            token=token,
            client_id=claims.sub,
            scopes=[claims.role],
            expires_at=claims.exp,
        )
