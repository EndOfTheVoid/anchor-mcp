class AnchorError(Exception):
    """Base class for all Anchor errors."""


class ConfigNotFoundError(AnchorError):
    """Raised when config.json does not exist."""


class AuthError(AnchorError):
    """Raised on OAuth or credential failures."""


class SyncError(AnchorError):
    """Raised when a Drive sync operation fails."""


class BackendError(AnchorError):
    """Raised on vector backend failures."""
