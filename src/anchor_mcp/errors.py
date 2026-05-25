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


class ExtractError(AnchorError):
    """Raised when text cannot be extracted from a file (e.g. scanned PDF)."""


class UnsupportedMimeTypeError(AnchorError):
    """Raised when a file's MIME type is not supported for extraction."""


class VerificationError(AnchorError):
    """Raised when the faithfulness judge cannot produce a usable verdict."""
