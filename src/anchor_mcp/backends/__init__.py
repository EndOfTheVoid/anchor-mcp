from typing import Any

from anchor_mcp.backends.base import VectorBackend
from anchor_mcp.config import AnchorConfig
from anchor_mcp.errors import BackendError


def get_backend(config: AnchorConfig) -> VectorBackend:
    if config.vector_backend == "pinecone":
        try:
            from pinecone import Pinecone  # type: ignore[import-untyped]
        except ImportError as exc:
            raise BackendError(
                "pinecone is not installed. Install with: pip install pinecone"
            ) from exc
        from anchor_mcp import secrets

        api_key = secrets.get_pinecone_api_key()
        if not api_key:
            raise BackendError(
                "PINECONE_API_KEY environment variable is not set. "
                "Get your key at https://app.pinecone.io → API Keys."
            )
        pc: Any = Pinecone(api_key=api_key)
        from anchor_mcp.backends.pinecone_backend import PineconeBackend

        return PineconeBackend(pc, config.pinecone_index)
    raise BackendError(f"Unknown vector backend: {config.vector_backend!r}")
