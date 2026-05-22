from anchor_mcp.backends.base import VectorBackend
from anchor_mcp.config import AnchorConfig
from anchor_mcp.errors import BackendError


def get_backend(config: AnchorConfig) -> VectorBackend:
    if config.vector_backend == "chroma":
        from anchor_mcp.backends.chroma_backend import ChromaBackend

        return ChromaBackend(config.state_dir)
    if config.vector_backend == "pinecone":
        from anchor_mcp.backends.pinecone_backend import PineconeBackend

        return PineconeBackend()
    raise BackendError(f"Unknown vector backend: {config.vector_backend!r}")
