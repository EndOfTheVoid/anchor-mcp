import os


def get_openrouter_api_key() -> str | None:
    return os.environ.get("OPENROUTER_API_KEY")


def get_pinecone_api_key() -> str | None:
    return os.environ.get("PINECONE_API_KEY")


def get_pinecone_index_name() -> str:
    return os.environ.get("PINECONE_INDEX_NAME", "anchor")
