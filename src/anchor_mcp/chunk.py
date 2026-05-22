import hashlib
from functools import lru_cache

import tiktoken
from pydantic import BaseModel

from anchor_mcp.drive import DriveFile

_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


class Chunk(BaseModel):
    id: str
    text: str
    file_id: str
    file_name: str
    chunk_index: int
    token_count: int
    modified_time: str
    source_url: str | None


@lru_cache(maxsize=1)
def _get_encoder() -> tiktoken.Encoding:
    return tiktoken.get_encoding("cl100k_base")


def _count(text: str, enc: tiktoken.Encoding) -> int:
    return len(enc.encode(text))


def _split(text: str, separators: list[str], chunk_size: int, enc: tiktoken.Encoding) -> list[str]:
    """Recursively split text into pieces each fitting within chunk_size tokens."""
    if _count(text, enc) <= chunk_size:
        return [text]

    for i, sep in enumerate(separators):
        if sep == "":
            # Last resort: split at exact token boundaries
            tokens = enc.encode(text)
            return [enc.decode(tokens[j : j + chunk_size]) for j in range(0, len(tokens), chunk_size)]

        if sep not in text:
            continue

        parts = [p for p in text.split(sep) if p]
        result: list[str] = []
        for part in parts:
            if _count(part, enc) <= chunk_size:
                result.append(part)
            else:
                result.extend(_split(part, separators[i + 1 :], chunk_size, enc))
        return result

    return [text]


def _merge(pieces: list[str], chunk_size: int, overlap: int, enc: tiktoken.Encoding) -> list[str]:
    """Merge split pieces into chunks of at most chunk_size tokens with overlap."""
    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for piece in pieces:
        piece_tokens = _count(piece, enc)

        if current_tokens + piece_tokens > chunk_size and current:
            chunks.append("\n".join(current))

            # Retain trailing pieces for overlap
            overlap_parts: list[str] = []
            overlap_tokens = 0
            for part in reversed(current):
                part_tokens = _count(part, enc)
                if overlap_tokens + part_tokens > overlap:
                    break
                overlap_parts.insert(0, part)
                overlap_tokens += part_tokens

            current = overlap_parts
            current_tokens = overlap_tokens

        current.append(piece)
        current_tokens += piece_tokens

    if current:
        chunks.append("\n".join(current))

    return [c for c in chunks if c.strip()]


def chunk_text(text: str, file: DriveFile, chunk_size: int, overlap: int) -> list[Chunk]:
    enc = _get_encoder()
    pieces = _split(text, list(_SEPARATORS), chunk_size, enc)
    raw_chunks = _merge(pieces, chunk_size, overlap, enc)

    result: list[Chunk] = []
    for i, chunk_content in enumerate(raw_chunks):
        chunk_id = hashlib.sha256(
            f"{file.id}:{i}:{chunk_content}".encode()
        ).hexdigest()
        result.append(
            Chunk(
                id=chunk_id,
                text=chunk_content,
                file_id=file.id,
                file_name=file.name,
                chunk_index=i,
                token_count=_count(chunk_content, enc),
                modified_time=file.modified_time,
                source_url=file.web_view_link,
            )
        )

    return result
