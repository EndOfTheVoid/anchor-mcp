from io import BytesIO

import pypdf

from anchor_mcp.drive import DriveFile
from anchor_mcp.errors import ExtractError, UnsupportedMimeTypeError

_PLAIN_TEXT_MIMES = frozenset(
    {
        "text/plain",
        "text/markdown",
        "application/vnd.google-apps.document",
    }
)


def extract_text(file: DriveFile, raw_bytes: bytes) -> str:
    if file.mime_type == "application/pdf":
        return _extract_pdf(raw_bytes, file.name)
    if file.mime_type in _PLAIN_TEXT_MIMES:
        return _decode_utf8(raw_bytes, file.name)
    raise UnsupportedMimeTypeError(
        f"Unsupported MIME type {file.mime_type!r} for file {file.name!r}. "
        "Supported types: PDF (text-extractable), text/plain, text/markdown, Google Docs."
    )


def _extract_pdf(raw_bytes: bytes, filename: str) -> str:
    reader = pypdf.PdfReader(BytesIO(raw_bytes))
    pages = [page.extract_text() or "" for page in reader.pages]
    text = "\n\n".join(p for p in pages if p.strip())
    if not text.strip():
        raise ExtractError(
            f"No extractable text found in {filename!r}. "
            "It may be a scanned PDF. OCR is not supported in v1."
        )
    return text


def _decode_utf8(raw_bytes: bytes, filename: str) -> str:
    try:
        return raw_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ExtractError(
            f"Failed to decode {filename!r} as UTF-8. Only UTF-8 encoded files are supported."
        ) from exc
