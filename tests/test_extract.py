from unittest.mock import MagicMock, patch

import pytest

from anchor_mcp.drive import DriveFile
from anchor_mcp.errors import ExtractError, UnsupportedMimeTypeError
from anchor_mcp.extract import extract_text


def _file(mime: str, name: str = "test") -> DriveFile:
    return DriveFile(
        id="file1",
        name=name,
        mime_type=mime,
        modified_time="2024-01-01T00:00:00.000Z",
    )


# ── plain text ────────────────────────────────────────────────────────────────

def test_extract_plain_text() -> None:
    result = extract_text(_file("text/plain", "notes.txt"), b"Hello, world!")
    assert result == "Hello, world!"


def test_extract_markdown() -> None:
    content = "# Heading\n\nSome **bold** text."
    result = extract_text(_file("text/markdown", "doc.md"), content.encode())
    assert result == content


def test_extract_google_doc() -> None:
    content = "Exported Google Doc content."
    result = extract_text(
        _file("application/vnd.google-apps.document", "gdoc"),
        content.encode(),
    )
    assert result == content


def test_extract_utf8_error_raises() -> None:
    bad_bytes = b"\xff\xfe invalid utf-8"
    with pytest.raises(ExtractError, match="UTF-8"):
        extract_text(_file("text/plain", "bad.txt"), bad_bytes)


# ── PDF ───────────────────────────────────────────────────────────────────────

def _mock_pdf(page_texts: list[str]) -> MagicMock:
    pages = [MagicMock(extract_text=MagicMock(return_value=t)) for t in page_texts]
    reader = MagicMock()
    reader.pages = pages
    return reader


def test_extract_pdf_single_page() -> None:
    with patch("anchor_mcp.extract.pypdf.PdfReader", return_value=_mock_pdf(["Page one text."])):
        result = extract_text(_file("application/pdf", "doc.pdf"), b"fake")
    assert result == "Page one text."


def test_extract_pdf_multiple_pages() -> None:
    with patch(
        "anchor_mcp.extract.pypdf.PdfReader",
        return_value=_mock_pdf(["Page 1.", "Page 2.", "Page 3."]),
    ):
        result = extract_text(_file("application/pdf", "doc.pdf"), b"fake")
    assert "Page 1." in result
    assert "Page 2." in result
    assert "Page 3." in result


def test_extract_pdf_empty_pages_skipped() -> None:
    with patch(
        "anchor_mcp.extract.pypdf.PdfReader",
        return_value=_mock_pdf(["Real content.", "", "   "]),
    ):
        result = extract_text(_file("application/pdf", "doc.pdf"), b"fake")
    assert result == "Real content."


def test_extract_scanned_pdf_raises() -> None:
    with (
        patch("anchor_mcp.extract.pypdf.PdfReader", return_value=_mock_pdf(["", "   "])),
        pytest.raises(ExtractError, match="scanned"),
    ):
        extract_text(_file("application/pdf", "scan.pdf"), b"fake")


# ── unsupported ───────────────────────────────────────────────────────────────

def test_extract_unsupported_mime_raises() -> None:
    with pytest.raises(UnsupportedMimeTypeError):
        extract_text(_file("application/vnd.google-apps.spreadsheet"), b"data")


def test_extract_image_mime_raises() -> None:
    with pytest.raises(UnsupportedMimeTypeError):
        extract_text(_file("image/png", "photo.png"), b"\x89PNG")
