from unittest.mock import MagicMock, patch

import pytest

from anchor_mcp.drive import FOLDER_MIME, DriveClient, DriveFile
from anchor_mcp.errors import SyncError


def _file_item(
    id: str,
    name: str,
    mime: str,
    modified: str = "2024-01-01T00:00:00.000Z",
    **extra: object,
) -> dict[str, object]:
    return {
        "id": id,
        "name": name,
        "mimeType": mime,
        "modifiedTime": modified,
        "parents": ["root"],
        "webViewLink": f"https://drive.google.com/file/d/{id}/view",
        **extra,
    }


def _make_client(*list_responses: dict[str, object]) -> tuple[DriveClient, MagicMock]:
    mock_service = MagicMock()
    mock_service.files.return_value.list.return_value.execute.side_effect = list(list_responses)
    with patch("anchor_mcp.drive.build", return_value=mock_service):
        client = DriveClient(MagicMock())
    return client, mock_service


def test_list_files_returns_drive_files() -> None:
    items = [
        _file_item("id1", "doc.txt", "text/plain"),
        _file_item("id2", "report.pdf", "application/pdf", md5Checksum="abc123"),
    ]
    client, _ = _make_client({"files": items, "nextPageToken": None})

    files = client.list_files("folder123")

    assert len(files) == 2
    assert all(isinstance(f, DriveFile) for f in files)
    assert files[0].id == "id1"
    assert files[1].md5_checksum == "abc123"


def test_list_files_pagination() -> None:
    page1 = {"files": [_file_item("id1", "a.txt", "text/plain")], "nextPageToken": "tok"}
    page2 = {"files": [_file_item("id2", "b.txt", "text/plain")]}
    client, _ = _make_client(page1, page2)

    files = client.list_files("folder123")

    assert len(files) == 2
    assert files[0].id == "id1"
    assert files[1].id == "id2"


def test_list_files_recursive_follows_subfolders() -> None:
    root_response = {
        "files": [
            _file_item("sub1", "Subfolder", FOLDER_MIME),
            _file_item("id1", "root_doc.txt", "text/plain"),
        ]
    }
    sub_response = {"files": [_file_item("id2", "sub_doc.pdf", "application/pdf")]}
    client, _ = _make_client(root_response, sub_response)

    files = client.list_files("root_folder", recursive=True)

    ids = {f.id for f in files}
    assert ids == {"id1", "id2"}
    assert "sub1" not in ids  # folders are not included


def test_list_files_non_recursive_skips_subfolders() -> None:
    root_response = {
        "files": [
            _file_item("sub1", "Subfolder", FOLDER_MIME),
            _file_item("id1", "doc.txt", "text/plain"),
        ]
    }
    client, _ = _make_client(root_response)

    files = client.list_files("root_folder", recursive=False)

    assert len(files) == 1
    assert files[0].id == "id1"


def test_list_files_http_error_raises_sync_error() -> None:
    from googleapiclient.errors import HttpError  # type: ignore[import-untyped]

    mock_resp = MagicMock()
    mock_resp.status = 403
    error = HttpError(resp=mock_resp, content=b"Forbidden")

    mock_service = MagicMock()
    mock_service.files.return_value.list.return_value.execute.side_effect = error
    with patch("anchor_mcp.drive.build", return_value=mock_service):
        client = DriveClient(MagicMock())

    with pytest.raises(SyncError):
        client.list_files("folder123")


def test_download_google_doc_uses_export_media() -> None:
    mock_service = MagicMock()
    mock_service.files.return_value.export_media.return_value.execute.return_value = (
        b"exported text"
    )
    with patch("anchor_mcp.drive.build", return_value=mock_service):
        client = DriveClient(MagicMock())

    result = client.download_file("docid", "application/vnd.google-apps.document")

    assert result == b"exported text"
    mock_service.files.return_value.export_media.assert_called_once_with(
        fileId="docid", mimeType="text/plain"
    )
    mock_service.files.return_value.get_media.assert_not_called()


def test_download_pdf_uses_get_media() -> None:
    mock_service = MagicMock()
    mock_service.files.return_value.get_media.return_value.execute.return_value = b"%PDF-1.4 binary"
    with patch("anchor_mcp.drive.build", return_value=mock_service):
        client = DriveClient(MagicMock())

    result = client.download_file("pdfid", "application/pdf")

    assert result == b"%PDF-1.4 binary"
    mock_service.files.return_value.get_media.assert_called_once_with(fileId="pdfid")
    mock_service.files.return_value.export_media.assert_not_called()


def test_download_http_error_raises_sync_error() -> None:
    from googleapiclient.errors import HttpError  # type: ignore[import-untyped]

    mock_resp = MagicMock()
    mock_resp.status = 404
    error = HttpError(resp=mock_resp, content=b"Not Found")

    mock_service = MagicMock()
    mock_service.files.return_value.get_media.return_value.execute.side_effect = error
    with patch("anchor_mcp.drive.build", return_value=mock_service):
        client = DriveClient(MagicMock())

    with pytest.raises(SyncError):
        client.download_file("missing_id", "application/pdf")
