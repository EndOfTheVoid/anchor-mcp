from typing import Any

from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build  # type: ignore[import-untyped]
from googleapiclient.errors import HttpError  # type: ignore[import-untyped]
from pydantic import BaseModel, Field
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from anchor_mcp.errors import SyncError

FOLDER_MIME = "application/vnd.google-apps.folder"
GOOGLE_WORKSPACE_PREFIX = "application/vnd.google-apps."
EXPORT_MIME_TYPE = "text/plain"

_LIST_FIELDS = (
    "nextPageToken, files(id, name, mimeType, modifiedTime, parents, webViewLink, md5Checksum)"
)


class DriveFile(BaseModel):
    id: str
    name: str
    mime_type: str
    modified_time: str
    parents: list[str] = Field(default_factory=list)
    web_view_link: str | None = None
    md5_checksum: str | None = None


class DriveClient:
    def __init__(self, credentials: Credentials) -> None:
        self._service: Any = build("drive", "v3", credentials=credentials)

    def list_files(self, folder_id: str, recursive: bool = True) -> list[DriveFile]:
        results: list[DriveFile] = []
        self._collect_files(folder_id, results, recursive)
        return results

    def _collect_files(
        self, folder_id: str, results: list[DriveFile], recursive: bool
    ) -> None:
        query = f"'{folder_id}' in parents and trashed = false"
        page_token: str | None = None

        while True:
            kwargs: dict[str, Any] = {
                "q": query,
                "fields": _LIST_FIELDS,
                "pageSize": 100,
            }
            if page_token:
                kwargs["pageToken"] = page_token

            try:
                response: dict[str, Any] = _execute(
                    self._service.files().list(**kwargs)
                )
            except HttpError as exc:
                raise SyncError(f"Failed to list files in {folder_id!r}") from exc

            for item in response.get("files", []):
                mime: str = item["mimeType"]
                if mime == FOLDER_MIME:
                    if recursive:
                        self._collect_files(item["id"], results, recursive)
                else:
                    results.append(
                        DriveFile(
                            id=item["id"],
                            name=item["name"],
                            mime_type=mime,
                            modified_time=item["modifiedTime"],
                            parents=item.get("parents", []),
                            web_view_link=item.get("webViewLink"),
                            md5_checksum=item.get("md5Checksum"),
                        )
                    )

            page_token = response.get("nextPageToken")
            if not page_token:
                break

    def download_file(self, file_id: str, mime_type: str) -> bytes:
        try:
            if mime_type.startswith(GOOGLE_WORKSPACE_PREFIX):
                request = self._service.files().export_media(
                    fileId=file_id, mimeType=EXPORT_MIME_TYPE
                )
            else:
                request = self._service.files().get_media(fileId=file_id)
            return _execute(request)
        except HttpError as exc:
            raise SyncError(f"Failed to download file {file_id!r}") from exc


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, HttpError):
        status: int = exc.resp.status
        return status in (429, 500, 502, 503, 504)
    return False


@retry(
    retry=retry_if_exception(_is_retryable),
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=1, max=30),
    reraise=True,
)
def _execute(request: Any) -> Any:
    return request.execute()
