import json
import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from anchor_mcp.errors import ConfigNotFoundError


def get_state_dir() -> Path:
    xdg = os.environ.get("XDG_CONFIG_HOME")
    if xdg:
        return Path(xdg) / "anchor"
    return Path.home() / ".anchor"


class AnchorConfig(BaseModel):
    drive_folder_id: str
    vector_backend: Literal["pinecone"] = "pinecone"
    pinecone_index: str = "anchor"
    pinecone_dense_model: str = "multilingual-e5-large"
    pinecone_sparse_model: str = "pinecone-sparse-english-v0"
    search_alpha: float = 0.7
    judge_model: str = "anthropic/claude-haiku-4-5"
    chunk_size: int = 800
    chunk_overlap: int = 100
    state_dir: Path = Field(default_factory=get_state_dir)


def load_config(state_dir: Path | None = None) -> AnchorConfig:
    if state_dir is None:
        state_dir = get_state_dir()
    path = state_dir / "config.json"
    if path.exists():
        raw: object = json.loads(path.read_text(encoding="utf-8"))
        return AnchorConfig.model_validate(raw)

    # Stateless fallback (Cloud Run): no config.json on disk, build from env vars.
    folder_id = os.environ.get("ANCHOR_DRIVE_FOLDER_ID")
    if folder_id:
        return _config_from_env(folder_id, state_dir)

    raise ConfigNotFoundError(
        f"No config found at {path} and ANCHOR_DRIVE_FOLDER_ID is not set. "
        f"Run `anchor init` to get started, or set ANCHOR_DRIVE_FOLDER_ID for a stateless deploy."
    )


def _config_from_env(folder_id: str, state_dir: Path) -> AnchorConfig:
    data: dict[str, object] = {"drive_folder_id": folder_id, "state_dir": state_dir}
    index = os.environ.get("ANCHOR_PINECONE_INDEX") or os.environ.get("PINECONE_INDEX_NAME")
    if index:
        data["pinecone_index"] = index
    alpha = os.environ.get("ANCHOR_SEARCH_ALPHA")
    if alpha:
        data["search_alpha"] = float(alpha)
    judge_model = os.environ.get("ANCHOR_JUDGE_MODEL")
    if judge_model:
        data["judge_model"] = judge_model
    return AnchorConfig.model_validate(data)


def save_config(cfg: AnchorConfig) -> None:
    path = cfg.state_dir / "config.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(cfg.model_dump_json(indent=2), encoding="utf-8")
    tmp.replace(path)
