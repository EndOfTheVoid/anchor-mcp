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
    vector_backend: Literal["chroma", "pinecone"] = "chroma"
    embedding_model: str = "BAAI/bge-m3"
    judge_model: str = "anthropic/claude-haiku-4-5"
    chunk_size: int = 800
    chunk_overlap: int = 100
    state_dir: Path = Field(default_factory=get_state_dir)
    # "auto" lets sentence-transformers pick (uses CUDA if available, else CPU).
    # Override with "cpu", "cuda", "cuda:0", "mps", etc.
    device: str = "auto"


def load_config(state_dir: Path | None = None) -> AnchorConfig:
    if state_dir is None:
        state_dir = get_state_dir()
    path = state_dir / "config.json"
    if not path.exists():
        raise ConfigNotFoundError(
            f"No config found at {path}. Run `anchor init` to get started."
        )
    raw: object = json.loads(path.read_text(encoding="utf-8"))
    return AnchorConfig.model_validate(raw)


def save_config(cfg: AnchorConfig) -> None:
    path = cfg.state_dir / "config.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(cfg.model_dump_json(indent=2), encoding="utf-8")
    tmp.replace(path)
