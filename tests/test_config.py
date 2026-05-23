from pathlib import Path

import pytest

from anchor_mcp.config import AnchorConfig, get_state_dir, load_config, save_config
from anchor_mcp.errors import ConfigNotFoundError


def test_save_load_roundtrip(tmp_path: Path) -> None:
    cfg = AnchorConfig(drive_folder_id="folder123", state_dir=tmp_path)
    save_config(cfg)
    loaded = load_config(state_dir=tmp_path)
    assert loaded.drive_folder_id == "folder123"
    assert loaded.vector_backend == "pinecone"
    assert loaded.pinecone_dense_model == "multilingual-e5-large"
    assert loaded.state_dir == tmp_path


def test_defaults(tmp_path: Path) -> None:
    cfg = AnchorConfig(drive_folder_id="x", state_dir=tmp_path)
    assert cfg.chunk_size == 800
    assert cfg.chunk_overlap == 100
    assert cfg.judge_model == "anthropic/claude-haiku-4-5"


def test_atomic_write_no_tmp_left(tmp_path: Path) -> None:
    cfg = AnchorConfig(drive_folder_id="x", state_dir=tmp_path)
    save_config(cfg)
    assert (tmp_path / "config.json").exists()
    assert not (tmp_path / "config.json.tmp").exists()


def test_load_missing_raises(tmp_path: Path) -> None:
    with pytest.raises(ConfigNotFoundError, match="anchor init"):
        load_config(state_dir=tmp_path)


def test_load_malformed_raises(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text("{not valid json}", encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(state_dir=tmp_path)


def test_state_dir_respects_xdg(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    assert get_state_dir() == tmp_path / "anchor"


def test_state_dir_default_home(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    result = get_state_dir()
    assert result.name == ".anchor"
