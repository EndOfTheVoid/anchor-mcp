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


def test_load_missing_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANCHOR_DRIVE_FOLDER_ID", raising=False)
    with pytest.raises(ConfigNotFoundError, match="anchor init"):
        load_config(state_dir=tmp_path)


def test_load_from_env_when_no_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANCHOR_DRIVE_FOLDER_ID", "env-folder")
    monkeypatch.delenv("ANCHOR_PINECONE_INDEX", raising=False)
    monkeypatch.delenv("PINECONE_INDEX_NAME", raising=False)

    cfg = load_config(state_dir=tmp_path)
    assert cfg.drive_folder_id == "env-folder"
    assert cfg.pinecone_index == "anchor"  # default
    assert cfg.vector_backend == "pinecone"


def test_env_overrides_apply(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANCHOR_DRIVE_FOLDER_ID", "env-folder")
    monkeypatch.setenv("ANCHOR_PINECONE_INDEX", "custom-index")
    monkeypatch.setenv("ANCHOR_SEARCH_ALPHA", "0.4")
    monkeypatch.setenv("ANCHOR_JUDGE_MODEL", "openai/gpt-4o-mini")

    cfg = load_config(state_dir=tmp_path)
    assert cfg.pinecone_index == "custom-index"
    assert cfg.search_alpha == 0.4
    assert cfg.judge_model == "openai/gpt-4o-mini"


def test_file_takes_precedence_over_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    save_config(AnchorConfig(drive_folder_id="file-folder", state_dir=tmp_path))
    monkeypatch.setenv("ANCHOR_DRIVE_FOLDER_ID", "env-folder")

    loaded = load_config(state_dir=tmp_path)
    assert loaded.drive_folder_id == "file-folder"


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
