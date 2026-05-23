from pathlib import Path

import pytest
from click.testing import CliRunner

from anchor_mcp.cli import cli


def test_init_creates_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    result = CliRunner().invoke(cli, ["init"], input="myfolder123\nanchor\n")
    assert result.exit_code == 0
    assert (tmp_path / "anchor" / "config.json").exists()


def test_init_creates_subdirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    CliRunner().invoke(cli, ["init"], input="myfolder\nanchor\n")
    state = tmp_path / "anchor"
    for subdir in ("cache", "notes", "logs"):
        assert (state / subdir).is_dir(), f"Missing subdir: {subdir}"


def test_config_show(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    runner = CliRunner()
    runner.invoke(cli, ["init"], input="myfolder\nanchor\n")
    result = runner.invoke(cli, ["config", "show"])
    assert result.exit_code == 0
    assert "drive_folder_id" in result.output
    assert "myfolder" in result.output


def test_config_show_no_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    result = CliRunner().invoke(cli, ["config", "show"])
    assert result.exit_code != 0


def test_config_set_valid(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    runner = CliRunner()
    runner.invoke(cli, ["init"], input="myfolder\nanchor\n")
    result = runner.invoke(cli, ["config", "set", "chunk_size", "1000"])
    assert result.exit_code == 0
    show = runner.invoke(cli, ["config", "show"])
    assert "1000" in show.output


def test_config_set_invalid_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    runner = CliRunner()
    runner.invoke(cli, ["init"], input="myfolder\nanchor\n")
    result = runner.invoke(cli, ["config", "set", "nonexistent_key", "value"])
    assert result.exit_code != 0


def test_config_set_invalid_value(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    runner = CliRunner()
    runner.invoke(cli, ["init"], input="myfolder\nanchor\n")
    result = runner.invoke(cli, ["config", "set", "vector_backend", "invalid_backend"])
    assert result.exit_code != 0


def test_doctor_missing_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    result = CliRunner().invoke(cli, ["doctor"])
    assert result.exit_code != 0
    assert "✗" in result.output


def test_doctor_all_green(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.setenv("PINECONE_API_KEY", "pc-test-key")
    runner = CliRunner()
    runner.invoke(cli, ["init"], input="myfolder\nanchor\n")
    (tmp_path / "anchor" / "oauth_token.json").write_text("{}", encoding="utf-8")
    result = runner.invoke(cli, ["doctor"])
    assert result.exit_code == 0
    assert "✗" not in result.output
