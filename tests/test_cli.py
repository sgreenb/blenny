from __future__ import annotations

from blenny.cli.main import app


def test_cli_version_flag(capsys) -> None:  # type: ignore[no-untyped-def]
    rc = app(["--version"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "blenny" in out


def test_cli_default_message(capsys) -> None:  # type: ignore[no-untyped-def]
    rc = app([])
    out = capsys.readouterr().out
    assert rc == 0
    assert "pre-alpha" in out
