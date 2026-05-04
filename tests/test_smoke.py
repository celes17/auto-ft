"""Smoke tests: package imports cleanly and the CLI surface is intact."""

from __future__ import annotations

from typer.testing import CliRunner


def test_package_imports() -> None:
    import auto_ft  # noqa: F401


def test_cli_help_runs() -> None:
    from auto_ft.cli import app

    result = CliRunner().invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "auto_ft" in result.output


def test_cli_lists_all_subcommands() -> None:
    from auto_ft.cli import app

    result = CliRunner().invoke(app, ["--help"])
    assert result.exit_code == 0
    expected = [
        "init",
        "prepare",
        "train",
        "status",
        "logs",
        "stop",
        "checkpoints",
        "samples",
        "export",
    ]
    for cmd in expected:
        assert cmd in result.output, f"missing subcommand: {cmd}"


def test_version_flag_exits_zero() -> None:
    from auto_ft.cli import app

    result = CliRunner().invoke(app, ["--version"])
    assert result.exit_code == 0
    assert result.output.strip()  # non-empty version string
