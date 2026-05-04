"""auto_ft CLI entry point."""

from __future__ import annotations

import json
import sys
from importlib.metadata import version as _pkg_version

import typer

from auto_ft.commands import checkpoints as _checkpoints_cmd
from auto_ft.commands import export as _export_cmd
from auto_ft.commands import init as _init_cmd
from auto_ft.commands import logs as _logs_cmd
from auto_ft.commands import prepare as _prepare_cmd
from auto_ft.commands import samples as _samples_cmd
from auto_ft.commands import status as _status_cmd
from auto_ft.commands import stop as _stop_cmd
from auto_ft.commands import train as _train_cmd
from auto_ft.errors import AutoFtError

app = typer.Typer(
    name="auto_ft",
    help="auto_ft: agentic LoRA fine-tuning tool wrapping ostris/ai-toolkit.",
    no_args_is_help=True,
    add_completion=False,
)


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(_pkg_version("auto_ft"))
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        None,
        "--version",
        callback=_version_callback,
        is_eager=True,
        help="Show the auto_ft version and exit.",
    ),
) -> None:
    """auto_ft top-level."""


# Subcommands are wired via `app.command()` (not `add_typer()`) so each
# accepts `<positional> --option` directly. `add_typer` would create a
# Click Group, which treats the next positional as a subcommand name and
# breaks invocations like `auto_ft prepare <dir> --trigger zyx`.
app.command(name="init", help="Create <cwd>/.autoft/state.json and register a job.")(
    _init_cmd.init_cmd
)
app.command(
    name="prepare",
    help="Validate a dataset and write .auto_ft_prep.json manifest.",
)(_prepare_cmd.prepare)
app.command(name="train", help="Spawn detached Ostris training (non-blocking).")(_train_cmd.train)
app.command(name="status", help="Show job status derived from filesystem.")(_status_cmd.status)
app.command(name="logs", help="Tail N lines of a job's train.log.")(_logs_cmd.logs)
app.command(name="stop", help="Stop a running training job (Windows: TerminateProcess).")(
    _stop_cmd.stop
)
app.command(name="checkpoints", help="List checkpoint files for a job.")(
    _checkpoints_cmd.checkpoints
)
app.command(name="samples", help="List sample image paths for a job.")(_samples_cmd.samples)
app.command(name="export", help="Copy a safetensors checkpoint to a deployable path.")(
    _export_cmd.export
)


def _entry() -> None:
    """Console-script entry; catches AutoFtError -> JSON + exit 1."""
    try:
        app()
    except AutoFtError as exc:
        json.dump(exc.to_json(), sys.stdout)
        sys.stdout.write("\n")
        sys.exit(1)


if __name__ == "__main__":
    _entry()
