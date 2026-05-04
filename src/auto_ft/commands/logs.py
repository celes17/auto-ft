"""auto_ft logs <job_name> [--tail N] — platform-independent log tail.

Pure-Python via `collections.deque` (status_parser.tail_lines); no shell
`tail` / Windows `Get-Content` dependency.

Lines are emitted as a JSON array via `json.dumps(list)` — control
characters in log content stay escaped inside string values and cannot
break JSON shape.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from auto_ft.errors import AutoFtError, ConfigError, ErrorCode
from auto_ft.training.state_index import read_state, resolve_save_root, state_path
from auto_ft.training.status_parser import tail_lines

app = typer.Typer(no_args_is_help=True, add_completion=False)


def _resolve_save_root_for_cwd(job_name: str) -> Path:
    """Read `.autoft/state.json` from cwd; delegate to `state_index.resolve_save_root`.

    Raises `ConfigError(E_RUN_NOT_FOUND)` if state.json is absent.
    `resolve_save_root` raises `OpsError(E_RUN_NOT_FOUND)` for unknown jobs.
    """
    path = state_path(Path.cwd().resolve())
    state = read_state(path)
    if state is None:
        raise ConfigError(
            ErrorCode.E_RUN_NOT_FOUND,
            ".autoft/state.json not found (run `auto_ft init` first)",
            {"cwd": str(Path.cwd())},
        )
    return resolve_save_root(state, job_name)


@app.callback(invoke_without_command=True)
def logs(
    job_name: Annotated[str, typer.Argument(help="Job name from .autoft/state.json")],
    tail: Annotated[
        int,
        typer.Option("--tail", "-n", help="Number of trailing lines (default 100)."),
    ] = 100,
) -> None:
    """Emit JSON list of the last N lines of train.log (empty list if missing)."""
    try:
        save_root = _resolve_save_root_for_cwd(job_name)
        log_path = save_root / "train.log"
        lines = [line.rstrip("\n").rstrip("\r") for line in tail_lines(log_path, n=tail)]
        typer.echo(json.dumps(lines))
    except AutoFtError as exc:
        typer.echo(json.dumps(exc.to_json()))
        raise typer.Exit(1) from None
