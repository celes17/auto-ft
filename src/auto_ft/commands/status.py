"""auto_ft status <job_name> — filesystem-derived job status.

No cache: state.json is a static index; state comes from the filesystem
(final safetensors / checkpoints / train.pid / train.log mtime).

The module-local `_resolve_save_root_for_cwd(job_name)` wrapper reads
cwd state.json and delegates save_root math to
`state_index.resolve_save_root`.

JSON error-output contract: AutoFtError subclasses are caught at the
callback boundary, serialised to stdout via `to_json()`, then raise
`typer.Exit(1)`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from auto_ft.errors import AutoFtError, ConfigError, ErrorCode
from auto_ft.training.state_index import read_state, resolve_save_root, state_path
from auto_ft.training.status_parser import derive_state

app = typer.Typer(no_args_is_help=True, add_completion=False)


def _read_pid(save_root: Path) -> int | None:
    """Read `train.pid` if present and parseable; return None otherwise.

    Malformed PID -> None: status degrades to stopped/failed rather than
    probing an arbitrary PID.
    """
    pid_path = save_root / "train.pid"
    if not pid_path.exists():
        return None
    try:
        return int(pid_path.read_text(encoding="utf-8").strip())
    except ValueError:
        return None


def _resolve_save_root_for_cwd(job_name: str) -> Path:
    """Read `.autoft/state.json` from cwd; delegate to `state_index.resolve_save_root`.

    Raises `ConfigError(E_RUN_NOT_FOUND)` if state.json is absent (user forgot
    `auto_ft init`). `resolve_save_root` itself raises `OpsError(E_RUN_NOT_FOUND)`
    if `job_name` is not registered in state.json.
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
def status(
    job_name: Annotated[str, typer.Argument(help="Job name from .autoft/state.json")],
) -> None:
    """Emit JobStatus JSON derived purely from filesystem state (no cache)."""
    try:
        save_root = _resolve_save_root_for_cwd(job_name)
        pid = _read_pid(save_root)
        log_path = save_root / "train.log"
        js = derive_state(job_name, save_root, pid=pid, log_path=log_path)
        typer.echo(js.model_dump_json())
    except AutoFtError as exc:
        typer.echo(json.dumps(exc.to_json()))
        raise typer.Exit(1) from None
