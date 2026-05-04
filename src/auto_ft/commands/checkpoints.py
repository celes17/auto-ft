"""auto_ft checkpoints <job_name> — list `*.safetensors` entries.

JSON error-output contract: AutoFtError subclasses are caught at the
callback boundary, serialised to stdout via `to_json()`, and raise
`typer.Exit(1)`.

save_root construction delegates to
`auto_ft.training.state_index.resolve_save_root` (the single point of
truth). This module's `_resolve_save_root_for_cwd(job_name)` is a thin
wrapper that reads the cwd-rooted state.json before delegating;
save_root math itself is NOT re-implemented here.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from auto_ft.errors import AutoFtError, ConfigError, ErrorCode
from auto_ft.training.artifacts import scan_checkpoints
from auto_ft.training.state_index import (
    read_state,
    resolve_save_root,
    state_path,
)

app = typer.Typer(no_args_is_help=True, add_completion=False)


def _resolve_save_root_for_cwd(job_name: str) -> Path:
    """Read state.json from `cwd` and delegate to `state_index.resolve_save_root`.

    The save_root math itself lives in `auto_ft.training.state_index` —
    the single point of truth for the `training_folder / job_name` rule.
    Callers MUST NOT re-implement the join.

    Raises `ConfigError(E_RUN_NOT_FOUND)` if `.autoft/state.json` is
    absent. `resolve_save_root` itself raises
    `OpsError(E_RUN_NOT_FOUND)` if the job is not registered.
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
def checkpoints(
    job_name: Annotated[str, typer.Argument(help="Job name from .autoft/state.json")],
) -> None:
    """List checkpoint files under `save_root/*.safetensors`."""
    try:
        save_root = _resolve_save_root_for_cwd(job_name)
        results = scan_checkpoints(save_root)
        payload = [c.model_dump(mode="json") for c in results]
        typer.echo(json.dumps(payload))
    except AutoFtError as exc:
        typer.echo(json.dumps(exc.to_json()))
        raise typer.Exit(1) from None
