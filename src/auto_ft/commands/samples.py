"""auto_ft samples <job_name> [--step N] — list sample image paths.

Imports the cwd-based resolver from `commands/checkpoints.py`, which
delegates save_root math to `state_index.resolve_save_root`. Raises
`OpsError(E_SAMPLES_MISSING)` when the `samples/` subdir has not been
written yet (pre-first-sample race).
"""

from __future__ import annotations

import json
from typing import Annotated

import typer

from auto_ft.commands.checkpoints import _resolve_save_root_for_cwd
from auto_ft.errors import AutoFtError, ErrorCode, OpsError
from auto_ft.training.artifacts import scan_samples

app = typer.Typer(no_args_is_help=True, add_completion=False)


@app.callback(invoke_without_command=True)
def samples(
    job_name: Annotated[str, typer.Argument(help="Job name from .autoft/state.json")],
    step: Annotated[
        int | None,
        typer.Option("--step", help="Filter by step number (exact match)."),
    ] = None,
) -> None:
    """List sample images under `save_root/samples/` (flat + subdir layouts)."""
    try:
        save_root = _resolve_save_root_for_cwd(job_name)
        samples_dir = save_root / "samples"
        if not samples_dir.exists():
            raise OpsError(
                ErrorCode.E_SAMPLES_MISSING,
                f"No samples/ directory yet: {samples_dir}",
                {"save_root": str(save_root)},
            )
        results = scan_samples(save_root, step=step)
        payload = [s.model_dump(mode="json") for s in results]
        typer.echo(json.dumps(payload))
    except AutoFtError as exc:
        typer.echo(json.dumps(exc.to_json()))
        raise typer.Exit(1) from None
