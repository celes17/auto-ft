"""auto_ft export <checkpoint_path> [--output <path>] — safetensors copy + metadata.

Copies preserving safetensors format (bytes + mtime via shutil.copy2);
missing source raises E_CHECKPOINT_NOT_FOUND; returns
{exported_path, size_bytes, source_step, source_job_name}.

`_parse_job_name` uses regex match-start indexing (not
`stem.rsplit("_", 1)[0]`) so job names ending in digit runs (e.g.
`test123456789_000000500.safetensors` -> `test123456789`) are preserved
instead of truncated.

JSON error-output contract: AutoFtError subclasses are caught at the
callback boundary, serialised to stdout via `to_json()`, and raise
`typer.Exit(1)`.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Annotated

import typer
from pydantic import BaseModel

from auto_ft.errors import AutoFtError, ErrorCode, OpsError
from auto_ft.training.artifacts import STEP_FILENAME_RE, _parse_step_from_checkpoint

app = typer.Typer(no_args_is_help=True, add_completion=False)


class ExportResult(BaseModel):
    """Return shape for `auto_ft export`."""

    exported_path: str
    size_bytes: int
    source_step: int | None  # None for final checkpoints
    source_job_name: str


def _parse_job_name(filename: str) -> str:
    """Extract job_name from an Ostris checkpoint filename.

    Patterns:
      `myjob_000000500.safetensors`            -> `myjob`
      `myjob.safetensors`                       -> `myjob`
      `test123456789_000000500.safetensors`    -> `test123456789`

    Uses regex match-start indexing (`STEP_FILENAME_RE.search(filename)` +
    `filename[: match.start()]`) so job names ending in digit runs are
    not accidentally truncated by stem-slice heuristics.
    """
    match = STEP_FILENAME_RE.search(filename)
    if match:
        return filename[: match.start()]
    return Path(filename).stem  # `myjob.safetensors` -> `myjob`


def _default_output_path(source: Path) -> Path:
    """Derive the default export path under cwd.

    `<cwd>/<job>_step{N}.safetensors` for interim checkpoints;
    `<cwd>/<job>_final.safetensors` for the final checkpoint.
    """
    job_name = _parse_job_name(source.name)
    step = _parse_step_from_checkpoint(source.name)
    suffix = f"step{step}" if step is not None else "final"
    return Path.cwd() / f"{job_name}_{suffix}.safetensors"


@app.callback(invoke_without_command=True)
def export(
    checkpoint_path: Annotated[
        Path,
        typer.Argument(help="Path to source safetensors file"),
    ],
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Destination path"),
    ] = None,
) -> None:
    """Copy a safetensors checkpoint to an exportable path; emit ExportResult JSON."""
    import json as _json

    try:
        source = (
            checkpoint_path.resolve()
            if checkpoint_path.is_absolute()
            else (Path.cwd() / checkpoint_path).resolve()
        )
        if not source.exists():
            raise OpsError(
                ErrorCode.E_CHECKPOINT_NOT_FOUND,
                f"Checkpoint not found: {source}",
                {"path": str(source)},
            )
        if not source.is_file():
            raise OpsError(
                ErrorCode.E_CHECKPOINT_NOT_FOUND,
                f"Not a file: {source}",
                {"path": str(source)},
            )

        dest = output.resolve() if output else _default_output_path(source)
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copy2(source, dest)  # preserves mtime; bytes-identical
        except PermissionError as e:
            raise OpsError(
                ErrorCode.E_PERMISSION,
                f"Cannot write to {dest}: {e}",
                {"dest": str(dest)},
            ) from e

        result = ExportResult(
            exported_path=str(dest),
            size_bytes=dest.stat().st_size,
            source_step=_parse_step_from_checkpoint(source.name),
            source_job_name=_parse_job_name(source.name),
        )
        typer.echo(result.model_dump_json())
    except AutoFtError as exc:
        typer.echo(_json.dumps(exc.to_json()))
        raise typer.Exit(1) from None
