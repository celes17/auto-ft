"""auto_ft stop <job_name> — TerminateProcess on Windows.

Non-graceful by contract: Windows `terminate` maps to `TerminateProcess`
which returns `clean=False, signal_sent="TerminateProcess"`. POSIX
escalates SIGTERM -> SIGKILL via `_posix.terminate`.

Steps:
  1. Read cwd state.json; delegate save_root resolution to
     `state_index.resolve_save_root` via
     `commands.checkpoints._resolve_save_root_for_cwd`.
  2. Read `<save_root>/train.pid`; raise E_RUN_NOT_FOUND if missing or
     corrupt.
  3. Call `platform.process.terminate(pid)`.
  4. Release BOTH locks: per-job `<save_root>/.lock` and gpu-wide
     `~/.auto_ft/gpu.lock`.
  5. Echo `TerminationResult` JSON.
"""

from __future__ import annotations

import json
from typing import Annotated

import typer

from auto_ft.commands.checkpoints import _resolve_save_root_for_cwd
from auto_ft.errors import AutoFtError, ConfigError, ErrorCode
from auto_ft.platform.lock import release
from auto_ft.platform.process import terminate
from auto_ft.training.launcher import gpu_lock_path

app = typer.Typer(no_args_is_help=True, add_completion=False)


@app.callback(invoke_without_command=True)
def stop(
    job_name: Annotated[str, typer.Argument(help="Job name from .autoft/state.json")],
) -> None:
    """Call `platform.process.terminate`; release the GPU + per-job locks."""
    try:
        save_root = _resolve_save_root_for_cwd(job_name)
        pid_path = save_root / "train.pid"
        if not pid_path.exists():
            raise ConfigError(
                ErrorCode.E_RUN_NOT_FOUND,
                f"No train.pid for job {job_name!r} at {pid_path}",
                {"job_name": job_name, "save_root": str(save_root)},
            )
        try:
            pid = int(pid_path.read_text(encoding="utf-8").strip())
        except ValueError as e:
            raise ConfigError(
                ErrorCode.E_RUN_NOT_FOUND,
                f"Corrupt train.pid: {e}",
                {"pid_path": str(pid_path)},
            ) from e

        result = terminate(pid)
        # Release per-job lock first, then the GPU-wide lock.
        release(save_root / ".lock")
        release(gpu_lock_path())
        typer.echo(result.model_dump_json())
    except AutoFtError as exc:
        typer.echo(json.dumps(exc.to_json()))
        raise typer.Exit(1) from None
