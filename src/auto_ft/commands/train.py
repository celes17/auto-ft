"""auto_ft train <cfg.yaml> — non-blocking detached Ostris spawn.

Resume is 100% Ostris-native: re-running `auto_ft train cfg.yaml` with
existing checkpoints under `save_root` triggers Ostris's internal
`get_latest_save_path()` path. There is deliberately no `--continue` flag.

First-invocation auto-register: if the cfg's `config.name` is not yet
present in `<cwd>/.autoft/state.json`, append a `JobEntry` and write it.
Re-invocation with the same name is a no-op (idempotent — no duplicate,
no force-overwrite).

JSON error contract: AutoFtError subclasses are caught at the callback
boundary, emitted as JSON to stdout, exit(1).
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

import typer
import yaml

from auto_ft.errors import AutoFtError, ConfigError, ErrorCode
from auto_ft.training.launcher import launch_ostris
from auto_ft.training.state_index import (
    create_state,
    read_state,
    register_job,
    state_path,
    write_state,
)
from auto_ft.training.types import JobEntry

app = typer.Typer(no_args_is_help=True, add_completion=False)


def _auto_register_job(cfg_path: Path, project_root: Path) -> None:
    """Idempotently append the cfg's `config.name` to `.autoft/state.json`.

    Creates the state.json if absent. If the job name already exists, this is
    a no-op (no duplicate, no force-overwrite). Raises `ConfigError` for
    malformed / missing-key cfg inputs to mirror the launcher's validation.
    """
    try:
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as e:
        raise ConfigError(
            ErrorCode.E_RECIPE_INVALID,
            f"Malformed YAML: {e}",
            {"path": str(cfg_path)},
        ) from e
    if not isinstance(cfg, dict):
        raise ConfigError(
            ErrorCode.E_RECIPE_INVALID,
            "YAML root must be a mapping",
            {"path": str(cfg_path)},
        )
    try:
        name = cfg["config"]["name"]
        training_folder = cfg["config"]["process"][0]["training_folder"]
    except (KeyError, IndexError, TypeError) as e:
        raise ConfigError(
            ErrorCode.E_RECIPE_INVALID,
            f"Missing required key: {e}",
            {"path": str(cfg_path), "missing": str(e)},
        ) from e

    path = state_path(project_root)
    state = read_state(path)
    if state is None:
        state = create_state(project_root)
        write_state(path, state)

    if any(j.name == name for j in state.jobs):
        return  # idempotent — already registered

    trigger = ""
    try:
        trigger = cfg["config"]["process"][0].get("trigger_word", "") or ""
    except (KeyError, IndexError, TypeError):
        trigger = ""

    entry = JobEntry(
        name=name,
        recipe="",  # unknown at CLI level; set via `auto_ft init --set-defaults`
        dataset_path="",
        trigger=trigger,
        training_folder=str(training_folder),
        created_at=datetime.now(UTC),
    )
    register_job(path, entry, force=False)


@app.callback(invoke_without_command=True)
def train(
    config_path: Annotated[
        Path,
        typer.Argument(
            exists=True,
            dir_okay=False,
            resolve_path=True,
            help="Path to Ostris cfg.yaml.",
        ),
    ],
) -> None:
    """Spawn Ostris training detached. Returns JSON `{job_name, pid, state, log_path}` in <2s."""
    try:
        cfg_path = Path(config_path).resolve()
        project_root = Path.cwd().resolve()
        _auto_register_job(cfg_path, project_root)
        result = launch_ostris(cfg_path)
        typer.echo(result.model_dump_json())
    except AutoFtError as exc:
        typer.echo(json.dumps(exc.to_json()))
        raise typer.Exit(1) from None
