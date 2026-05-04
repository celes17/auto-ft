"""`.autoft/state.json` read/write — the per-project job index.

Shape: {version, project_root, created_at, updated_at, default_recipe,
default_trigger, jobs[]}.

Runtime state (running/stopped/step/loss) is NOT cached here — the
filesystem (`save_root/`) is the source of truth. See
`training/status_parser.py` and `training/artifacts.py` for those
derived views.

`training_folder` canonicalisation: the stored value is the PARENT
directory, WITHOUT `/job_name` appended. `resolve_save_root(state,
job_name)` below applies the `/job_name` rule — it is the single point
of truth for save_root construction.
"""

from __future__ import annotations

import contextlib
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from auto_ft.errors import ConfigError, ErrorCode, OpsError
from auto_ft.training.types import JobEntry, StateIndex

STATE_FILENAME = "state.json"
STATE_DIRNAME = ".autoft"


def state_path(project_root: Path) -> Path:
    """Canonical state.json location for a project root."""
    return project_root / STATE_DIRNAME / STATE_FILENAME


def read_state(path: Path) -> StateIndex | None:
    """Return parsed StateIndex or None if file is absent or not a valid state."""
    if not path.exists():
        return None
    try:
        return StateIndex.model_validate_json(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def write_state(path: Path, state: StateIndex) -> None:
    """Atomic write via tempfile.mkstemp + os.replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".state-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(state.model_dump_json(indent=2))
        os.replace(tmp, path)
    except Exception:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp)
        raise


def create_state(project_root: Path) -> StateIndex:
    """Return a freshly-initialised StateIndex for `project_root`."""
    now = datetime.now(UTC)
    return StateIndex(
        version="1.0",
        project_root=str(project_root.resolve()),
        created_at=now,
        updated_at=now,
        default_recipe=None,
        default_trigger=None,
        jobs=[],
    )


def register_job(path: Path, entry: JobEntry, force: bool = False) -> StateIndex:
    """Append (or replace with force=True) a job entry in the state index.

    Raises ConfigError(E_RECIPE_INVALID) on duplicate job_name unless
    `force=True`. Raises ConfigError(E_RUN_NOT_FOUND) if state.json is
    missing — the caller (init command) is expected to have created the
    state first.
    """
    state = read_state(path)
    if state is None:
        raise ConfigError(
            ErrorCode.E_RUN_NOT_FOUND,
            f"state.json not found at {path}",
            {"path": str(path)},
        )
    existing = [j for j in state.jobs if j.name == entry.name]
    if existing and not force:
        raise ConfigError(
            ErrorCode.E_RECIPE_INVALID,
            f"Job name '{entry.name}' already registered (use --force to override)",
            {"job_name": entry.name, "collision": True},
        )
    state.jobs = [j for j in state.jobs if j.name != entry.name] + [entry]
    state.updated_at = datetime.now(UTC)
    write_state(path, state)
    return state


def set_defaults(
    path: Path,
    default_recipe: str | None = None,
    default_trigger: str | None = None,
) -> StateIndex:
    """Update default_recipe / default_trigger on the state index."""
    state = read_state(path)
    if state is None:
        raise ConfigError(
            ErrorCode.E_RUN_NOT_FOUND,
            f"state.json not found at {path}",
            {"path": str(path)},
        )
    if default_recipe is not None:
        state.default_recipe = default_recipe
    if default_trigger is not None:
        state.default_trigger = default_trigger
    state.updated_at = datetime.now(UTC)
    write_state(path, state)
    return state


def resolve_save_root(state: StateIndex, job_name: str) -> Path:
    """Return `Path(job.training_folder) / job_name` for the named job.

    `state.json.jobs[*].training_folder` is the PARENT directory, WITHOUT
    the job_name appended. Ostris owns the final `/name` join via its
    internal `save_root = os.path.join(training_folder, name)`. auto_ft
    applies the same `/job_name` rule here so callers land on the same
    `<parent>/<job_name>/` path Ostris writes to.

    Raises OpsError(E_RUN_NOT_FOUND) if `job_name` is not registered.
    """
    job = next((j for j in state.jobs if j.name == job_name), None)
    if job is None:
        raise OpsError(
            ErrorCode.E_RUN_NOT_FOUND,
            f"Unknown job: {job_name}",
            {"job_name": job_name, "known_jobs": [j.name for j in state.jobs]},
        )
    return Path(job.training_folder) / job_name
