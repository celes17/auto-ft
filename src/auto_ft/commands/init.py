"""auto_ft init — project initialisation.

Creates `<cwd>/.autoft/state.json` (idempotent), registers `<job_name>`
(uniqueness check, filesystem-safe regex), and optionally sets
default_recipe / default_trigger.

The default training_folder is the PARENT directory (`<cwd>/output`),
NOT `<cwd>/output/<name>`. Ostris appends `/<name>` via its internal
`save_root = os.path.join(training_folder, name)` — auto_ft MUST NOT
pre-append it. The `resolve_save_root(state, job_name)` helper in
`training.state_index` is the single point of truth for save_root
construction.

JSON error-output contract: AutoFtError subclasses are caught in the
callback, serialised to stdout, exit(1). Mirrors cli._entry.
"""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

import typer

from auto_ft.errors import AutoFtError, ConfigError, ErrorCode
from auto_ft.training.state_index import (
    create_state,
    read_state,
    register_job,
    set_defaults,
    state_path,
    write_state,
)
from auto_ft.training.types import JobEntry

# Kebab-case recommended, filesystem-safe required.
# Start-char must be alphanumeric; tail allows [A-Za-z0-9_-]; max 63
# chars (the typical filesystem path-component safe boundary). Rejects
# spaces, dots, slashes, backslashes, and NUL by construction.
JOB_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,62}$")

app = typer.Typer(no_args_is_help=True, add_completion=False)


@app.callback(invoke_without_command=True)
def init_cmd(
    name: Annotated[str, typer.Option("--name", help="Job name (kebab-case; filesystem-safe)")],
    recipe: Annotated[
        str | None, typer.Option("--recipe", help="Recipe to record for this job.")
    ] = None,
    trigger: Annotated[
        str | None, typer.Option("--trigger", help="Trigger word to record.")
    ] = None,
    dataset: Annotated[
        Path | None,
        typer.Option("--dataset", help="Dataset directory to record for this job."),
    ] = None,
    training_folder: Annotated[
        Path | None,
        typer.Option(
            "--training-folder",
            help="PARENT output dir; Ostris appends /<name> (default: <cwd>/output).",
        ),
    ] = None,
    force: Annotated[
        bool, typer.Option("--force", help="Overwrite an existing same-name job.")
    ] = False,
    set_defaults_only: Annotated[
        bool,
        typer.Option(
            "--set-defaults",
            help="Update default_recipe / default_trigger without registering a new job.",
        ),
    ] = False,
) -> None:
    """Create .autoft/state.json in <cwd> and register the job."""
    try:
        _run_init(
            name=name,
            recipe=recipe,
            trigger=trigger,
            dataset=dataset,
            training_folder=training_folder,
            force=force,
            set_defaults_only=set_defaults_only,
        )
    except AutoFtError as exc:
        typer.echo(json.dumps(exc.to_json()))
        raise typer.Exit(1) from None


def _run_init(
    *,
    name: str,
    recipe: str | None,
    trigger: str | None,
    dataset: Path | None,
    training_folder: Path | None,
    force: bool,
    set_defaults_only: bool,
) -> None:
    if not JOB_NAME_RE.match(name):
        raise ConfigError(
            ErrorCode.E_RECIPE_INVALID,
            f"Invalid job name '{name}' (must match {JOB_NAME_RE.pattern})",
            {"job_name": name},
        )

    project_root = Path.cwd().resolve()
    path = state_path(project_root)

    # Idempotent create: reuse existing state if present, else init fresh.
    state = read_state(path)
    if state is None:
        state = create_state(project_root)
        write_state(path, state)

    if set_defaults_only:
        updated = set_defaults(path, default_recipe=recipe, default_trigger=trigger)
        typer.echo(updated.model_dump_json())
        return

    # training_folder is the PARENT — NEVER append /<name> here.
    # Ostris does the `save_root = os.path.join(training_folder, name)` join.
    entry = JobEntry(
        name=name,
        recipe=recipe or "",
        dataset_path=str(dataset.resolve()) if dataset else "",
        trigger=trigger or "",
        training_folder=(
            str(training_folder.resolve()) if training_folder else str(project_root / "output")
        ),
        created_at=datetime.now(UTC),
    )
    register_job(path, entry, force=force)

    final_state = read_state(path)
    assert final_state is not None  # just wrote it
    typer.echo(final_state.model_dump_json())
