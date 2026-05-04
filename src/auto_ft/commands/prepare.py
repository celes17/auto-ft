"""auto_ft prepare — dataset validation + prep manifest.

Validates count / format / resolution / PIL integrity; writes a SHA-256
manifest; idempotent.

JSON error-output contract: AutoFtError subclasses (ConfigError /
OpsError) are caught in the callback, serialised to stdout via
to_json(), and exit(1). This duplicates cli._entry's top-level catch —
necessary because `typer.testing.CliRunner().invoke(app, ...)` bypasses
`_entry`, and the command's JSON contract is what downstream tests parse.
"""

from __future__ import annotations

import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

import typer

from auto_ft.errors import AutoFtError, ConfigError, ErrorCode
from auto_ft.prep.manifest import (
    MANIFEST_FILENAME,
    compute_dataset_sha,
    list_candidate_images,
    read_manifest,
    write_manifest,
)
from auto_ft.prep.types import PrepareResult, PrepManifest
from auto_ft.prep.validate import validate_image

MIN_IMAGES = 1  # raw count floor.

app = typer.Typer(no_args_is_help=True, add_completion=False)


@app.callback(invoke_without_command=True)
def prepare(
    dataset_dir: Annotated[
        Path,
        typer.Argument(exists=True, file_okay=False, dir_okay=True, resolve_path=True),
    ],
    trigger: Annotated[str, typer.Option("--trigger", help="Trigger word (e.g. 'zyx')")],
) -> None:
    """Validate images, write hash manifest, JSON-print PrepareResult."""
    try:
        _run_prepare(dataset_dir, trigger)
    except AutoFtError as exc:
        typer.echo(json.dumps(exc.to_json()))
        raise typer.Exit(1) from None


def _run_prepare(dataset_dir: Path, trigger: str) -> None:
    dataset_dir = dataset_dir.resolve()
    manifest_path = dataset_dir / MANIFEST_FILENAME
    dataset_sha = compute_dataset_sha(dataset_dir)

    # Idempotency: existing manifest with matching hash + trigger → skip.
    existing = read_manifest(manifest_path)
    if existing is not None and existing.dataset_sha == dataset_sha and existing.trigger == trigger:
        typer.echo(
            PrepareResult(
                dataset_path=str(dataset_dir),
                manifest_path=str(manifest_path),
                image_count=existing.image_count,
                trigger=trigger,
                cached=True,
            ).model_dump_json()
        )
        return

    images = list_candidate_images(dataset_dir)
    if len(images) < MIN_IMAGES:
        raise ConfigError(
            ErrorCode.E_INSUFFICIENT_DATA,
            f"Need at least {MIN_IMAGES} image(s); found {len(images)} in {dataset_dir}",
            {"dataset_path": str(dataset_dir), "image_count": len(images)},
        )

    resolutions: Counter[str] = Counter()
    low_res_paths: list[dict] = []
    invalid_paths: list[dict] = []
    for img in images:
        ok, reason, size = validate_image(img)
        if ok:
            assert size is not None
            resolutions[f"{size[0]}x{size[1]}"] += 1
        elif reason and reason.startswith("low_res"):
            low_res_paths.append({"path": str(img), "size": list(size) if size else None})
        else:
            invalid_paths.append({"path": str(img), "reason": reason})

    if low_res_paths:
        raise ConfigError(
            ErrorCode.E_LOW_RES,
            f"{len(low_res_paths)} image(s) below 512x512",
            {"paths": low_res_paths},
        )
    if invalid_paths:
        raise ConfigError(
            ErrorCode.E_RECIPE_INVALID,
            f"{len(invalid_paths)} image(s) failed PIL integrity check",
            {"paths": invalid_paths},
        )

    manifest = PrepManifest(
        version="1.0",
        image_count=len(images),
        trigger=trigger,
        resolutions=dict(resolutions),
        dataset_sha=dataset_sha,
        created_at=datetime.now(UTC),
    )
    write_manifest(manifest_path, manifest)
    typer.echo(
        PrepareResult(
            dataset_path=str(dataset_dir),
            manifest_path=str(manifest_path),
            image_count=len(images),
            trigger=trigger,
            cached=False,
        ).model_dump_json()
    )
