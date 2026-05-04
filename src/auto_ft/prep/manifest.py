"""Prep manifest read/write + dataset SHA-256 hash.

Atomic write via tempfile.mkstemp + os.replace; last-writer-wins is
acceptable for current usage.

Hash stability:
  - Sorted filename ordering → deterministic on Windows and POSIX.
  - First 1KB of each file → captures PNG/JPEG magic+header; edits to
    image bytes change the hash.
  - Filenames joined with \\0 separator → prevents adjacent-rename
    collisions.
"""

from __future__ import annotations

import contextlib
import hashlib
import os
import tempfile
from pathlib import Path

from auto_ft.prep.types import PrepManifest
from auto_ft.prep.validate import ALLOWED_EXTS

MANIFEST_FILENAME = ".auto_ft_prep.json"


def list_candidate_images(dataset_dir: Path) -> list[Path]:
    """Return sorted image paths (deterministic ordering for hash stability)."""
    return sorted(
        p for p in dataset_dir.iterdir() if p.is_file() and p.suffix.lower() in ALLOWED_EXTS
    )


def compute_dataset_sha(dataset_dir: Path) -> str:
    """SHA-256 over `count:N\\n` + (filename, \\0, first-1KB) per image.

    Manifest filename is excluded by virtue of ALLOWED_EXTS filter
    (`.auto_ft_prep.json` is not an image extension).
    """
    h = hashlib.sha256()
    images = list_candidate_images(dataset_dir)
    h.update(f"count:{len(images)}\n".encode())
    for img in images:
        h.update(img.name.encode("utf-8"))
        h.update(b"\0")
        with img.open("rb") as f:
            h.update(f.read(1024))
    return h.hexdigest()


def write_manifest(path: Path, manifest: PrepManifest) -> None:
    """Atomic write: tempfile.mkstemp in parent dir + os.replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".prep-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(manifest.model_dump_json(indent=2))
        os.replace(tmp, path)
    except Exception:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp)
        raise


def read_manifest(path: Path) -> PrepManifest | None:
    """Return parsed PrepManifest or None if path is absent / not a valid manifest."""
    if not path.exists():
        return None
    try:
        return PrepManifest.model_validate_json(path.read_text(encoding="utf-8"))
    except Exception:
        return None
