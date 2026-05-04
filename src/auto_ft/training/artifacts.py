"""Checkpoint + sample filesystem scan.

Ostris-native save_root layout:

  <save_root>/                            # <training_folder>/<job_name>/
    {name}_{step:09d}.safetensors         # interim checkpoints (step parsed)
    {name}.safetensors                    # final checkpoint (no step suffix)
    samples/
      {name}_{step:09d}_{prompt_idx}.jpg  # flat layout (primary)
      step_{step:09d}/                    # subdir layout (fallback)
        *.jpg | *.png
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from pathlib import Path

from auto_ft.training.types import CheckpointInfo, SampleInfo

# Ostris naming: `{name}_{step:09d}.safetensors` for interim; `{name}.safetensors` final.
STEP_FILENAME_RE = re.compile(r"_(\d{9})\.safetensors$")
# Matches `_NNNNNNNNN_` (prompt-index follows) or `_NNNNNNNNN.` (extension follows).
# Also tolerant of fewer digits for non-Ostris fixtures in tests.
SAMPLE_STEP_RE = re.compile(r"_(\d+)(?:_|\.)")


def _parse_step_from_checkpoint(filename: str) -> int | None:
    m = STEP_FILENAME_RE.search(filename)
    return int(m.group(1)) if m else None


def scan_checkpoints(save_root: Path) -> list[CheckpointInfo]:
    """Return all `{save_root.name}*.safetensors` entries under `save_root`.

    Sort order: interim checkpoints ascending by step, final (`is_final=True`)
    last. Empty list when the directory is absent or holds no matching files.
    """
    if not save_root.exists():
        return []
    name = save_root.name
    results: list[CheckpointInfo] = []
    for p in save_root.glob(f"{name}*.safetensors"):
        step = _parse_step_from_checkpoint(p.name)
        # `is_final` is exactly `{name}.safetensors` — stem equals the job name.
        is_final = step is None and p.stem == name
        if step is None and not is_final:
            continue  # sibling files that happen to match prefix
        st = p.stat()
        results.append(
            CheckpointInfo(
                path=str(p),
                step=step,
                is_final=is_final,
                size_bytes=st.st_size,
                mtime=datetime.fromtimestamp(st.st_mtime, tz=UTC),
            )
        )
    # interim ascending, then final last — `(is_final, step)` sorts correctly
    # because False < True and None-step interims can't appear (filtered above).
    results.sort(key=lambda c: (c.is_final, c.step if c.step is not None else -1))
    return results


def scan_samples(save_root: Path, step: int | None = None) -> list[SampleInfo]:
    """Return sample image entries under `save_root/samples/`.

    Supports both flat layout (primary Ostris behaviour) and
    subdir-per-step layout via `rglob`. Step is parsed from the filename
    substring or the parent `step_NNNN` directory name. Filter by
    `step=N` (exact match) or pass `None` for all.
    """
    samples_dir = save_root / "samples"
    if not samples_dir.exists():
        return []
    results: list[SampleInfo] = []
    for ext in ("jpg", "jpeg", "png"):
        for p in samples_dir.rglob(f"*.{ext}"):
            sample_step = _extract_sample_step(p)
            if sample_step is None:
                continue
            if step is not None and sample_step != step:
                continue
            results.append(
                SampleInfo(
                    path=str(p),
                    step=sample_step,
                    prompt_index=None,
                )
            )
    results.sort(key=lambda s: (s.step, s.path))
    return results


def _extract_sample_step(path: Path) -> int | None:
    """Parse step from the parent directory name first, then the filename.

    Accepted patterns:
      samples/step_000000050/img.png  -> 50   (subdir layout)
      samples/myjob_000000050_0.jpg   -> 50   (flat layout)
      samples/myjob_000000050.jpg     -> 50
    """
    parent_name = path.parent.name
    if parent_name.startswith("step_"):
        try:
            return int(parent_name[len("step_") :])
        except ValueError:
            pass
    m = SAMPLE_STEP_RE.search(path.name)
    if m:
        return int(m.group(1))
    return None
