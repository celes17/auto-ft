"""Log tailing + Ostris log format parsing + state-derivation helpers.

Filesystem is the source of truth for runtime state: state.json is a
static index; status is always computed on-demand here.

Ostris log format (verified at
ai-toolkit/jobs/process/BaseSDTrainProcess.py:2280-2282):
  - tqdm postfix: "lr: 1.0e-04 loss: 2.345e-01"
  - save events:  "Saving at step 500"
  - iter rate:    "2.10s/it"  (parseable from tqdm; absent in no-TTY
    subprocess)

tqdm in a non-TTY subprocess emits CR-prefixed progress chunks that
concatenate onto a single line. We tolerate this by stripping \\r before
regex and by accepting missing 's/it' annotations.
"""

from __future__ import annotations

import contextlib
import re
import time
from collections import deque
from pathlib import Path

from auto_ft.platform.process import is_alive
from auto_ft.training.artifacts import scan_checkpoints
from auto_ft.training.types import JobStatus

# Patterns verified against ai-toolkit source.
# Accept `lr: <float>` with an optional following `loss: <float>`.
LR_LOSS_RE = re.compile(r"lr:\s*([-+\d.eE]+)(?:.*?loss:\s*([-+\d.eE]+))?")
SAVE_RE = re.compile(r"Saving at step (\d+)")
IT_RATE_RE = re.compile(r"([-+\d.]+)s/it")

_DEFAULT_AVG_STEP_S = 2.0
_STALENESS_MULT = 2.0


def tail_lines(path: Path, n: int = 100) -> list[str]:
    """Return last n lines of a file (pure-Python deque — cross-platform).

    Returns [] if the file is missing (training may not have started yet).
    UTF-8 with errors='replace' handles mid-byte torn reads gracefully.
    """
    if not path.exists():
        return []
    with path.open(encoding="utf-8", errors="replace") as f:
        return list(deque(f, maxlen=n))


def parse_log_tail(lines: list[str]) -> tuple[int | None, float | None]:
    """Scan `lines`; return (last_save_step, last_tqdm_loss).

    Step is sourced from concrete "Saving at step N" events (not
    tqdm-implied). Loss is sourced from the most recent tqdm postfix line
    (CR-prefixed chunks are stripped before regex).
    """
    last_step: int | None = None
    last_loss: float | None = None
    for line in lines:
        # CR-only progress chunks from tqdm in non-TTY subprocess.
        stripped = line.replace("\r", "")
        if m := SAVE_RE.search(stripped):
            last_step = int(m.group(1))
        if (m := LR_LOSS_RE.search(stripped)) and m.group(2):
            with contextlib.suppress(ValueError):
                last_loss = float(m.group(2))
    return last_step, last_loss


def avg_step_time_from_log(lines: list[str], default_s: float = _DEFAULT_AVG_STEP_S) -> float:
    """Return rolling-average seconds-per-step from tqdm '<x>s/it' tokens.

    Scans the last 40 lines for signal; falls back to `default_s` when no
    'Ns/it' tokens parse (no-TTY subprocesses usually suppress these).
    """
    vals: list[float] = []
    for line in lines[-40:]:
        for m in IT_RATE_RE.finditer(line):
            try:
                vals.append(float(m.group(1)))
            except ValueError:
                continue
    if not vals:
        return default_s
    return sum(vals) / len(vals)


def derive_state(
    job_name: str,
    save_root: Path,
    pid: int | None,
    log_path: Path,
    total_steps: int | None = None,
) -> JobStatus:
    """Pure read: filesystem -> JobStatus. No cache, no writes.

    State precedence (highest first):
      1. `{name}.safetensors` final exists           -> "completed"
      2. pid alive AND log idle > 2 * avg_step_time  -> "stale"
      3. pid alive                                   -> "running"
      4. pid dead AND some checkpoints exist         -> "stopped"
      5. pid dead AND no checkpoints                 -> "failed"
    """
    checkpoints = scan_checkpoints(save_root)
    final_exists = any(c.is_final for c in checkpoints)
    step_checkpoints = [c for c in checkpoints if not c.is_final]
    current_step = max((c.step for c in step_checkpoints if c.step is not None), default=0)

    # 1. Completed — highest priority (final tensor exists).
    if final_exists:
        return JobStatus(
            job_name=job_name,
            state="completed",
            current_step=current_step,
            total_steps=total_steps,
            last_loss=None,
            eta_seconds=None,
            pid=pid,
        )

    # Parse log once; used for step/loss enrichment AND staleness calc.
    lines = tail_lines(log_path, n=200)
    last_save_step, last_loss = parse_log_tail(lines)
    if last_save_step is not None and last_save_step > current_step:
        current_step = last_save_step

    avg_step = avg_step_time_from_log(lines)
    eta: float | None = None
    if total_steps is not None and current_step and avg_step:
        eta = max(0.0, float(total_steps - current_step) * avg_step)

    # 2/3. Process alive branch.
    if pid is not None and is_alive(pid):
        # Stale if log has not grown for > STALENESS_MULT * avg_step_time.
        try:
            log_mtime = log_path.stat().st_mtime if log_path.exists() else 0.0
            idle = time.time() - log_mtime
            if idle > _STALENESS_MULT * avg_step:
                return JobStatus(
                    job_name=job_name,
                    state="stale",
                    current_step=current_step,
                    total_steps=total_steps,
                    last_loss=last_loss,
                    eta_seconds=eta,
                    pid=pid,
                )
        except OSError:
            pass
        return JobStatus(
            job_name=job_name,
            state="running",
            current_step=current_step,
            total_steps=total_steps,
            last_loss=last_loss,
            eta_seconds=eta,
            pid=pid,
        )

    # 4/5. Process dead branch.
    if checkpoints:
        return JobStatus(
            job_name=job_name,
            state="stopped",
            current_step=current_step,
            total_steps=total_steps,
            last_loss=last_loss,
            eta_seconds=None,
            pid=pid,
        )
    return JobStatus(
        job_name=job_name,
        state="failed",
        current_step=0,
        total_steps=total_steps,
        last_loss=None,
        eta_seconds=None,
        pid=pid,
    )
