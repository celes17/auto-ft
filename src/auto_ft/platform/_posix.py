"""POSIX process primitives. NOT CI-exercised (T-05); mocked-tested only via Option-B mocking."""

from __future__ import annotations

import contextlib
import os
import signal
import subprocess
import time

from .types import TerminationResult

# SIGKILL is POSIX-only; on Windows the `signal` module lacks this attribute.
# Bind signal.SIGTERM and signal.SIGKILL at module-import time with a Windows
# fallback so Option-B mocking (which runs on windows-latest CI) can exercise
# this module without hitting AttributeError. On POSIX, these resolve to the
# real signal.SIGTERM / signal.SIGKILL constants (15 / 9).
try:
    _SIGTERM = signal.SIGTERM
except AttributeError:  # pragma: no cover — signal.SIGTERM exists on Win+POSIX
    _SIGTERM = 15
try:
    _SIGKILL = signal.SIGKILL
except AttributeError:  # Windows: signal has no SIGKILL attr; 9 is the POSIX value
    _SIGKILL = 9


def spawn_detached(cmd: list[str], cwd: str, env: dict[str, str], log_path: str) -> int:
    log_fp = open(log_path, "ab")  # noqa: SIM115
    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=log_fp,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        close_fds=True,
    )
    return proc.pid


def is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # Process exists, we just can't signal it
    return True


def terminate(pid: int, grace_s: float = 5.0) -> TerminationResult:
    try:
        os.kill(pid, _SIGTERM)
    except ProcessLookupError:
        return TerminationResult(clean=True, signal_sent="SIGTERM")
    deadline = time.monotonic() + grace_s
    while time.monotonic() < deadline:
        if not is_alive(pid):
            return TerminationResult(clean=True, signal_sent="SIGTERM")
        time.sleep(0.1)
    # Escalate
    with contextlib.suppress(ProcessLookupError):
        os.kill(pid, _SIGKILL)
    # Brief final poll
    for _ in range(20):
        if not is_alive(pid):
            break
        time.sleep(0.05)
    return TerminationResult(clean=False, signal_sent="SIGKILL")
