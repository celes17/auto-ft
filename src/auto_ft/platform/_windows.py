"""Windows process primitives. Uses stdlib only (no psutil dep)."""

from __future__ import annotations

import ctypes
import subprocess
import time

from .types import TerminationResult

# Win32 constants (avoid importing win32con; ctypes-only)
PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
STILL_ACTIVE = 259


def spawn_detached(cmd: list[str], cwd: str, env: dict[str, str], log_path: str) -> int:
    flags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
    log_fp = open(log_path, "ab")  # noqa: SIM115 — child owns the handle for its lifetime
    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=log_fp,
        stderr=subprocess.STDOUT,
        creationflags=flags,
        close_fds=True,
    )
    return proc.pid


def is_alive(pid: int) -> bool:
    """O(1) liveness check via OpenProcess + GetExitCodeProcess.

    Returns False for: nonexistent PID, zombie/exited PID, access-denied PID
    (latter is fine for our use-case since we only inspect our own subprocesses).
    """
    h = ctypes.windll.kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not h:
        return False
    try:
        exit_code = ctypes.c_ulong()
        ok = ctypes.windll.kernel32.GetExitCodeProcess(h, ctypes.byref(exit_code))
        if not ok:
            return False
        return exit_code.value == STILL_ACTIVE
    finally:
        ctypes.windll.kernel32.CloseHandle(h)


def terminate(pid: int, grace_s: float = 5.0) -> TerminationResult:
    """Windows terminate is non-graceful (TerminateProcess)."""
    h = ctypes.windll.kernel32.OpenProcess(0x0001 | PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not h:
        # Already dead
        return TerminationResult(clean=True, signal_sent="TerminateProcess")
    try:
        ctypes.windll.kernel32.TerminateProcess(h, 1)
    finally:
        ctypes.windll.kernel32.CloseHandle(h)
    # Brief poll to let the OS reap
    deadline = time.monotonic() + min(grace_s, 2.0)
    while time.monotonic() < deadline:
        if not is_alive(pid):
            break
        time.sleep(0.05)
    return TerminationResult(clean=False, signal_sent="TerminateProcess")
