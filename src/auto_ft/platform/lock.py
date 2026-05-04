"""File-based lockfile.

Caller is responsible for providing a resolved path within a trusted
directory; this module does NOT sanitize or resolve paths.
"""

from __future__ import annotations

import contextlib
import json
import os
import socket
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from .process import is_alive
from .types import LockInfo

LOCK_VERSION = 1


def acquire(path: str | os.PathLike, pid: int) -> LockInfo:
    """Write a lockfile atomically.

    Caller is responsible for calling check_stale() first if they want to detect
    contention.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "pid": pid,
        "started_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "hostname": socket.gethostname(),
        "version": LOCK_VERSION,
    }
    fd, tmp = tempfile.mkstemp(dir=p.parent, prefix=".lock-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(payload, f)
        os.replace(tmp, p)
    except Exception:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp)
        raise
    return LockInfo(**payload, stale=False)


def release(path: str | os.PathLike) -> None:
    with contextlib.suppress(FileNotFoundError):
        os.unlink(path)


def check_stale(path: str | os.PathLike) -> LockInfo | None:
    """Return LockInfo with stale=True if the lockfile's pid is dead.

    Returns None if no lockfile exists. No start-time match heuristic.
    """
    p = Path(path)
    if not p.exists():
        return None
    try:
        with p.open() as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        # Corrupt lock — treat as stale
        return LockInfo(
            pid=-1,
            started_at=datetime.now(UTC),
            hostname="?",
            version=LOCK_VERSION,
            stale=True,
        )
    info = LockInfo(**data, stale=False)
    info.stale = not is_alive(info.pid)
    return info
