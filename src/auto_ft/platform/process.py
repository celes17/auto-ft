"""Public process API.

Branches at import time on sys.platform; private impls in _windows / _posix.
"""

from __future__ import annotations

import sys

from .types import TerminationResult

if sys.platform == "win32":
    from . import _windows as _impl
else:
    from . import _posix as _impl


def spawn_detached(cmd: list[str], cwd: str, env: dict[str, str], log_path: str) -> int:
    return _impl.spawn_detached(cmd, cwd, env, log_path)


def is_alive(pid: int) -> bool:
    return _impl.is_alive(pid)


def terminate(pid: int, grace_s: float = 5.0) -> TerminationResult:
    return _impl.terminate(pid, grace_s)
