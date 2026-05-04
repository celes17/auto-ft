"""Public platform API."""

from .lock import acquire, check_stale, release
from .process import is_alive, spawn_detached, terminate
from .types import LockInfo, ProcessStatus, TerminationResult

__all__ = [
    "LockInfo",
    "ProcessStatus",
    "TerminationResult",
    "acquire",
    "check_stale",
    "is_alive",
    "release",
    "spawn_detached",
    "terminate",
]
