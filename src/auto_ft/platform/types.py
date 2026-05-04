"""Cross-module return contracts for the platform layer. Pydantic v2 per P-01."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel


class TerminationResult(BaseModel):
    clean: bool
    signal_sent: str  # "SIGTERM" | "SIGKILL" | "TerminateProcess"


class LockInfo(BaseModel):
    pid: int
    started_at: datetime
    hostname: str
    version: int = 1
    stale: bool = False


class ProcessStatus(BaseModel):
    alive: bool
    pid: int
