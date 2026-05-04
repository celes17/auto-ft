"""Stable error codes and exception hierarchy."""

from __future__ import annotations

from enum import StrEnum
from typing import Any


class ErrorCode(StrEnum):
    # Asset / data
    E_CHECKPOINT_NOT_FOUND = "E_CHECKPOINT_NOT_FOUND"
    E_INSUFFICIENT_DATA = "E_INSUFFICIENT_DATA"
    E_LOW_RES = "E_LOW_RES"
    E_SAMPLES_MISSING = "E_SAMPLES_MISSING"
    E_DISK_FULL = "E_DISK_FULL"
    E_PERMISSION = "E_PERMISSION"

    # Recipe / config
    E_UNKNOWN_RECIPE = "E_UNKNOWN_RECIPE"
    E_RECIPE_INVALID = "E_RECIPE_INVALID"
    E_OSTRIS_CONFIG_MISSING = "E_OSTRIS_CONFIG_MISSING"

    # Training / runtime
    E_TRAINING_CRASHED = "E_TRAINING_CRASHED"
    E_GPU_BUSY = "E_GPU_BUSY"
    E_GPU_OOM = "E_GPU_OOM"
    E_DEGENERATE_OUTPUT = "E_DEGENERATE_OUTPUT"
    E_RUN_NOT_FOUND = "E_RUN_NOT_FOUND"
    E_CAPTION_FAILED = "E_CAPTION_FAILED"

    # Evaluator
    E_MISSING_CITATIONS = "E_MISSING_CITATIONS"
    E_INVALID_CONFIDENCE = "E_INVALID_CONFIDENCE"
    E_INVALID_FAULTS = "E_INVALID_FAULTS"
    E_CORRUPT_EVAL = "E_CORRUPT_EVAL"


class AutoFtError(Exception):
    """Base for every auto_ft-raised exception.

    Carries a stable code + JSON-serialisable details.
    """

    def __init__(
        self, code: ErrorCode, message: str, details: dict[str, Any] | None = None
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details or {}

    def to_json(self) -> dict[str, Any]:
        return {
            "error_code": self.code.value,
            "message": self.message,
            "details": self.details,
        }


class ConfigError(AutoFtError):
    """Configuration / recipe / pyproject / pydantic-validation failure."""


class OpsError(AutoFtError):
    """Operational failure: process, GPU, disk, training-time issue."""


class AgentError(AutoFtError):
    """Evaluator sub-agent contract violation."""
