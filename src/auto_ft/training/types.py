"""Cross-module return contracts for the training layer.

JobEntry pins `model_config = ConfigDict(extra="ignore")` so legacy
state.json files containing extra keys continue to deserialise without
raising — pinned rather than defaulted so the behaviour survives future
pydantic default-behaviour changes or project-wide strict-mode flips.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict


class LaunchResult(BaseModel):
    job_name: str
    pid: int
    state: str  # "running" | "failed"
    log_path: str


class JobStatus(BaseModel):
    job_name: str
    state: str  # "running" | "completed" | "stopped" | "stale" | "failed"
    current_step: int
    total_steps: int | None = None
    last_loss: float | None = None
    eta_seconds: float | None = None
    pid: int | None = None


class CheckpointInfo(BaseModel):
    path: str
    step: int | None  # None for final (no step suffix)
    is_final: bool
    size_bytes: int
    mtime: datetime


class SampleInfo(BaseModel):
    path: str
    step: int
    prompt_index: int | None = None


class JobEntry(BaseModel):
    """One registered training job within the project's state index.

    Carries explicit `extra="ignore"` so legacy state.json files with
    extra keys deserialise without raising. Pinned rather than defaulted
    so the behaviour survives future pydantic default-behaviour changes
    or project-wide strict-mode flips.
    """

    model_config = ConfigDict(extra="ignore")

    name: str
    recipe: str
    dataset_path: str
    trigger: str
    training_folder: str
    created_at: datetime


class StateIndex(BaseModel):
    version: str = "1.0"
    project_root: str
    created_at: datetime
    updated_at: datetime
    default_recipe: str | None = None
    default_trigger: str | None = None
    jobs: list[JobEntry] = []
