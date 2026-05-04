"""Prep-layer return contracts. Pydantic v2."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel


class PrepManifest(BaseModel):
    version: str = "1.0"
    image_count: int
    trigger: str
    resolutions: dict[str, int]  # {"1024x1024": 12, "1024x768": 3}
    dataset_sha: str  # SHA-256 over sorted filenames + first 1KB
    created_at: datetime


class PrepareResult(BaseModel):
    dataset_path: str
    manifest_path: str
    image_count: int
    trigger: str
    cached: bool  # True when existing manifest matched hash
