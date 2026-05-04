"""Ostris environment config loader — `~/.auto_ft/config.toml`.

Single source of truth for Ostris run.py and Python interpreter paths.
Resolution hierarchy is owned by `auto_ft.training.launcher`; this module
is a thin TOML reader with validation.

Schema (all keys optional at load time — launcher enforces which are required):

    [ostris]
    run_py = "C:/path/to/ai-toolkit/run.py"
    python = "C:/path/to/standalone/python.exe"   # standalone interpreter
    venv   = "C:/path/to/non-standard/venv"        # or a venv root
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

from auto_ft.errors import ConfigError, ErrorCode

CONFIG_PATH = Path.home() / ".auto_ft" / "config.toml"

_KNOWN_KEYS = frozenset({"run_py", "python", "venv"})


def load_ostris_config(path: Path | None = None) -> dict[str, Any]:
    """Return the `[ostris]` table from `~/.auto_ft/config.toml`.

    Returns an empty dict when the file is absent (launcher then falls
    through to env vars / auto-derive / error as appropriate).

    Raises:
      ConfigError(E_OSTRIS_CONFIG_MISSING): malformed TOML or unknown keys.
    """
    cfg_path = path or CONFIG_PATH
    if not cfg_path.exists():
        return {}
    try:
        with cfg_path.open("rb") as f:
            data = tomllib.load(f)
    except tomllib.TOMLDecodeError as e:
        raise ConfigError(
            ErrorCode.E_OSTRIS_CONFIG_MISSING,
            f"Malformed TOML in {cfg_path}: {e}",
            {"path": str(cfg_path)},
        ) from e

    ostris = data.get("ostris") or {}
    if not isinstance(ostris, dict):
        raise ConfigError(
            ErrorCode.E_OSTRIS_CONFIG_MISSING,
            f"[ostris] must be a table in {cfg_path}",
            {"path": str(cfg_path)},
        )

    unknown = set(ostris) - _KNOWN_KEYS
    if unknown:
        raise ConfigError(
            ErrorCode.E_OSTRIS_CONFIG_MISSING,
            f"Unknown [ostris] keys in {cfg_path}: {sorted(unknown)}. "
            f"Allowed: {sorted(_KNOWN_KEYS)}",
            {"path": str(cfg_path), "unknown_keys": sorted(unknown)},
        )
    return ostris
