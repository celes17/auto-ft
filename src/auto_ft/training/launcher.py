"""Ostris subprocess launcher.

Wraps `python run.py cfg.yaml` via `platform.process.spawn_detached`.
Owns the GPU-wide lock at `~/.auto_ft/gpu.lock` and a per-job `.lock`
(co-located with Ostris `save_root`). Resume semantics are delegated
entirely to Ostris — there is no `--continue` flag; re-running
`auto_ft train cfg.yaml` with existing checkpoints in `save_root` triggers
Ostris-native resume via its internal `get_latest_save_path()`.

Trust boundaries:
  * YAML load uses `yaml.safe_load` exclusively.
  * `training_folder` MUST be absolute.
  * `VIRTUAL_ENV` is stripped from the subprocess env.
  * `cmd=` is always a list (`shell=True` is never used).

`training_folder` canonicalisation: `cfg.config.process[0].training_folder`
is the PARENT directory. Ostris appends `/<config.name>` internally — this
launcher must NOT pre-append.
"""

from __future__ import annotations

import os
from pathlib import Path

import yaml

from auto_ft.errors import ConfigError, ErrorCode, OpsError
from auto_ft.platform.config import CONFIG_PATH, load_ostris_config
from auto_ft.platform.lock import acquire, check_stale
from auto_ft.platform.process import spawn_detached
from auto_ft.training.types import LaunchResult


def _venv_python_from_root(venv_root: Path, *, windows: bool | None = None) -> Path:
    """Given a venv root, return the interpreter path for the current OS.

    `windows` defaults to the actual host OS; tests may override.
    """
    if windows is None:
        windows = os.name == "nt"
    if windows:
        return venv_root / "Scripts" / "python.exe"
    return venv_root / "bin" / "python"


def _resolve_ostris_run_py(config: dict | None = None) -> Path:
    """Resolve Ostris `run.py`.

    Resolution order:
      1. `AUTO_FT_OSTRIS_RUN_PY` env var (explicit override).
      2. `[ostris].run_py` in `~/.auto_ft/config.toml`.
      3. Raise `ConfigError(E_OSTRIS_CONFIG_MISSING)` — no hardcoded default.
    """
    explicit = os.environ.get("AUTO_FT_OSTRIS_RUN_PY")
    if explicit:
        return Path(explicit)
    cfg = load_ostris_config() if config is None else config
    run_py = cfg.get("run_py")
    if run_py:
        return Path(run_py)
    raise ConfigError(
        ErrorCode.E_OSTRIS_CONFIG_MISSING,
        f"Ostris `run.py` not configured. Set AUTO_FT_OSTRIS_RUN_PY or "
        f"[ostris].run_py in {CONFIG_PATH}.",
        {"config_path": str(CONFIG_PATH)},
    )


def _resolve_ostris_python(run_py: Path, config: dict | None = None) -> Path:
    """Resolve the Ostris Python interpreter.

    Resolution order (highest priority first):
      1. `AUTO_FT_OSTRIS_PYTHON` env var.
      2. `[ostris].python` in config — standalone interpreter (absolute path).
      3. `[ostris].venv` in config — non-standard venv root; appends
         `Scripts/python.exe` on Windows or `bin/python` elsewhere.
      4. Auto-derive from `run_py.parent`: assume a standard `venv/` sibling
         and append the platform-appropriate interpreter subpath.
      5. Raise `ConfigError(E_OSTRIS_CONFIG_MISSING)`.

    An entry at priority 2/3/4 is accepted only if the resulting path exists
    on disk — this lets the resolver fall through from an empty config to
    auto-derive without surprising the user.
    """
    explicit = os.environ.get("AUTO_FT_OSTRIS_PYTHON")
    if explicit:
        return Path(explicit)
    cfg = load_ostris_config() if config is None else config

    standalone = cfg.get("python")
    if standalone:
        return Path(standalone)

    venv = cfg.get("venv")
    if venv:
        return _venv_python_from_root(Path(venv))

    auto = _venv_python_from_root(run_py.parent / "venv")
    if auto.exists():
        return auto

    raise ConfigError(
        ErrorCode.E_OSTRIS_CONFIG_MISSING,
        f"Ostris Python interpreter not resolvable. Set AUTO_FT_OSTRIS_PYTHON, "
        f"[ostris].python (standalone), [ostris].venv (venv root), or place a "
        f"venv at {run_py.parent / 'venv'}.",
        {"config_path": str(CONFIG_PATH), "run_py_parent": str(run_py.parent)},
    )


def gpu_lock_path() -> Path:
    """GPU-wide lock location (per-machine, not per-job)."""
    return Path.home() / ".auto_ft" / "gpu.lock"


def _load_and_validate_cfg(cfg_path: Path) -> dict:
    """Load cfg.yaml via `yaml.safe_load` and validate launcher-required keys.

    Raises:
      ConfigError(E_RUN_NOT_FOUND) when the file is absent.
      ConfigError(E_RECIPE_INVALID) when the YAML is malformed, the root is
        not a mapping, required keys are missing, or `training_folder` is
        not absolute.
    """
    if not cfg_path.exists():
        raise ConfigError(
            ErrorCode.E_RUN_NOT_FOUND,
            f"Config not found: {cfg_path}",
            {"path": str(cfg_path)},
        )
    try:
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as e:
        raise ConfigError(
            ErrorCode.E_RECIPE_INVALID,
            f"Malformed YAML: {e}",
            {"path": str(cfg_path)},
        ) from e
    if not isinstance(cfg, dict):
        raise ConfigError(
            ErrorCode.E_RECIPE_INVALID,
            "YAML root must be a mapping",
            {"path": str(cfg_path)},
        )
    try:
        name = cfg["config"]["name"]
        training_folder = cfg["config"]["process"][0]["training_folder"]
    except (KeyError, IndexError, TypeError) as e:
        raise ConfigError(
            ErrorCode.E_RECIPE_INVALID,
            f"Missing required key: {e}",
            {"path": str(cfg_path), "missing": str(e)},
        ) from e
    # `training_folder` MUST be absolute (avoid cwd surprises).
    if not Path(training_folder).is_absolute():
        raise ConfigError(
            ErrorCode.E_RECIPE_INVALID,
            f"training_folder must be absolute; got {training_folder!r}",
            {"path": str(cfg_path), "training_folder": training_folder},
        )
    return {"name": name, "training_folder": Path(training_folder).resolve()}


def _build_subprocess_env() -> dict[str, str]:
    """Copy `os.environ` but strip `VIRTUAL_ENV`.

    `auto_ft` is typically installed via pipx or its own venv. Inheriting
    `VIRTUAL_ENV` would mislead any child shells spawned by Ostris. The
    Ostris venv Python binds its interpreter to its on-disk path
    independently of `VIRTUAL_ENV`, so stripping is safe.
    """
    return {k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV"}


def launch_ostris(config_path: str | Path) -> LaunchResult:
    """Spawn Ostris training detached.

    Steps:
      1. Load + validate cfg.yaml (raises E_RUN_NOT_FOUND / E_RECIPE_INVALID).
      2. Resolve `save_root = training_folder / config.name`; mkdir exist_ok.
      3. Check the GPU-wide lock (`~/.auto_ft/gpu.lock`):
         - Non-stale → raise E_GPU_BUSY.
         - Stale or absent → proceed (auto-clean).
      4. `spawn_detached` the Ostris subprocess (stdout+stderr → `train.log`).
      5. Acquire both the GPU lock and the per-job `.lock` with the spawned PID.
      6. Write `save_root/train.pid`.
      7. Return a `LaunchResult`.
    """
    cfg_path = Path(config_path).resolve()
    cfg_info = _load_and_validate_cfg(cfg_path)
    name: str = cfg_info["name"]
    save_root: Path = cfg_info["training_folder"] / name
    save_root.mkdir(parents=True, exist_ok=True)

    log_path = save_root / "train.log"
    pid_path = save_root / "train.pid"
    job_lock = save_root / ".lock"
    gpu_lock = gpu_lock_path()

    # Gate on a non-stale GPU lock; stale auto-cleans.
    existing = check_stale(gpu_lock)
    if existing is not None and not existing.stale:
        raise OpsError(
            ErrorCode.E_GPU_BUSY,
            f"GPU busy: pid={existing.pid} hostname={existing.hostname}",
            {"holder_pid": existing.pid, "hostname": existing.hostname},
        )

    cfg = load_ostris_config()
    run_py = _resolve_ostris_run_py(cfg)
    ostris_python = _resolve_ostris_python(run_py, cfg)
    env = _build_subprocess_env()

    pid = spawn_detached(
        cmd=[str(ostris_python), str(run_py), str(cfg_path)],
        # cwd = project parent of save_root's parent; Ostris owns save_root layout.
        cwd=str(save_root.parent.parent),
        env=env,
        log_path=str(log_path),
    )
    # Acquire both locks with the spawned PID so `is_alive(pid)` drives staleness.
    acquire(gpu_lock, pid)
    acquire(job_lock, pid)
    pid_path.write_text(str(pid), encoding="utf-8")

    return LaunchResult(
        job_name=name,
        pid=pid,
        state="running",
        log_path=str(log_path),
    )
