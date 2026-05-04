# auto_ft

A Python CLI that wraps [ostris/ai-toolkit](https://github.com/ostris/ai-toolkit)
to launch and manage LoRA fine-tuning jobs from a plain shell.

`auto_ft` does not train anything itself — it composes none of the YAML, runs
none of the GPU code. It is a thin orchestration layer: spawn a detached
Ostris subprocess, watch the on-disk artifacts, and surface JSON-shaped
status the way other shell tools and scripts can consume.

## What you get

A nine-subcommand CLI over an existing Ostris install:

| Command | Purpose |
|---------|---------|
| `auto_ft init` | Create `.autoft/state.json` and register a job. |
| `auto_ft prepare` | Validate a dataset (count, format, resolution, PIL integrity) and write a hash manifest. |
| `auto_ft train` | Spawn detached Ostris training; returns in ~2s. |
| `auto_ft status` | Filesystem-derived job status (running / completed / stopped / stale / failed). |
| `auto_ft logs` | Tail N lines of `train.log` (platform-independent). |
| `auto_ft stop` | Terminate a running job (Windows: `TerminateProcess`). |
| `auto_ft checkpoints` | List `*.safetensors` checkpoints. |
| `auto_ft samples` | List sample image paths. |
| `auto_ft export` | Copy a checkpoint to a deployable path. |

Every successful invocation emits a single JSON object on stdout. Errors
emit `{"error_code": "...", "message": "...", "details": {...}}` and exit 1.

## Install

```powershell
pipx install auto-ft
```

`auto_ft` requires Python 3.11+ and a working
[ostris/ai-toolkit](https://github.com/ostris/ai-toolkit) install on the
same machine. The CLI launches Ostris as a subprocess; it does not vendor
or install Ostris itself.

## Configure Ostris

Tell `auto_ft` how to find your Ostris install. Two options:

**Environment variables:**

```powershell
$env:AUTO_FT_OSTRIS_PYTHON = "C:\path\to\ai-toolkit\venv\Scripts\python.exe"
$env:AUTO_FT_OSTRIS_RUN_PY = "C:\path\to\ai-toolkit\run.py"
```

**Config file** at `~/.auto_ft/config.toml`:

```toml
[ostris]
python = "C:\\path\\to\\ai-toolkit\\venv\\Scripts\\python.exe"
run_py = "C:\\path\\to\\ai-toolkit\\run.py"
```

Env vars take precedence. If both are absent, `auto_ft` raises
`E_OSTRIS_CONFIG_MISSING`.

## Quickstart

```powershell
# 1. Register a job in the current project.
auto_ft init --name mydog --trigger zyx --dataset .\images

# 2. Validate the dataset (writes <images>\.auto_ft_prep.json).
auto_ft prepare .\images --trigger zyx

# 3. Hand auto_ft an Ostris-shaped YAML and launch.
auto_ft train .\config.yaml

# 4. Check progress (read-only; no side effects).
auto_ft status mydog
auto_ft logs mydog --tail 50

# 5. When the job is done, export the LoRA.
auto_ft checkpoints mydog
auto_ft export .\output\mydog\mydog.safetensors
```

`auto_ft` does not generate the Ostris YAML for you — bring your own,
hand-authored or composed by any tool you prefer. The CLI's only YAML
requirement is that `config.process[0].training_folder` is an absolute
path; Ostris owns the `<training_folder>/<config.name>/` join itself.

## Trust boundary

`auto_ft` is a thin wrapper. When you run `auto_ft train cfg.yaml`, you are
trusting:

- the Python CLI you installed (this package), and
- the Ostris installation pointed at by `AUTO_FT_OSTRIS_PYTHON` /
  `AUTO_FT_OSTRIS_RUN_PY`.

The CLI never imports `ai-toolkit`; it only spawns the Ostris Python
interpreter as a subprocess. Resume semantics, model loading, GPU
ownership — everything compute-bearing — happens inside Ostris.

## Status

Alpha (`0.1.x`). Tested on Windows 10/11 with NVIDIA 8GB VRAM running SDXL
LoRA training. The CLI itself is arch-agnostic — any model Ostris can
train, `auto_ft` can launch — but only the SDXL path has been exercised
end-to-end so far.

## License

MIT — see `LICENSE`.
