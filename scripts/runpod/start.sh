#!/usr/bin/env bash
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$root"
export HF_HOME=/workspace/hf
export HUGGINGFACE_HUB_CACHE=/workspace/hf/hub
export HF_HUB_CACHE=/workspace/hf/hub
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TMPDIR=/tmp

setup=0
if [[ "${1:-}" == --setup ]]; then setup=1; shift; fi
venv="${AEVON_RUNPOD_VENV:-${VIRTUAL_ENV:-}}"
if [[ -z "$venv" ]]; then
    if [[ -x "$root/.venv/bin/python" ]]; then venv="$root/.venv"
    else venv=/workspace/venv; fi
fi
if [[ ! -e "$venv" ]]; then
    # Retain the pod image's CUDA PyTorch/TorchVision installation.
    python3 -m venv --system-site-packages "$venv"
    setup=1
fi
python="$venv/bin/python"
if [[ ! -x "$python" ]]; then
    echo "Venv is not usable: $venv. Set AEVON_RUNPOD_VENV to the existing Linux venv." >&2
    exit 1
fi
if [[ "$setup" == 1 ]]; then
    "$python" -m pip install -e '.[runtime]'
fi
if ! "$python" -c 'import importlib.util, sys; sys.exit(any(importlib.util.find_spec(n) is None for n in ("httpx", "uvicorn", "torch", "torchvision", "transformers", "accelerate")))'; then
    echo "Runtime dependencies are missing. Use the validated CUDA pod image and run this script with --setup." >&2
    exit 1
fi
exec "$python" -m scripts.startup runpod "$@"
