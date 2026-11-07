#!/usr/bin/env bash
# Source from the repository checkout before running the launcher.
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa JAX_PLATFORMS=tpu PYTHONUNBUFFERED=1
runtime="$HOME/.cache/ogpo-tpu"
mkdir -p "$runtime"
(
    flock -x 9
    if [[ ! -f "$runtime/system-ready" ]]; then
        sudo apt-get -o DPkg::Lock::Timeout=3600 update -qq
        sudo env DEBIAN_FRONTEND=noninteractive apt-get -o DPkg::Lock::Timeout=3600 install -y -qq \
            libosmesa6 libgl1 libglib2.0-0 libegl1-mesa-dev cmake build-essential bc
        touch "$runtime/system-ready"
    fi
    if [[ ! -x "$runtime/uv" ]]; then
        curl -fsSL https://github.com/astral-sh/uv/releases/download/0.11.11/uv-x86_64-unknown-linux-gnu.tar.gz \
            | tar -xz -C "$runtime" --strip-components=1 uv-x86_64-unknown-linux-gnu/uv
    fi
) 9>"$runtime/system.lock"
export PATH="$runtime:$PATH"
# Each checkout has its own environment, avoiding editable-install cross-talk.
uv sync --frozen --extra robomimic --no-dev --python 3.12
uv pip install --python .venv/bin/python 'jax[tpu]==0.10.1' 'google-cloud-storage==3.7.0'
export UV_NO_SYNC=1
export PATH="$PWD/.venv/bin:$PATH" PYTHONPATH="$PWD"
python - <<'PY'
import jax
assert jax.default_backend() == 'tpu', jax.devices()
print('OGPO TPU runtime:', jax.__version__, jax.devices(), flush=True)
PY
