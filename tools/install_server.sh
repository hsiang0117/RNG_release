#!/usr/bin/env bash
set -euo pipefail
_rng_install_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$_rng_install_root"
if [[ ! -x .venv/bin/python ]]; then
  "${RNG_PYTHON:-/opt/conda/bin/python3.12}" -m venv .venv
fi
source tools/env_server.sh
mkdir -p dependency-cache/wheels temporary-build
wheel_args=(--find-links dependency-cache/wheels)
for cache in /myfiles/projects/Vol3DGS/dependency-cache/wheels /myfiles/projects/ever_training/dependency-cache/wheels; do
  if [[ -d "$cache" ]]; then wheel_args+=(--find-links "$cache"); fi
done
python -m pip install --upgrade pip
python -m pip install "${wheel_args[@]}" --cache-dir dependency-cache/pip -r requirements-server-lock.txt
bash tools/build_server.sh
PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" python tools/verify_environment.py
PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" python tools/verify_orthographic_gradients.py
python train.py --help > temporary-build/train-help.txt
python render.py --help > temporary-build/render-help.txt
