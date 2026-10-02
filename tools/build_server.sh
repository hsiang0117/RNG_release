#!/usr/bin/env bash
set -euo pipefail
_rng_build_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$_rng_build_root"
source tools/env_server.sh
mkdir -p temporary-build dependency-cache/wheels
for extension in simple-knn diff-gaussian-rasterization diff-gaussian-rasterization-orthographic; do
  echo "Building $extension for sm_89"
  python -m pip wheel --no-build-isolation --no-deps "submodules/$extension" -w dependency-cache/wheels > "temporary-build/$extension-build.log" 2>&1
done
python -m pip install --no-index --no-deps --force-reinstall --find-links dependency-cache/wheels simple-knn diff-gaussian-rasterization diff-gaussian-rasterization-orthographic
python -m pip check
python -m pip freeze > temporary-build/requirements-installed.txt
