#!/usr/bin/env bash
# Source this file from any directory.
export CUDA_HOME=/usr/local/cuda-12.8
_rng_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export PATH="$_rng_root/.venv/bin:$CUDA_HOME/bin:$PATH"
export TORCH_CUDA_ARCH_LIST=8.9
export MAX_JOBS=4
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
unset _rng_root
