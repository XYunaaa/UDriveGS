#!/usr/bin/env bash
# One sequential GPU queue; original scene2 artifacts remain read-only inputs.
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
env_prefix="$project_dir/.envs/splatfacto"
result_dir="${SCENE2_RESULT_DIR:-$project_dir/results/splatfacto/scene2_quality}"
action="${1:-rgb}"
if [[ $# -gt 0 ]]; then shift; fi
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MAX_JOBS=4
export TORCHINDUCTOR_COMPILE_THREADS=1
export PATH="$env_prefix/bin:$PATH"
export CUDA_HOME="$env_prefix"
export CPATH="$env_prefix/targets/x86_64-linux/include${CPATH:+:$CPATH}"
export CC=/usr/bin/gcc-12 CXX=/usr/bin/g++-12
export TORCH_EXTENSIONS_DIR="$project_dir/.cache/torch_extensions_gcc12"
export MPLCONFIGDIR="$project_dir/.cache/matplotlib"
# Only this project's local depth subclass may extend the Python module path.
export PYTHONPATH="$project_dir/code"
mkdir -p "$result_dir"
exec > >(tee -a "$result_dir/host-queue.log") 2>&1
cd "$project_dir"
"$env_prefix/bin/python" "$project_dir/code/scene2_quality.py" "$action" --root "$result_dir" "$@"
