#!/usr/bin/env bash
# Overlay packages reuse the existing CUDA/PyTorch install without changing it.
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
base_env="$project_dir/.envs/splatfacto"
env_prefix="$project_dir/.envs/sugar"
vendor="$project_dir/code/third_party"
result_dir="${1:-$project_dir/results/mesh/scene1_segment_a_sugar}"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
unset PYTHONPATH
export CUDA_HOME="$base_env"
export PATH="$env_prefix/bin:$base_env/bin:$PATH"
export CPATH="$base_env/targets/x86_64-linux/include${CPATH:+:$CPATH}"
export CC=/usr/bin/gcc-12 CXX=/usr/bin/g++-12
export TORCH_CUDA_ARCH_LIST=8.9 FORCE_CUDA=1 MAX_JOBS=4
export TORCH_EXTENSIONS_DIR="$project_dir/.cache/sugar_extensions_gcc12"
export MPLCONFIGDIR="$project_dir/.cache/matplotlib"
export PIP_CACHE_DIR=/tmp/udrive-sugar-pip-cache
mkdir -p "$result_dir"
exec > >(tee -a "$result_dir/setup.log") 2>&1
[[ -x "$base_env/bin/nvcc" && -x "$CXX" ]] || { echo 'CUDA toolkit or GCC12 is missing.' >&2; exit 1; }
[[ -f "$vendor/SuGaR/extract_mesh.py" ]] || { echo 'The prepared SuGaR repository is missing.' >&2; exit 1; }
[[ "$(git -C "$vendor/SuGaR" rev-parse HEAD)" == 7c10c4ae4a267dece512f5c7f40ed212a0a2ab44 ]] || { echo 'Unexpected SuGaR revision.' >&2; exit 1; }
if [[ ! -d "$vendor/pytorch3d" ]]; then
    git clone --depth 1 --branch V0.7.8 https://github.com/facebookresearch/pytorch3d.git "$vendor/pytorch3d"
fi
[[ "$(git -C "$vendor/pytorch3d" rev-parse HEAD)" == 75ebeeaea0908c5527e7b1e305fbc7681382db47 ]] || { echo 'Unexpected PyTorch3D revision.' >&2; exit 1; }
if [[ ! -x "$env_prefix/bin/python" ]]; then
    "$base_env/bin/python" -m venv --system-site-packages "$env_prefix"
fi
"$env_prefix/bin/python" - <<'PY'
import shutil
import torch
assert torch.__version__ == '2.4.0+cu124', torch.__version__
assert shutil.disk_usage('.').free > 8 * 1024**3, 'Need at least8GiB free for extension builds.'
print('Build PyTorch:', torch.__version__, '| GPU execution is checked by the extraction launcher.')
PY
"$env_prefix/bin/python" -m pip install --index-url https://pypi.org/simple --retries 0 --timeout 20 --no-build-isolation iopath==0.1.10 fvcore==0.1.5.post20221221
for package_dir in "$vendor/pytorch3d" \
    "$vendor/SuGaR/gaussian_splatting/submodules/diff-gaussian-rasterization" \
    "$vendor/SuGaR/gaussian_splatting/submodules/simple-knn"; do
    "$env_prefix/bin/python" -m pip install --no-build-isolation --no-deps "$package_dir"
done
cd "$vendor/SuGaR"
"$env_prefix/bin/python" - <<'PY'
import torch
import pytorch3d._C
import simple_knn._C
import diff_gaussian_rasterization._C
from sugar_extractors.coarse_mesh import extract_mesh_from_coarse_sugar
print('SuGaR and all CUDA extensions imported:', torch.__version__)
PY
"$env_prefix/bin/python" -m pip freeze > "$result_dir/environment-pip.txt"
echo "SuGaR environment prepared: $env_prefix"
