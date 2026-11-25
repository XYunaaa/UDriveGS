#!/usr/bin/env bash
# Run from the host terminal, where network and NVIDIA devices are accessible.
set -euo pipefail
# Global user packages and ROS/CARLA paths must not satisfy project dependencies.
export PYTHONNOUSERSITE=1
unset PYTHONPATH
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
env_prefix="$project_dir/.envs/splatfacto"
log_dir="$project_dir/results/splatfacto/scene1"
mkdir -p "$log_dir"
exec > >(tee -a "$log_dir/setup.log") 2>&1

nvidia-smi
conda_exe="${CONDA_EXE:-$(command -v conda || true)}"
if [[ ! -x "$conda_exe" ]]; then
    echo 'Conda executable is unavailable.' >&2
    exit 1
fi
if [[ ! -d "$env_prefix/conda-meta" ]]; then
    "$conda_exe" create --prefix "$env_prefix" -c conda-forge -c nvidia \
        python=3.10 pip 'colmap=3.9.1' 'cuda-toolkit=12.4' -y
fi
export PATH="$env_prefix/bin:$PATH"
export CUDA_HOME="$env_prefix"
export CPATH="$env_prefix/targets/x86_64-linux/include${CPATH:+:$CPATH}"
# CUDA 12.4 cannot compile with the Conda environment's GCC 15.
export CC=/usr/bin/gcc-12
export CXX=/usr/bin/g++-12
# PyTorch detects the visible GPU architecture unless the user supplies an override.
export TORCH_EXTENSIONS_DIR="$project_dir/.cache/torch_extensions_gcc12"
export MAX_JOBS=4
nvcc --version
"$env_prefix/bin/python" -m pip install 'setuptools<81' wheel
"$env_prefix/bin/python" -m pip install -c "$project_dir/code/splatfacto-constraints.txt" \
    torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu124
"$env_prefix/bin/python" -m pip install -c "$project_dir/code/splatfacto-constraints.txt" \
    nerfstudio==1.1.5
"$env_prefix/bin/python" -m pip check
"$env_prefix/bin/python" -m pip freeze > "$log_dir/environment-pip.txt"
"$conda_exe" list --prefix "$env_prefix" --explicit > "$log_dir/environment-conda.txt"
"$env_prefix/bin/python" - <<'PY'
import torch
from gsplat import rasterization
assert torch.cuda.is_available(), 'CUDA is not accessible in this terminal'
x = torch.ones(2, device='cuda', requires_grad=True)
(x * x).sum().backward()
torch.cuda.synchronize()
assert torch.equal(x.grad, torch.full_like(x, 2))
print('PyTorch CUDA forward/backward passed:', torch.cuda.get_device_name(0), torch.__version__)
means = torch.tensor([[0., 0., 3.]], device='cuda', requires_grad=True)
colors = torch.tensor([[0.8, 0.2, 0.1]], device='cuda', requires_grad=True)
rgb, alpha, _ = rasterization(
    means=means, quats=torch.tensor([[1., 0., 0., 0.]], device='cuda'),
    scales=torch.full((1, 3), 0.1, device='cuda'),
    opacities=torch.tensor([0.8], device='cuda'), colors=colors,
    viewmats=torch.eye(4, device='cuda')[None],
    Ks=torch.tensor([[[40., 0., 16.], [0., 40., 16.], [0., 0., 1.]]], device='cuda'),
    width=32, height=32, packed=False,
)
(rgb.sum() + alpha.sum()).backward()
torch.cuda.synchronize()
assert torch.isfinite(rgb).all() and rgb.sum() > 0
assert means.grad is not None and torch.isfinite(means.grad).all()
assert colors.grad is not None and torch.isfinite(colors.grad).all() and colors.grad.abs().sum() > 0
print('gsplat CUDA rasterization forward/backward passed')
PY
"$env_prefix/bin/colmap" -h
echo "Setup complete. Environment: $env_prefix"
