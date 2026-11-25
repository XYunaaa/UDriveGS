#!/usr/bin/env bash
# Host GPU terminal only. This is a training-only smoke run, not held-out evaluation.
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
env_prefix="$project_dir/.envs/splatfacto"
dataset_dir="$project_dir/data/scene1_splatfacto/processed_smoke"
run_dir="$project_dir/results/splatfacto/scene1/runs/smoke"
export PYTHONNOUSERSITE=1
unset PYTHONPATH
export PATH="$env_prefix/bin:$PATH"
export CUDA_HOME="$env_prefix"
export CPATH="$env_prefix/targets/x86_64-linux/include${CPATH:+:$CPATH}"
export CC=/usr/bin/gcc-12
export CXX=/usr/bin/g++-12
# Use the visible GPU architecture or a user-provided TORCH_CUDA_ARCH_LIST.
export MAX_JOBS=4
export MPLCONFIGDIR="$project_dir/.cache/matplotlib"
export TORCH_EXTENSIONS_DIR="$project_dir/.cache/torch_extensions_gcc12"

[[ ! -e "$run_dir" ]] || { echo "Refusing to overwrite $run_dir" >&2; exit 1; }
"$env_prefix/bin/python" -m pip check
"$env_prefix/bin/python" - "$dataset_dir" <<'PY'
import json
from pathlib import Path
import sys
import torch
root = Path(sys.argv[1])
assert torch.cuda.is_available(), 'Run this script in the host GPU terminal'
provenance = json.loads((root / 'provenance.json').read_text())
meta = json.loads((root / 'transforms.json').read_text())
assert provenance['all_frames_in_original_train_split'] is True
assert len(meta['frames']) == provenance['frame_count'] == 52
actual = {Path(frame['file_path']).name for frame in meta['frames']}
expected = set(json.loads((root.parent / 'smoke_train.json').read_text()))
original_train = set(json.loads((root.parent / 'splits.json').read_text())['train'])
assert actual == expected and actual <= original_train
assert all((root / frame['file_path']).is_file() for frame in meta['frames'])
assert (root / meta['ply_file_path']).is_file()
print('Smoke inputs checked: 52 original training frames, sparse point initialization available.')
PY
mkdir -p "$(dirname -- "$run_dir")"
mkdir "$run_dir"
exec > >(tee "$run_dir/train.log") 2>&1
"$env_prefix/bin/ns-train" splatfacto \
    --data "$dataset_dir" --output-dir "$run_dir" --experiment-name scene1_smoke \
    --max-num-iterations 2000 --machine.seed 42 --vis tensorboard \
    --steps-per-eval-image 0 --steps-per-eval-all-images 0 \
    nerfstudio-data --downscale-factor 1 --eval-mode all
echo "Smoke training finished. Checkpoint and config are under $run_dir"
