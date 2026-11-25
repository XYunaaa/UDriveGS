#!/usr/bin/env bash
# Scene2 full-resolution, full-video-held-out Gaussian reconstruction.
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
env_prefix="$project_dir/.envs/splatfacto"
dataset_dir="${1:-$project_dir/data/scene2_splatfacto/processed}"
result_dir="${2:-$project_dir/results/splatfacto/scene2}"
experiment_name="${3:-scene2_big}"
run_dir="$result_dir/runs/$experiment_name/splatfacto/quality_30k_v1"
export PYTHONNOUSERSITE=1
export PYTHONFAULTHANDLER=1
export PYTHONUNBUFFERED=1
unset PYTHONPATH
export PATH="$env_prefix/bin:$PATH"
export CUDA_HOME="$env_prefix"
export CPATH="$env_prefix/targets/x86_64-linux/include${CPATH:+:$CPATH}"
export CC=/usr/bin/gcc-12
export CXX=/usr/bin/g++-12
export TORCH_CUDA_ARCH_LIST=8.9
export TORCH_EXTENSIONS_DIR="$project_dir/.cache/torch_extensions_gcc12"
export MPLCONFIGDIR="$project_dir/.cache/matplotlib"
export MAX_JOBS=4
mkdir -p "$result_dir"
exec > >(tee -a "$result_dir/training.log") 2>&1
"$env_prefix/bin/python" - "$dataset_dir" <<'PY'
import json
from pathlib import Path
import sys
import torch
root = Path(sys.argv[1])
assert torch.cuda.is_available(), 'This execution session has no GPU access; run in the host GPU terminal.'
free, total = torch.cuda.mem_get_info()
assert free >= 12 * 1024**3, f'Need12GiB free GPU memory for this initial1080p big run; currently{free / 1024**3:.1f}GiB free'
provenance = json.loads((root / 'provenance.json').read_text())
assert provenance['training_geometry_unchanged'] is True
meta = json.loads((root / 'transforms.json').read_text())
names = {s: set(meta[f'{s}_filenames']) for s in ['train', 'val', 'test']}
assert all(names.values())
assert names['train'].isdisjoint(names['val'] | names['test']) and names['val'].isdisjoint(names['test'])
assert len(names['train']) >= .9 * provenance['original_counts']['train']
assert all((root / name).is_file() for values in names.values() for name in values)
assert (root / meta['ply_file_path']).is_file()
print('GPU:', torch.cuda.get_device_name(0), '| free GiB:', free / 1024**3)
print('Frozen video splits:', {s: len(v) for s, v in names.items()})
PY
[[ ! -e "$run_dir" ]] || { echo "Refusing to overwrite $run_dir" >&2; exit 1; }
"$env_prefix/bin/ns-train" splatfacto-big \
    --data "$dataset_dir" --output-dir "$result_dir/runs" \
    --experiment-name "$experiment_name" --timestamp quality_30k_v1 \
    --max-num-iterations 30000 --machine.seed 42 --vis tensorboard \
    --steps-per-eval-image 1000 --steps-per-eval-all-images 5000 \
    nerfstudio-data --downscale-factor 1
"$env_prefix/bin/ns-eval" --load-config "$run_dir/config.yml" \
    --output-path "$result_dir/test_metrics.json" --render-output-path "$result_dir/test_images"
"$env_prefix/bin/ns-export" gaussian-splat --load-config "$run_dir/config.yml" \
    --output-dir "$result_dir/exports/scene2_big_30k"
"$env_prefix/bin/ns-render" interpolate --load-config "$run_dir/config.yml" \
    --pose-source train --interpolation-steps 2 --frame-rate 24 \
    --output-path "$result_dir/capture_paths.mp4"
echo "Finished: $result_dir/exports/scene2_big_30k/splat.ply"
