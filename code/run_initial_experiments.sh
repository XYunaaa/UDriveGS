#!/usr/bin/env bash
# Native 30k baselines on the two independently reconstructed segments.
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
env_prefix="$project_dir/.envs/splatfacto"
result_dir="$project_dir/results/splatfacto/scene1"
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
exec > >(tee -a "$result_dir/initial-experiments.log") 2>&1
"$env_prefix/bin/python" - <<'PY'
import torch
assert torch.cuda.is_available(), 'GPU is unavailable in this execution session; native training cannot start.'
print('GPU:', torch.cuda.get_device_name(0))
PY
for segment in a; do
    dataset_dir="$project_dir/data/scene1_splatfacto/processed_segment_$segment"
    run_dir="$result_dir/runs/full_unmasked/scene1_segment_$segment/splatfacto/initial_30k_v1"
    [[ ! -e "$run_dir" ]] || { echo "Refusing to overwrite $run_dir" >&2; exit 1; }
    echo "Starting segment $segment: $dataset_dir"
    "$env_prefix/bin/ns-train" splatfacto \
        --data "$dataset_dir" --output-dir "$result_dir/runs/full_unmasked" \
        --experiment-name "scene1_segment_$segment" --timestamp initial_30k_v1 \
        --max-num-iterations 30000 --machine.seed 42 --vis tensorboard \
        --steps-per-eval-image 500 --steps-per-eval-all-images 30000 \
        nerfstudio-data --downscale-factor 1
    "$env_prefix/bin/ns-eval" --load-config "$run_dir/config.yml" \
        --output-path "$result_dir/segment_${segment}_test_metrics.json" \
        --render-output-path "$result_dir/segment_${segment}_test_images"
    "$env_prefix/bin/ns-export" gaussian-splat --load-config "$run_dir/config.yml" \
        --output-dir "$result_dir/exports/segment_${segment}_30k"
    "$env_prefix/bin/ns-render" interpolate --load-config "$run_dir/config.yml" \
        --pose-source train --interpolation-steps 2 --frame-rate 24 \
        --output-path "$result_dir/segment_${segment}_camera_path.mp4"
done
