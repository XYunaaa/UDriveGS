#!/usr/bin/env bash
# Initial full-sequence baseline without dynamic masks; frozen held-out images excluded.
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
dataset_dir="$project_dir/data/scene1_splatfacto"
run_dir="$dataset_dir/colmap/full_unmasked"
colmap_exe="$project_dir/.envs/splatfacto/bin/colmap"
[[ ! -e "$run_dir" ]] || { echo "Refusing to overwrite $run_dir" >&2; exit 1; }
mkdir -p "$run_dir/sparse"
exec > >(tee "$run_dir/run.log") 2>&1
export QT_QPA_PLATFORM=offscreen
"$colmap_exe" feature_extractor \
    --database_path "$run_dir/database.db" --image_path "$dataset_dir/images" \
    --image_list_path "$dataset_dir/train.txt" \
    --ImageReader.single_camera 1 --ImageReader.camera_model SIMPLE_RADIAL \
    --SiftExtraction.use_gpu 0 --SiftExtraction.num_threads 8 --SiftExtraction.max_image_size 960
"$colmap_exe" sequential_matcher \
    --database_path "$run_dir/database.db" \
    --SequentialMatching.overlap 10 --SiftMatching.use_gpu 0 --SiftMatching.num_threads 8
"$colmap_exe" mapper \
    --database_path "$run_dir/database.db" --image_path "$dataset_dir/images" \
    --output_path "$run_dir/sparse" --Mapper.num_threads 1 --Mapper.init_min_tri_angle 4
found_model=false
for model_dir in "$run_dir"/sparse/*; do
    [[ -f "$model_dir/images.bin" ]] || continue
    found_model=true
    "$colmap_exe" model_analyzer --path "$model_dir"
done
[[ "$found_model" == true ]] || { echo 'No sparse model was generated.' >&2; exit 1; }
