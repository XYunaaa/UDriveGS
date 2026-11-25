#!/usr/bin/env bash
# Scene2 train-only SfM, including matches between independent capture videos.
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
dataset_dir="$project_dir/data/scene2_splatfacto"
run_dir="$dataset_dir/colmap/initial"
colmap_exe="$project_dir/.envs/splatfacto/bin/colmap"
export PYTHONNOUSERSITE=1
unset PYTHONPATH
export QT_QPA_PLATFORM=offscreen
[[ ! -e "$run_dir" ]] || { echo "Refusing to overwrite $run_dir" >&2; exit 1; }
"$project_dir/.envs/splatfacto/bin/python" - "$dataset_dir" <<'PY'
import json
from pathlib import Path
import sys
root = Path(sys.argv[1])
splits = json.loads((root / 'splits.json').read_text())
train = set(splits['train'])
assert train and splits['val'] and splits['test']
assert train.isdisjoint(splits['val']) and train.isdisjoint(splits['test'])
assert all((root / 'images' / n).is_file() for n in train)
for pair in (root / 'pairs.txt').read_text().splitlines():
    a, b = pair.split()
    assert a in train and b in train
print('Train-only input and cross-video pair membership checked.', flush=True)
PY
mkdir -p "$run_dir/sparse"
exec > >(tee "$run_dir/run.log") 2>&1
"$colmap_exe" feature_extractor \
    --database_path "$run_dir/database.db" --image_path "$dataset_dir/images" \
    --image_list_path "$dataset_dir/train.txt" \
    --ImageReader.single_camera 1 --ImageReader.camera_model SIMPLE_RADIAL \
    --SiftExtraction.use_gpu 0 --SiftExtraction.num_threads 8 \
    --SiftExtraction.max_image_size 1600 --SiftExtraction.max_num_features 4096
"$colmap_exe" matches_importer \
    --database_path "$run_dir/database.db" --match_list_path "$dataset_dir/pairs.txt" \
    --match_type pairs --SiftMatching.use_gpu 0 --SiftMatching.num_threads 8
"$colmap_exe" mapper \
    --database_path "$run_dir/database.db" --image_path "$dataset_dir/images" \
    --output_path "$run_dir/sparse" --Mapper.num_threads 1
found_model=false
for model_dir in "$run_dir"/sparse/*; do
    [[ -f "$model_dir/images.bin" ]] || continue
    found_model=true
    "$colmap_exe" model_analyzer --path "$model_dir"
done
[[ "$found_model" == true ]] || { echo 'No sparse model was generated.' >&2; exit 1; }
