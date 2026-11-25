#!/usr/bin/env bash
# New4FPS/clarity experiment: GPU SIFT1920/8192; COLMAP3.9.1 Mapper remainsCPU.
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
env_prefix="$project_dir/.envs/splatfacto"
dataset_dir="$project_dir/data/scene2_4fps_splatfacto"
result_dir="$project_dir/results/splatfacto/scene2_4fps"
run_dir="$dataset_dir/colmap/initial"
export PYTHONNOUSERSITE=1
export PYTHONFAULTHANDLER=1
export PYTHONUNBUFFERED=1
export QT_QPA_PLATFORM=offscreen
export MPLCONFIGDIR="$project_dir/.cache/matplotlib"
export PATH="$env_prefix/bin:$PATH"
export CUDA_HOME="$env_prefix"
export CPATH="$env_prefix/targets/x86_64-linux/include${CPATH:+:$CPATH}"
export CC=/usr/bin/gcc-12
export CXX=/usr/bin/g++-12
export TORCH_CUDA_ARCH_LIST=8.9
export TORCH_EXTENSIONS_DIR="$project_dir/.cache/torch_extensions_gcc12"
export MAX_JOBS=4
unset PYTHONPATH
mkdir -p "$result_dir"
exec > >(tee -a "$result_dir/host-pipeline.log") 2>&1
"$env_prefix/bin/python" - "$dataset_dir" <<'PY'
import json
from pathlib import Path
import sys
import torch
root = Path(sys.argv[1])
assert torch.cuda.is_available(), 'Run this script in the host GPU terminal.'
manifest = json.loads((root / 'manifest.json').read_text())
assert manifest['colmap_parameters'] == {'max_image_size': 1920, 'max_num_features': 8192}
assert manifest['sharpness_filter']['enabled'] is True
splits = json.loads((root / 'splits.json').read_text())
baseline = Path(manifest['baseline'])
old = json.loads((baseline / 'splits.json').read_text())
assert splits['val'] == old['val'] and splits['test'] == old['test']
train = set(splits['train'])
assert train and train.isdisjoint(splits['val'] + splits['test'])
assert all((root / 'images' / name).is_file() for name in train)
for pair in (root / 'pairs.txt').read_text().splitlines():
    assert set(pair.split()) <= train
print('GPU:', torch.cuda.get_device_name(0), '| new counts:', {s: len(v) for s, v in splits.items()})
PY
if [[ ! -e "$run_dir/mapping_complete" ]]; then
    [[ ! -e "$run_dir" ]] || { echo "Inspect partial4FPS COLMAP output before retrying: $run_dir" >&2; exit 1; }
    mkdir -p "$run_dir/sparse"
    "$env_prefix/bin/colmap" feature_extractor \
        --database_path "$run_dir/database.db" --image_path "$dataset_dir/images" \
        --image_list_path "$dataset_dir/train.txt" \
        --ImageReader.single_camera 1 --ImageReader.camera_model SIMPLE_RADIAL \
        --SiftExtraction.use_gpu 1 --SiftExtraction.gpu_index 0 \
        --SiftExtraction.max_image_size 1920 --SiftExtraction.max_num_features 8192
    "$env_prefix/bin/colmap" matches_importer \
        --database_path "$run_dir/database.db" --match_list_path "$dataset_dir/pairs.txt" \
        --match_type pairs --SiftMatching.use_gpu 1 --SiftMatching.gpu_index 0
    echo 'Feature extraction/matching used GPU. COLMAP3.9.1 Mapper/BA below uses CPU.'
    "$env_prefix/bin/colmap" mapper \
        --database_path "$run_dir/database.db" --image_path "$dataset_dir/images" \
        --output_path "$run_dir/sparse" --Mapper.num_threads 1
    touch "$run_dir/mapping_complete"
fi
"$env_prefix/bin/python" - "$project_dir" "$dataset_dir" <<'PY'
from pathlib import Path
import sys
from nerfstudio.data.utils import colmap_parsing_utils as reader
project, root = map(Path, sys.argv[1:])
models = []
for path in sorted((root / 'colmap/initial/sparse').iterdir()):
    if not (path / 'images.bin').is_file():
        continue
    cameras, images, points = reader.read_model(path, ext='.bin')
    print('Sparse model:', path.name, '| cameras:', len(images), '| points:', len(points))
    if set(cameras) == {1} and points:
        camera = cameras[1]
        error = sum(p.error for p in points.values()) / len(points)
        if .25 * camera.width < camera.params[0] < 3 * camera.width and abs(camera.params[-1]) < .5 and error < 2:
            models.append((len(images), -error, path))
assert models, 'No plausible model; inspect host-pipeline.log before training.'
model = max(models, key=lambda row: row[:2])[2]
sys.path.insert(0, str(project / 'code'))
from finish_scene2 import finish
finish(model, dataset_root=root)
PY
bash "$project_dir/code/run_scene2_training.sh" "$dataset_dir/processed" "$result_dir" scene2_4fps_big
"$env_prefix/bin/python" "$project_dir/code/compare_reconstructions.py" \
    --baseline-config "$project_dir/results/splatfacto/scene2/runs/scene2_big/splatfacto/quality_30k_v1/config.yml" \
    --candidate-config "$result_dir/runs/scene2_4fps_big/splatfacto/quality_30k_v1/config.yml" \
    --output "$result_dir/common_test_metrics.json"
