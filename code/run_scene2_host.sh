#!/usr/bin/env bash
# Host continuation: reuse completed features/matches, map on CPU, train on GPU.
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
env_prefix="$project_dir/.envs/splatfacto"
dataset_dir="$project_dir/data/scene2_splatfacto"
run_dir="$dataset_dir/colmap/host_resume"
export PYTHONNOUSERSITE=1
export PYTHONFAULTHANDLER=1
export PYTHONUNBUFFERED=1
export QT_QPA_PLATFORM=offscreen
export MPLCONFIGDIR="$project_dir/.cache/matplotlib"
unset PYTHONPATH
mkdir -p "$project_dir/results/splatfacto/scene2"
exec > >(tee -a "$project_dir/results/splatfacto/scene2/host-pipeline.log") 2>&1
"$env_prefix/bin/python" - "$dataset_dir" <<'PY'
import json
from pathlib import Path
import sqlite3
import sys
import torch
assert torch.cuda.is_available(), 'Run this command in the host GPU terminal.'
root = Path(sys.argv[1])
train = set(json.loads((root / 'splits.json').read_text())['train'])
with sqlite3.connect(f'file:{root}/colmap/initial/database.db?mode=ro', uri=True) as database:
    assert {name for name, in database.execute('SELECT name FROM images')} == train
    assert database.execute('SELECT COUNT(*) FROM descriptors').fetchone()[0] == len(train)
    expected = len((root / 'pairs.txt').read_text().splitlines())
    assert database.execute('SELECT COUNT(*) FROM two_view_geometries').fetchone()[0] == expected
print('GPU:', torch.cuda.get_device_name(0))
print(f'Reusing {len(train)} training images and {expected} completed match pairs.')
PY
if [[ ! -e "$run_dir/mapping_complete" ]]; then
    [[ ! -e "$run_dir" ]] || { echo "Inspect partial host mapper output before retrying: $run_dir" >&2; exit 1; }
    mkdir -p "$run_dir/sparse"
    echo 'COLMAP3.9.1 Mapper/BA uses CPU; completed SIFT/matching will not be repeated.'
    "$env_prefix/bin/colmap" mapper \
        --database_path "$dataset_dir/colmap/initial/database.db" \
        --image_path "$dataset_dir/images" --output_path "$run_dir/sparse" \
        --Mapper.num_threads 1
    touch "$run_dir/mapping_complete"
fi
"$env_prefix/bin/python" - "$project_dir" <<'PY'
import json
from pathlib import Path
import sys
from nerfstudio.data.utils import colmap_parsing_utils as reader
project = Path(sys.argv[1])
root = project / 'data/scene2_splatfacto'
models = []
for path in sorted((root / 'colmap/host_resume/sparse').iterdir()):
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
finish(model)
PY
bash "$project_dir/code/run_scene2_training.sh"
