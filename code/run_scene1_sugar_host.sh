#!/usr/bin/env bash
# Run in the host GPU terminal; optional scene2 argument, default scene1.
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
base_env="$project_dir/.envs/splatfacto"
env_prefix="$project_dir/.envs/sugar"
scene="${1:-scene1}"
case "$scene" in
    scene1) result_dir="$project_dir/results/mesh/scene1_segment_a_sugar" ;;
    scene2) result_dir="$project_dir/results/mesh/scene2_sugar" ;;
    *) echo 'Usage: bash code/run_scene1_sugar_host.sh [scene1|scene2]' >&2; exit 1 ;;
esac
repo_dir="$project_dir/code/third_party/SuGaR"
mesh_dir="$result_dir/mesh_vanilla_preview_v1"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export MPLCONFIGDIR="$project_dir/.cache/matplotlib"
unset PYTHONPATH
mkdir -p "$result_dir"
exec > >(tee -a "$result_dir/host-extraction.log") 2>&1
if [[ ! -e "$result_dir/input" ]]; then
    "$base_env/bin/python" "$project_dir/code/prepare_sugar_scene1.py" --scene "$scene" --output "$result_dir/input"
fi
"$base_env/bin/python" - "$result_dir" "$scene" <<'PY'
import json
from pathlib import Path
import sys
import torch
from plyfile import PlyData
root = Path(sys.argv[1])
manifest = json.loads((root / 'input/manifest.json').read_text())
assert manifest.get('scene', 'scene1') == sys.argv[2], 'Prepared input belongs to a different scene'
assert manifest['checkpoint_attributes_verified'] and manifest['camera_count'] > 0
assert len(json.loads((root / 'input/gs/cameras.json').read_text())) == manifest['camera_count']
assert len(PlyData.read(root / 'input/gs/point_cloud/iteration_30000/point_cloud.ply', mmap=True)['vertex']) == manifest['source_gaussians']
if not torch.cuda.is_available():
    status = {'status': 'blocked_cuda', 'mesh_generated': False,
              'reason': 'CUDA is inaccessible in this process. Run in the host GPU terminal.'}
    (root / 'status.json').write_text(json.dumps(status, indent=2) + '\n')
    raise SystemExit(status['reason'])
print('GPU:', torch.cuda.get_device_name(0))
PY
if [[ ! -x "$env_prefix/bin/python" ]] || ! "$env_prefix/bin/python" -c \
    'import torch; import pytorch3d._C; import simple_knn._C; import diff_gaussian_rasterization._C' >/dev/null 2>&1; then
    bash "$project_dir/code/setup_sugar.sh" "$result_dir"
fi
[[ ! -e "$mesh_dir" ]] || { echo "Inspect existing or partial mesh output before retrying: $mesh_dir" >&2; exit 1; }
cd "$repo_dir"
"$env_prefix/bin/python" - "$result_dir" <<'PY'
import json
from pathlib import Path
import sys
root = Path(sys.argv[1])
source = Path('sugar_extractors/coarse_mesh.py').read_text()
assert 'n_total_points = 1_000_000' in source
assert 'surface_level_n_points_per_pass = 200_000' in source
assert 'poisson_depth = 9' in source
(root / 'status.json').write_text(json.dumps({'status': 'extracting', 'mesh_generated': False}, indent=2) + '\n')
PY
"$env_prefix/bin/python" extract_mesh.py \
    --scene_path "$result_dir/input/scene" \
    --checkpoint_path "$result_dir/input/gs/" \
    --iteration_to_load 30000 --use_vanilla_3dgs True --eval False \
    --surface_level 0.3 --decimation_target 100000 \
    --project_mesh_on_surface_points False --gpu 0 --mesh_output_dir "$mesh_dir"
"$env_prefix/bin/python" - "$mesh_dir" "$result_dir" <<'PY'
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import open3d as o3d
import trimesh
mesh_dir, root = map(Path, sys.argv[1:])
manifest = json.loads((root / 'input/manifest.json').read_text())
files = list(mesh_dir.glob('*.ply'))
assert len(files) == 1, f'Expected one mesh, found: {files}'
source = files[0]
mesh = o3d.io.read_triangle_mesh(str(source))
vertices, faces = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
assert len(vertices) > 0 and len(faces) > 0, 'Empty mesh output'
assert np.isfinite(vertices).all() and faces.min() >= 0 and faces.max() < len(vertices)
assert mesh.has_vertex_colors(), 'Expected colored mesh'
colors = np.asarray(mesh.vertex_colors)
assert np.isfinite(colors).all()
converted = trimesh.Trimesh(vertices=vertices, faces=faces,
                            vertex_colors=np.rint(np.clip(colors, 0, 1) * 255).astype(np.uint8), process=False)
# glTF is Y-up; retain the original Z-up coordinates in the mesh PLY.
glb_basis = np.array([[1.,0.,0.,0.], [0.,0.,1.,0.], [0.,-1.,0.,0.], [0.,0.,0.,1.]])
converted.apply_transform(glb_basis)
glb = mesh_dir / ('scene2_mesh.glb' if manifest.get('scene', 'scene1') == 'scene2' else 'segment_a_mesh.glb')
assert not glb.exists()
converted.export(glb)
reload = trimesh.load(glb, force='scene')
assert sum(len(g.faces) for g in reload.geometry.values()) == len(faces)
original = Path(manifest['source_ply'])
digest = hashlib.sha256()
with original.open('rb') as stream:
    for block in iter(lambda: stream.read(1024 * 1024), b''):
        digest.update(block)
assert digest.hexdigest() == manifest['source_ply_sha256'], 'Original Gaussian PLY changed'
summary = {'status': 'complete', 'mesh_generated': True, 'ply': str(source), 'glb': str(glb),
           'vertices': len(vertices), 'triangles': len(faces), 'z_up_bounds': np.asarray(mesh.get_axis_aligned_bounding_box().get_box_points()).tolist(),
           'glb_from_ply_transform': glb_basis.tolist(), 'original_ply_unchanged': True,
           'scope': 'SuGaR vanilla3dgs direct extraction preview; vertex colors, no UV texture or PBR material recovery; no surface regularization/refinement.'}
(root / 'mesh_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
(root / 'status.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps(summary, indent=2))
PY
