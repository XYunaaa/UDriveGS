"""Localize held-out views without retriangulating or changing the training maps."""

import csv
import json
from pathlib import Path
import sqlite3
import subprocess

import numpy as np


def temporal_pairs(heldout, training, times):
    lower, upper = min(times[n] for n in training), max(times[n] for n in training)
    return [(image, neighbor) for image in heldout if lower <= times[image] <= upper
            for neighbor in training if abs(times[image] - times[neighbor]) <= 5]


def run_segment(label, model_index):
    from nerfstudio.data.utils import colmap_parsing_utils as reader

    project = Path(__file__).resolve().parents[1]
    root = project / 'data/scene1_splatfacto'
    colmap = project / '.envs/splatfacto/bin/colmap'
    model = root / f'colmap/full_unmasked/sparse_single_thread/{model_index}'
    output = root / f'colmap/heldout_{label}'
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite {output}')
    output.mkdir()
    (output / 'sparse').mkdir()
    before_images = reader.read_images_binary(model / 'images.bin')
    before_points = reader.read_points3D_binary(model / 'points3D.bin')
    before_cameras = reader.read_cameras_binary(model / 'cameras.bin')
    train_names = sorted(image.name for image in before_images.values())
    splits = json.loads((root / 'splits.json').read_text())
    assert set(train_names) <= set(splits['train'])
    times = {r['image']: float(r['relative_seconds']) for r in csv.DictReader((root / 'frames.csv').open())}
    pairs = temporal_pairs(splits['val'] + splits['test'], train_names, times)
    candidates = sorted({p[0] for p in pairs})
    (output / 'candidates.txt').write_text('\n'.join(candidates) + '\n')
    (output / 'pairs.txt').write_text('\n'.join(f'{a} {b}' for a, b in pairs) + '\n')
    source = sqlite3.connect(f'file:{root}/colmap/full_unmasked/database.db?mode=ro', uri=True)
    database = output / 'database.db'
    dest = sqlite3.connect(database)
    source.backup(dest)
    source.close()
    keep_ids = set(before_images)
    dest.execute('PRAGMA foreign_keys=OFF')
    for table in ['matches', 'two_view_geometries']:
        for (pair_id,) in dest.execute(f'SELECT pair_id FROM {table}').fetchall():
            a, b = divmod(pair_id, 2147483647)
            if a not in keep_ids or b not in keep_ids:
                dest.execute(f'DELETE FROM {table} WHERE pair_id=?', (pair_id,))
    for (image_id,) in dest.execute('SELECT image_id FROM images').fetchall():
        if image_id not in keep_ids:
            for table in ['keypoints', 'descriptors', 'images']:
                dest.execute(f'DELETE FROM {table} WHERE image_id=?', (image_id,))
    camera = before_cameras[1]
    dest.execute('UPDATE cameras SET params=?,prior_focal_length=1 WHERE camera_id=1', (camera.params.tobytes(),))
    dest.commit()
    assert {name for name, in dest.execute('SELECT name FROM images')} == set(train_names)
    dest.close()
    commands = [
        [str(colmap), 'feature_extractor', '--database_path', str(database), '--image_path', str(root / 'images'),
         '--image_list_path', str(output / 'candidates.txt'), '--ImageReader.existing_camera_id', '1',
         '--SiftExtraction.use_gpu', '0', '--SiftExtraction.num_threads', '4', '--SiftExtraction.max_image_size', '960'],
        [str(colmap), 'matches_importer', '--database_path', str(database), '--match_list_path', str(output / 'pairs.txt'),
         '--match_type', 'pairs', '--SiftMatching.use_gpu', '0', '--SiftMatching.num_threads', '4'],
        [str(colmap), 'image_registrator', '--database_path', str(database), '--input_path', str(model),
         '--output_path', str(output / 'sparse'), '--Mapper.num_threads', '1',
         '--Mapper.ba_refine_focal_length', '0', '--Mapper.ba_refine_extra_params', '0'],
    ]
    for number, command in enumerate(commands):
        with (output / f'step_{number}.log').open('w') as log:
            subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
    after_images = reader.read_images_binary(output / 'sparse/images.bin')
    after_points = reader.read_points3D_binary(output / 'sparse/points3D.bin')
    after_cameras = reader.read_cameras_binary(output / 'sparse/cameras.bin')
    assert set(before_points) == set(after_points)
    for key, point in before_points.items():
        assert np.array_equal(point.xyz, after_points[key].xyz)
        assert np.array_equal(point.rgb, after_points[key].rgb)
    for key, camera in before_cameras.items():
        np.testing.assert_allclose(camera.params, after_cameras[key].params, rtol=0, atol=1e-10)
    for key, image in before_images.items():
        np.testing.assert_allclose(image.qvec, after_images[key].qvec, rtol=0, atol=1e-10)
        np.testing.assert_allclose(image.tvec, after_images[key].tvec, rtol=0, atol=1e-10)
    names = {image.name for image in after_images.values()}
    assert names <= set(train_names + candidates)
    result = {'segment': label, 'frozen_training_model': str(model), 'training_geometry_unchanged': True,
              'training_cameras_unchanged': True, 'training_frames': len(before_images), 'candidate_heldout': candidates,
              'localized_val': sorted(names & set(splits['val'])), 'localized_test': sorted(names & set(splits['test'])),
              'failed_candidates': sorted(set(candidates) - names),
              'outside_segment_span': sorted(set(splits['val'] + splits['test']) - set(candidates)),
              'localized_model': str(output / 'sparse'), 'commands': commands}
    (output / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: (len(v) if isinstance(v, list) else v) for k, v in result.items() if k != 'commands'}), flush=True)


if __name__ == '__main__':
    run_segment('a', 1)
    run_segment('b', 2)
