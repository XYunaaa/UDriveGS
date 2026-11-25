"""Localize frozen scene2 holdouts and convert the accepted training map."""

import argparse
import csv
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import tempfile

import numpy as np


def assert_frozen(before, after):
    cameras, images, points = before
    new_cameras, new_images, new_points = after
    assert set(cameras) == set(new_cameras)
    assert set(points) == set(new_points)
    for key, camera in cameras.items():
        np.testing.assert_allclose(camera.params, new_cameras[key].params, rtol=0, atol=1e-10)
    for key, image in images.items():
        assert image.name == new_images[key].name
        np.testing.assert_allclose(image.qvec, new_images[key].qvec, rtol=0, atol=1e-10)
        np.testing.assert_allclose(image.tvec, new_images[key].tvec, rtol=0, atol=1e-10)
    for key, point in points.items():
        np.testing.assert_array_equal(point.xyz, new_points[key].xyz)
        np.testing.assert_array_equal(point.rgb, new_points[key].rgb)


def finish(model, dataset_root=None):
    from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
    from nerfstudio.data.utils import colmap_parsing_utils as reader
    from nerfstudio.process_data.colmap_utils import colmap_to_json

    project = Path(__file__).resolve().parents[1]
    root = Path(dataset_root).resolve() if dataset_root is not None else project / 'data/scene2_splatfacto'
    binary = project / '.envs/splatfacto/bin/colmap'
    manifest = json.loads((root / 'manifest.json').read_text())
    extraction = manifest.get('colmap_parameters', {'max_image_size': 1600, 'max_num_features': 4096})
    splits = json.loads((root / 'splits.json').read_text())
    before = reader.read_model(model, ext='.bin')
    cameras, images, points = before
    training = {image.name for image in images.values()}
    assert set(cameras) == {1}, 'Initial scene2 calibration expects one shared camera'
    assert training <= set(splits['train'])
    assert len(training) >= .9 * len(splits['train']), 'Single map covers less than90% of training frames'
    camera = cameras[1]
    assert .25 * camera.width < camera.params[0] < 3 * camera.width
    assert abs(camera.params[-1]) < .5
    assert np.mean([point.error for point in points.values()]) < 2
    output = root / 'colmap/heldout'
    summary_path = output / 'summary.json'
    if not summary_path.exists():
        if output.exists():
            raise FileExistsError(f'Partial localization exists; inspect before retrying: {output}')
        (output / 'sparse').mkdir(parents=True)
        candidates = sorted(splits['val'] + splits['test'])
        anchors = []
        anchor_interval = manifest.get('anchor_interval_seconds')
        if anchor_interval is not None:
            from prepare_scene import sample_indices
            with (root / 'frames.csv').open() as stream:
                times = {row['image']: float(row['relative_seconds']) for row in csv.DictReader(stream)}
        for clip in sorted({Path(name).parent for name in training}):
            names = sorted(name for name in training if Path(name).parent == clip)
            indices = (range(0, len(names), 4) if anchor_interval is None else
                       sample_indices([times[name] for name in names], interval_seconds=anchor_interval))
            anchors.extend(names[index] for index in indices)
        pairs = [(candidate, anchor) for candidate in candidates for anchor in anchors]
        (output / 'candidates.txt').write_text('\n'.join(candidates) + '\n')
        (output / 'pairs.txt').write_text('\n'.join(f'{a} {b}' for a, b in pairs) + '\n')
        database = output / 'database.db'
        with sqlite3.connect(f'file:{root}/colmap/initial/database.db?mode=ro', uri=True) as source:
            with sqlite3.connect(database) as dest:
                source.backup(dest)
                keep_ids = set(images)
                for table in ['matches', 'two_view_geometries']:
                    for (pair_id,) in dest.execute(f'SELECT pair_id FROM {table}').fetchall():
                        a, b = divmod(pair_id, 2147483647)
                        if a not in keep_ids or b not in keep_ids:
                            dest.execute(f'DELETE FROM {table} WHERE pair_id=?', (pair_id,))
                for (image_id,) in dest.execute('SELECT image_id FROM images').fetchall():
                    if image_id not in keep_ids:
                        for table in ['keypoints', 'descriptors', 'images']:
                            dest.execute(f'DELETE FROM {table} WHERE image_id=?', (image_id,))
                dest.execute('UPDATE cameras SET params=?,prior_focal_length=1 WHERE camera_id=1',
                             (camera.params.tobytes(),))
                assert {n for n, in dest.execute('SELECT name FROM images')} == training
        commands = [
            [str(binary), 'feature_extractor', '--database_path', str(database), '--image_path', str(root / 'images'),
             '--image_list_path', str(output / 'candidates.txt'), '--ImageReader.existing_camera_id', '1',
             '--SiftExtraction.use_gpu', '1', '--SiftExtraction.num_threads', '8',
             '--SiftExtraction.max_image_size', str(extraction['max_image_size']),
             '--SiftExtraction.max_num_features', str(extraction['max_num_features'])],
            [str(binary), 'matches_importer', '--database_path', str(database),
             '--match_list_path', str(output / 'pairs.txt'), '--match_type', 'pairs',
             '--SiftMatching.use_gpu', '1', '--SiftMatching.num_threads', '8'],
            [str(binary), 'image_registrator', '--database_path', str(database), '--input_path', str(model),
             '--output_path', str(output / 'sparse'), '--Mapper.num_threads', '1',
             '--Mapper.ba_refine_focal_length', '0', '--Mapper.ba_refine_extra_params', '0'],
        ]
        for i, command in enumerate(commands):
            print(f'Localization stage {i}: {command[1]}', flush=True)
            with (output / f'step_{i}.log').open('w') as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        after = reader.read_model(output / 'sparse', ext='.bin')
        assert_frozen(before, after)
        names = {image.name for image in after[1].values()}
        assert names <= training | set(candidates)
        summary = {'frozen_model': str(model.resolve()), 'training_geometry_unchanged': True,
                   'training_frames': len(training), 'training_total': len(splits['train']),
                   'localized': {split: sorted(names & set(splits[split])) for split in ['val', 'test']},
                   'heldout_total': {split: len(splits[split]) for split in ['val', 'test']},
                   'failed': sorted(set(candidates) - names), 'commands': commands}
        summary_path.write_text(json.dumps(summary, indent=2) + '\n')
    summary = json.loads(summary_path.read_text())
    assert summary['frozen_model'] == str(model.resolve())
    after = reader.read_model(output / 'sparse', ext='.bin')
    assert_frozen(before, after)
    names = {image.name for image in after[1].values()}
    actual = {split: sorted(names & set(values)) for split, values in splits.items()}
    assert all(actual.values()), 'Training, validation and test must all contain localized views'
    processed = root / 'processed'
    if not processed.exists():
        with tempfile.TemporaryDirectory(prefix='.processed-', dir=root) as temporary:
            stage = Path(temporary) / 'dataset'
            stage.mkdir()
            count = colmap_to_json(output / 'sparse', stage)
            assert count == len(names)
            meta = json.loads((stage / 'transforms.json').read_text())
            for split, values in actual.items():
                meta[f'{split}_filenames'] = ['images/' + name for name in values]
            assert {f['file_path'] for f in meta['frames']} == {'images/' + name for name in names}
            (stage / 'transforms.json').write_text(json.dumps(meta, indent=2) + '\n')
            for name in names:
                target = stage / 'images' / name
                target.parent.mkdir(parents=True, exist_ok=True)
                os.link(root / 'images' / name, target)
            (stage / 'provenance.json').write_text(json.dumps({
                'source_training_model': str(model.resolve()), 'training_geometry_unchanged': True,
                'counts': {k: len(v) for k, v in actual.items()},
                'original_counts': {k: len(v) for k, v in splits.items()},
                'source_holdouts': {'val': 'IMG_9765.MOV', 'test': 'IMG_9780.MOV'},
                'status': 'poses_and_sparse_initialization_ready; no Gaussian training yet',
            }, indent=2) + '\n')
            stage.rename(processed)
    provenance = json.loads((processed / 'provenance.json').read_text())
    assert provenance['source_training_model'] == str(model.resolve())
    parser = NerfstudioDataParserConfig(data=processed, downscale_factor=1, load_3D_points=True).setup()
    for split, expected in actual.items():
        parsed = parser.get_dataparser_outputs(split=split)
        assert {str(p.relative_to(processed / 'images')) for p in parsed.image_filenames} == set(expected)
        assert len(parsed.metadata['points3D_xyz']) == len(points)
    print(json.dumps({'processed': str(processed), 'counts': {k: len(v) for k, v in actual.items()},
                      'failed_heldout': len(summary['failed'])}, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--dataset-root', type=Path)
    args = parser.parse_args()
    finish(args.model, args.dataset_root)
