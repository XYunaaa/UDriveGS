"""Export Nerfstudio-aligned camera paths for SuperSplat's native Timeline."""

import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path
import tempfile

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


# SuperSplat imports both PLY and INRIA cameras with a180deg Z rotation.
# This extra rotation makes Nerfstudio's Z-up data Y-up in the editor.
MODEL_BASIS = Rotation.from_euler('x', 90, degrees=True).as_matrix()


def transform_pose(rotation, center, affine, scale):
    affine = np.asarray(affine, dtype=float)
    if affine.shape != (3, 4) or not np.isfinite(affine).all() or not np.isfinite(scale) or scale <= 0:
        raise ValueError('Expected a finite3x4 training transform and positive scale')
    np.testing.assert_allclose(affine[:, :3].T @ affine[:, :3], np.eye(3), atol=1e-5)
    return (MODEL_BASIS @ affine[:, :3] @ rotation,
            MODEL_BASIS @ (scale * (affine[:, :3] @ center + affine[:, 3])))


def sample_track(poses, fps=30):
    times = np.array([p['time'] for p in poses], dtype=float)
    if fps <= 0 or not len(times) or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError('Track timestamps must be finite and strictly increasing; FPS must be positive')
    times -= times[0]
    duration = float(times[-1])
    samples = np.minimum(np.arange(max(1, math.ceil(duration * fps - 1e-8) + 1)) / fps, duration)
    positions = np.array([p['position'] for p in poses], dtype=float)
    matrices = np.array([p['rotation'] for p in poses], dtype=float)
    if positions.shape != (len(poses), 3) or not np.isfinite(positions).all() or not np.isfinite(matrices).all():
        raise ValueError('Camera positions and rotations must be finite')
    np.testing.assert_allclose(matrices.transpose(0, 2, 1) @ matrices,
                               np.tile(np.eye(3), (len(poses), 1, 1)), atol=1e-5)
    rotations = (Slerp(times, Rotation.from_matrix(matrices))(samples).as_matrix()
                 if len(poses) > 1 else matrices)
    centers = np.stack([np.interp(samples, times, positions[:, axis]) for axis in range(3)], axis=1)
    result = []
    for i, (time, center, rotation) in enumerate(zip(samples, centers, rotations)):
        result.append({'id': i + 1, 'img_name': f"{Path(poses[0]['img_name']).parent}/time_{time:.4f}",
                       'position': center.tolist(), 'rotation': rotation.tolist(),
                       **{key: poses[0][key] for key in ['width', 'height', 'fx', 'fy']}})
    return result


def export(ply, run_dir, colmap, output):
    import torch
    import yaml
    from nerfstudio.data.datamanagers.full_images_datamanager import _undistort_image
    from nerfstudio.data.utils import colmap_parsing_utils as reader

    if output.exists():
        raise FileExistsError(f'Refusing to overwrite {output}')
    with ply.open('rb') as stream:
        header = []
        for _ in range(256):
            line = stream.readline().decode('ascii').strip()
            header.append(line)
            if line == 'end_header':
                break
        else:
            raise ValueError('Invalid or unexpectedly large PLY header')
    for field in ['x', 'y', 'z', 'f_dc_0', 'opacity', 'scale_0', 'rot_0']:
        if f'property float {field}' not in header:
            raise ValueError(f'Expected Gaussian PLY field {field}; ordinary point clouds need a point-cloud viewer')
    if 'comment Vertical Axis: z' not in header:
        raise ValueError('This exporter expects the original Z-up Nerfstudio PLY, without additional editor transforms')
    saved = json.loads((run_dir / 'dataparser_transforms.json').read_text())
    settings = yaml.load((run_dir / 'config.yml').read_text(), Loader=yaml.Loader)
    if settings.pipeline.model.camera_optimizer.mode != 'off':
        raise ValueError('Learned camera corrections must be exported from the checkpoint when camera optimization is on')
    if settings.pipeline.datamanager.dataparser.orientation_method != 'up':
        raise ValueError('Expected the current Z-up training orientation')
    dataset = settings.pipeline.datamanager.data
    settings.pipeline.datamanager.dataparser.data = dataset
    parsed = settings.pipeline.datamanager.dataparser.setup().get_dataparser_outputs(split='train')
    with (dataset.parent / 'frames.csv').open() as stream:
        timestamps = {row['image']: row for row in csv.DictReader(stream)}
    _, images, _ = reader.read_model(colmap, ext='.bin')
    by_name = {image.name: image for image in images.values()}
    calibrations = {}
    max_position_error = 0.
    for i, file in enumerate(parsed.image_filenames):
        name = str(file.relative_to(dataset / 'images'))
        image = by_name[name]
        rotation = image.qvec2rotmat().T
        converted_r, converted_c = transform_pose(rotation, -rotation @ image.tvec,
                                                  np.array(saved['transform']), saved['scale'])
        native = parsed.cameras.camera_to_worlds[i].numpy().copy()
        native[:, 1:3] *= -1
        np.testing.assert_allclose(converted_r, MODEL_BASIS @ native[:, :3], atol=1e-5)
        np.testing.assert_allclose(converted_c, MODEL_BASIS @ native[:, 3], atol=1e-5)
        max_position_error = max(max_position_error, float(np.max(np.abs(converted_c - MODEL_BASIS @ native[:, 3]))))
        if image.camera_id not in calibrations:
            camera = parsed.cameras[i].reshape(())
            width, height = int(camera.width), int(camera.height)
            K = camera.get_intrinsics_matrices().numpy()
            if camera.distortion_params is not None and torch.any(camera.distortion_params != 0):
                K, reference, _ = _undistort_image(camera, camera.distortion_params.numpy(), {},
                                                  np.zeros((height, width, 3), dtype=np.uint8), K)
                height, width = reference.shape[:2]
            calibrations[image.camera_id] = {'width': width, 'height': height,
                                             'fx': float(K[0, 0]), 'fy': float(K[1, 1])}
    tracks = defaultdict(list)
    for image in images.values():
        row = timestamps[image.name]
        rotation = image.qvec2rotmat().T
        rotation, center = transform_pose(rotation, -rotation @ image.tvec,
                                           np.array(saved['transform']), saved['scale'])
        tracks[Path(image.name).parent.name].append({
            'time': float(row['relative_seconds']), 'img_name': image.name,
            'position': center.tolist(), 'rotation': rotation.tolist(), **calibrations[image.camera_id]})
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.camera-paths-', dir=output.parent) as temporary:
        stage = Path(temporary) / 'paths'
        stage.mkdir()
        manifest = {'ply': str(ply.resolve()), 'run': str(run_dir.resolve()), 'colmap': str(colmap.resolve()),
                    'model_rotation_euler_degrees': [90, 0, 180], 'model_position': [0, 0, 0], 'model_scale': [1, 1, 1],
                    'timeline_fps': 30, 'loop': False, 'smoothness': 0,
                    'max_native_parser_position_difference': max_position_error,
                    'scope': 'SuperSplat position/forward playback; camera roll is not retained by its JSON importer',
                    'source_camera_count': len(images), 'tracks': []}
        for clip, poses in sorted(tracks.items()):
            poses.sort(key=lambda p: p['time'])
            samples = sample_track(poses)
            filename = f'{clip}.cameras.json'
            (stage / filename).write_text(json.dumps(samples, separators=(',', ':')) + '\n')
            manifest['tracks'].append({'video': clip, 'file': filename, 'source_cameras': len(poses),
                                       'timeline_frames': len(samples), 'source_duration_seconds': poses[-1]['time'] - poses[0]['time'],
                                       'playback_duration_seconds': (len(samples) - 1) / 30})
        (stage / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        stage.rename(output)
    print(json.dumps({'output': str(output), 'tracks': len(tracks), 'source_cameras': len(images),
                      'max_native_parser_position_difference': max_position_error}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ply', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--colmap', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    export(args.ply, args.run_dir, args.colmap, args.output)
