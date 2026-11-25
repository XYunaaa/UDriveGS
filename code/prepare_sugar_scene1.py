"""Adapt an unchanged scene1/scene2 PLY and its actual training cameras to SuGaR."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

import cv2
import numpy as np


SCENE_INPUTS = {
    'scene1': ('scene1/exports/segment_a_30k/splat.ply',
               'scene1/runs/full_unmasked/scene1_segment_a/splatfacto/initial_30k_v1'),
    'scene2': ('scene2/exports/scene2_big_30k/splat.ply',
               'scene2/runs/scene2_big/splatfacto/quality_30k_v1'),
}


def image_name(path, image_root):
    return '__'.join(path.relative_to(image_root).with_suffix('').parts) + '.png'


def camera_entry(pose, K, width, height, name, index):
    pose, K = np.asarray(pose), np.asarray(K)
    if (pose.shape != (3, 4) or K.shape != (3, 3) or not np.isfinite(pose).all()
            or not np.isfinite(K).all() or min(width, height, K[0, 0], K[1, 1]) <= 0):
        raise ValueError('Invalid camera pose or calibration')
    R = pose[:, :3]
    if not np.allclose(R.T @ R, np.eye(3), atol=1e-5) or not np.isclose(np.linalg.det(R), 1, atol=1e-5):
        raise ValueError('Camera rotation must be rigid')
    if not np.allclose(K[:2, 2], [width / 2, height / 2], atol=1e-5):
        raise ValueError('SuGaR requires centered pinhole calibration')
    return {'id': index, 'img_name': Path(name).stem, 'width': width, 'height': height,
            'position': pose[:, 3].tolist(), 'rotation': (R @ np.diag([1., -1., -1.])).tolist(),
            'fx': float(K[0, 0]), 'fy': float(K[1, 1])}


def center_image(image, K):
    K = np.array(K, dtype=float, copy=True)
    h, w = image.shape[:2]
    shift = np.array([w / 2, h / 2]) - K[:2, 2]
    centered = cv2.warpAffine(image, np.c_[np.eye(2), shift], (w, h), flags=cv2.INTER_LINEAR)
    K[:2, 2] = [w / 2, h / 2]
    return centered, K


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def prepare(project, output, scene='scene1'):
    import torch
    import yaml
    from plyfile import PlyData
    from nerfstudio.data.datamanagers.full_images_datamanager import _undistort_image
    from export_checkpoint_cpu import ply_attributes

    if output.exists():
        raise FileExistsError(f'Refusing to overwrite {output}')
    ply_relative, run_relative = SCENE_INPUTS[scene]
    ply = project / 'results/splatfacto' / ply_relative
    run = project / 'results/splatfacto' / run_relative
    checkpoint = run / 'nerfstudio_models/step-000029999.ckpt'
    settings = yaml.load((run / 'config.yml').read_text(), Loader=yaml.Loader)
    if settings.pipeline.model.camera_optimizer.mode != 'off':
        raise ValueError('Learned camera corrections require checkpoint camera export')
    settings.pipeline.datamanager.dataparser.data = settings.pipeline.datamanager.data
    parsed = settings.pipeline.datamanager.dataparser.setup().get_dataparser_outputs(split='train')
    original = PlyData.read(ply, mmap=True)['vertex'].data
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    params = {k.removeprefix('_model.gauss_params.'): v for k, v in state['pipeline'].items()
              if k.startswith('_model.gauss_params.')}
    expected = ply_attributes(params)
    if len(original) != len(expected['x']):
        raise ValueError('PLY does not match the training checkpoint Gaussian count')
    for name, values in expected.items():
        np.testing.assert_allclose(original[name], values, atol=1e-6, rtol=1e-5, err_msg=f'Checkpoint/PLY mismatch: {name}')
    if len([n for n in original.dtype.names if n.startswith('f_rest_')]) != 45:
        raise ValueError('Expected degree3 SH coefficients for upstream SuGaR')
    if np.any(np.linalg.norm(np.stack([original[f'rot_{i}'] for i in range(4)], axis=1), axis=1) == 0):
        raise ValueError('Gaussian quaternion must be nonzero')
    del state, params, expected

    manifest = {'scene': scene, 'source_ply': str(ply), 'source_ply_sha256': sha256(ply), 'run': str(run),
                'source_gaussians': len(original), 'checkpoint_attributes_verified': True,
                'coordinate_system': 'unchanged Nerfstudio training world, Z-up; cameras OpenCV C2W; no SuperSplat transform',
                'camera_count': len(parsed.image_filenames), 'split': 'train only',
                'extraction': 'SuGaR vanilla3dgs direct extraction, no regularization or refinement',
                'cameras': []}
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.sugar-input-', dir=output.parent) as temporary:
        stage = Path(temporary) / 'input'
        image_dir = stage / 'scene/images'
        point_dir = stage / 'gs/point_cloud/iteration_30000'
        image_dir.mkdir(parents=True)
        point_dir.mkdir(parents=True)
        cameras = []
        for i, path in enumerate(parsed.image_filenames):
            camera = parsed.cameras[i].reshape(())
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None or image.shape[:2] != (int(camera.height), int(camera.width)):
                raise ValueError(f'Unreadable image or incorrect dimensions: {path}')
            K = camera.get_intrinsics_matrices().numpy()
            if camera.distortion_params is not None and torch.any(camera.distortion_params != 0):
                K, image, _ = _undistort_image(camera, camera.distortion_params.numpy(), {}, image, K)
            h, w = image.shape[:2]
            shift = (np.array([w / 2, h / 2]) - K[:2, 2]).tolist()
            image, K = center_image(image, K)
            name = image_name(path, settings.pipeline.datamanager.dataparser.data / 'images')
            if (image_dir / name).exists():
                raise ValueError(f'Duplicate image basename: {name}')
            if not cv2.imwrite(str(image_dir / name), image):
                raise OSError(f'Unable to write image: {name}')
            cameras.append(camera_entry(camera.camera_to_worlds.numpy(), K, w, h, name, i))
            manifest['cameras'].append({'source': str(path), 'output': name, 'principal_point_shift_pixels': shift})
        (stage / 'gs/cameras.json').write_text(json.dumps(cameras, indent=2) + '\n')
        # Copy rather than link: upstream experiments must never write through to the original PLY.
        copied = point_dir / 'point_cloud.ply'
        shutil.copyfile(ply, copied)
        if sha256(copied) != manifest['source_ply_sha256']:
            raise ValueError('Copied PLY hash mismatch')
        (stage / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        stage.rename(output)
    print(json.dumps({'output': str(output), 'cameras': len(cameras), 'gaussians': len(original),
                      'checkpoint_attributes_verified': True}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene', choices=SCENE_INPUTS, default='scene1')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    prepare(Path(__file__).resolve().parents[1], args.output.resolve(), args.scene)
