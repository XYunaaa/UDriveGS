"""Train-only Depth Anything V2 supervision for the existing Splatfacto renderer.

Targets are calibrated inverse camera-Z in Nerfstudio's normalized world, never
ray distance. Confidence combines sparse-fit support, flip agreement, supported
prediction range and removal of depth edges; it is a heuristic, not ground truth.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import urllib.request

import numpy as np
import torch
import torch.nn.functional as F
from nerfstudio.models.splatfacto import SplatfactoModel, SplatfactoModelConfig


SOURCE_URL = 'https://github.com/DepthAnything/Depth-Anything-V2'
SOURCE_COMMIT = 'a561b849ebae10a6f5ef49e26c83cbbcd36c71bf'
WEIGHTS_URL = ('https://huggingface.co/depth-anything/Depth-Anything-V2-Base/'
               'resolve/main/depth_anything_v2_vitb.pth')
WEIGHTS_SHA256 = '0d2b7002e62d39d655571c371333340bd88f67ab95050c03591555aa05645328'
PROJECT = Path(__file__).resolve().parents[1]


def align_inverse_depth(prediction, inverse_z):
    """Robust positive affine fit to sparse correspondences; reject weak fits."""
    x, y = np.asarray(prediction, dtype=float), np.asarray(inverse_z, dtype=float)
    usable = np.isfinite(x) & np.isfinite(y) & (x >= 0) & (y > 0)
    a, b = x[usable], y[usable]
    if a.size < 30 or np.ptp(a) < 1e-6:
        raise ValueError('Fewer than 30 usable observations or flat prediction')
    rng = np.random.default_rng(42)
    best = np.zeros(a.size, dtype=bool)
    best_error = float('inf')
    for i, j in rng.integers(0, len(a), size=(256, 2)):
        if abs(a[i] - a[j]) < 1e-8:
            continue
        scale = (b[i] - b[j]) / (a[i] - a[j])
        if scale <= 0:
            continue
        shift = b[i] - scale * a[i]
        error = np.abs(scale * a + shift - b) / b
        good = error < 0.15
        score = float(np.minimum(error, 0.15).mean())
        if good.sum() > best.sum() or (good.sum() == best.sum() and score < best_error):
            best, best_error = good, score
    for _ in range(4):
        if best.sum() < 30:
            raise ValueError('Insufficient affine-fit inliers')
        scale, shift = np.linalg.lstsq(np.column_stack([a[best], np.ones(best.sum())]), b[best], rcond=None)[0]
        error = np.abs(scale * a + shift - b) / b
        best = error < 0.15
    median_error = float(np.median(error))
    if scale <= 0 or best.sum() < 30 or best.mean() < 0.6 or median_error > 0.1:
        raise ValueError('Sparse inverse-depth alignment failed quality thresholds')
    inliers = np.zeros(x.shape, dtype=bool)
    inliers[usable] = best
    return float(scale), float(shift), inliers, {
        'observations': int(a.size), 'inliers': int(best.sum()),
        'inlier_fraction': float(best.mean()), 'median_relative_error': median_error,
    }


def masked_inverse_depth_loss(depth, target_inverse_z, valid):
    """Dimensionless L1 inverse-Z loss, with safe zero loss for empty masks."""
    valid = valid.bool() & torch.isfinite(depth) & (depth > 0)
    valid = valid & torch.isfinite(target_inverse_z) & (target_inverse_z > 0)
    safe_depth = torch.where(valid, depth, torch.ones_like(depth))
    if not valid.any():
        return safe_depth.sum() * 0
    target = target_inverse_z[valid]
    return (safe_depth.reciprocal()[valid] - target).abs().mean() / target.mean().clamp_min(1e-6)


@dataclass
class CalibratedDepthConfig(SplatfactoModelConfig):
    _target: type = field(default_factory=lambda: CalibratedDepthModel)
    depth_root: Path = Path('.')
    depth_loss_weight: float = 0.01
    depth_warmup_steps: int = 1000
    output_depth_during_training: bool = True


def depth_model_config(base_model_config, depth_root: Path):
    """Preserve every base model setting and add only depth-specific fields."""
    depth_root = Path(depth_root).resolve()
    manifest = json.loads((depth_root / 'manifest.json').read_text())
    if manifest['status'] != 'complete' or manifest.get('image_coverage', 0) < 0.7:
        raise ValueError('Depth preparation has not completed with adequate coverage')
    total = len(manifest['identity']['training_images'])
    accepted = sum(record['status'] == 'accepted' for record in manifest['frames'].values())
    if not total or len(manifest['frames']) != total or accepted / total < 0.7:
        raise ValueError('Depth manifest is incomplete or has inadequate accepted images')
    for index, record in manifest['frames'].items():
        if record['status'] == 'accepted':
            target = depth_root / f'{int(index):06d}.npz'
            if _sha256(target) != record['target_sha256']:
                raise ValueError(f'Depth target changed: {target}')
    config = CalibratedDepthConfig()
    config.__dict__.update(deepcopy(base_model_config.__dict__))
    config._target = CalibratedDepthModel
    config.depth_root = depth_root
    config.output_depth_during_training = True
    return config


@lru_cache(maxsize=8)
def _read_target(filename: str):
    with np.load(filename, allow_pickle=False) as data:
        return torch.from_numpy(data['inverse_z'].astype(np.float32)), torch.from_numpy(data['valid'])


class CalibratedDepthModel(SplatfactoModel):
    config: CalibratedDepthConfig

    def get_outputs(self, camera):
        outputs = super().get_outputs(camera)
        if self.training and outputs.get('depth') is None:
            raise RuntimeError('Calibrated supervision requires output_depth_during_training=True')
        return outputs

    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        losses = super().get_loss_dict(outputs, batch, metrics_dict)
        if not self.training or self.step < self.config.depth_warmup_steps:
            return losses
        path = self.config.depth_root / f'{int(batch["image_idx"]):06d}.npz'
        if not path.exists():
            manifest = json.loads((self.config.depth_root / 'manifest.json').read_text())
            if manifest['frames'][str(int(batch['image_idx']))]['status'] == 'rejected':
                return losses
            raise FileNotFoundError(f'Accepted depth target is missing: {path}')
        target, valid = _read_target(str(path))
        depth = outputs['depth']
        size = depth.shape[:2]
        # Require all source pixels valid, so resampling does not spread invalid depths.
        coverage = F.interpolate(valid[None, None].float(), size=size, mode='area')
        target = F.interpolate((target * valid)[None, None], size=size, mode='area')
        target = (target / coverage.clamp_min(1e-6))[0, 0].to(depth.device)
        valid = (coverage[0, 0] > 0.999).to(depth.device)
        losses['calibrated_depth_loss'] = self.config.depth_loss_weight * masked_inverse_depth_loss(
            depth[..., 0], target, valid)
        return losses


def _sha256(path: Path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _write_manifest(path: Path, manifest):
    temporary = path.with_suffix('.json.part')
    temporary.write_text(json.dumps(manifest, indent=2) + '\n')
    temporary.replace(path)


def _load_depth_model():
    repository = PROJECT / 'code/third_party/Depth-Anything-V2'
    if not repository.exists():
        subprocess.run(['git', 'clone', '--depth', '1', SOURCE_URL, str(repository)], check=True)
    commit = subprocess.check_output(['git', '-C', str(repository), 'rev-parse', 'HEAD'], text=True).strip()
    if commit != SOURCE_COMMIT:
        raise RuntimeError(f'Depth Anything source revision changed: {commit}')
    weights = repository / 'checkpoints/depth_anything_v2_vitb.pth'
    if not weights.exists():
        weights.parent.mkdir(exist_ok=True)
        partial = weights.with_suffix('.pth.part')
        urllib.request.urlretrieve(WEIGHTS_URL, partial)
        if _sha256(partial) != WEIGHTS_SHA256:
            raise RuntimeError('Official Base checkpoint download checksum mismatch')
        partial.replace(weights)
    if _sha256(weights) != WEIGHTS_SHA256:
        raise RuntimeError('Official Base checkpoint checksum mismatch')
    sys.path.insert(0, str(repository))
    from depth_anything_v2.dpt import DepthAnythingV2
    model = DepthAnythingV2(encoder='vitb', features=128, out_channels=[96, 192, 384, 768])
    model.load_state_dict(torch.load(weights, map_location='cpu', weights_only=True), strict=True)
    return model.cuda().eval(), {
        'repository': SOURCE_URL, 'commit': commit, 'weights_url': WEIGHTS_URL,
        'weights_sha256': WEIGHTS_SHA256, 'encoder': 'vitb', 'input_size': 518,
        'license': 'Apache-2.0 code; CC-BY-NC-4.0 Base weights (official README)',
    }


def prepare_depths(base_config: Path, output_root: Path) -> Path:
    """Stream native training images only; fit against each image's frozen tracks.

    Runs only when CUDA is normally accessible. No RGB copies or eval-image
    inference are produced. Resume requires matching sources and target hashes.
    """
    if not torch.cuda.is_available():
        raise RuntimeError('Depth preparation requires accessible CUDA; run the host queue')
    import cv2
    from scene2_quality import load_config
    from nerfstudio.data.datasets.base_dataset import InputDataset
    from nerfstudio.data.datamanagers.full_images_datamanager import _undistort_image
    from nerfstudio.data.utils import colmap_parsing_utils as colmap

    base_config = Path(base_config).resolve()
    config = load_config(base_config)
    manager = deepcopy(config.pipeline.datamanager)
    if config.data is not None:
        manager.data = config.data
    if manager.data is not None:
        manager.dataparser.data = manager.data
    data_root = Path(manager.dataparser.data)
    provenance = json.loads((data_root / 'provenance.json').read_text())
    if provenance.get('training_geometry_unchanged') is not True:
        raise RuntimeError('Training geometry provenance is not frozen')
    sparse_root = Path(provenance['source_training_model'])
    if not sparse_root.is_absolute():
        sparse_root = PROJECT / sparse_root
    metadata = json.loads((data_root / 'transforms.json').read_text())
    training_names = set(metadata['train_filenames'])
    if training_names & (set(metadata['val_filenames']) | set(metadata['test_filenames'])):
        raise RuntimeError('Training and held-out filenames overlap')
    parser = manager.dataparser.setup()
    outputs = parser.get_dataparser_outputs(split='train')
    names = [str(path.relative_to(data_root)) for path in outputs.image_filenames]
    if len(names) != len(training_names) or set(names) != training_names:
        raise RuntimeError('Native parser train split differs from frozen manifest')
    dataset = InputDataset(outputs, scale_factor=manager.camera_res_scale_factor)
    _, images, points = colmap.read_model(sparse_root, ext='.bin')
    frames = {frame['file_path']: frame for frame in metadata['frames']}
    image_lookup = {image.name: image for image in images.values()}
    if set(image_lookup) != {str(Path(name).relative_to('images')) for name in training_names}:
        raise RuntimeError('Frozen COLMAP map must contain exactly the training images')
    target_root = Path(output_root).resolve()
    target_root.mkdir(parents=True, exist_ok=True)
    manifest_path = target_root / 'manifest.json'
    source_paths = [base_config, data_root / 'transforms.json', data_root / 'provenance.json',
                    data_root / metadata['ply_file_path']]
    source_paths += [sparse_root / f'{name}.bin' for name in ['cameras', 'images', 'points3D']]
    sources = {str(path): _sha256(path) for path in source_paths}
    transform = outputs.dataparser_transform.cpu().numpy()
    world_scale = float(outputs.dataparser_scale)
    # Check original tracks and native PLY initialization really share this world.
    from scipy.spatial import cKDTree
    all_xyz = np.array([point.xyz for point in points.values()])
    normalized = (all_xyz @ transform[:, :3].T + transform[:, 3]) * world_scale
    distance = cKDTree(outputs.metadata['points3D_xyz'].cpu().numpy()).query(normalized)[0]
    if not np.isfinite(distance).all() or distance.max() > 1e-4:
        raise RuntimeError('Frozen COLMAP tracks differ from normalized native sparse geometry')
    model, model_provenance = _load_depth_model()
    identity = {
        'sources': sources, 'model': model_provenance, 'training_images': names,
        'dataparser_transform': outputs.dataparser_transform.tolist(),
        'dataparser_scale': float(outputs.dataparser_scale),
        'camera_res_scale_factor': manager.camera_res_scale_factor,
        'max_normalized_sparse_geometry_distance': float(distance.max()),
        'implementation_sha256': _sha256(Path(__file__)),
    }
    manifest = {'identity': identity, 'status': 'preparing', 'frames': {},
                'confidence': 'positive robust sparse affine inverse-Z fit; flip error<=15%; '
                              'inlier prediction range; relative inverse-depth gradient<=15%',
                'thresholds': {'min_observations': 30, 'min_inlier_fraction': 0.6,
                               'max_median_relative_error': 0.1, 'min_spatial_bins_of_16': 6,
                               'min_valid_pixel_fraction': 0.15, 'min_accepted_image_fraction': 0.7}}
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if previous['identity'] != identity:
            raise RuntimeError('Refusing stale depth targets: source/config/model identity changed')
        manifest['frames'] = previous['frames']
    _write_manifest(manifest_path, manifest)
    for index, (name, filename) in enumerate(zip(names, outputs.image_filenames)):
        image_hash = _sha256(filename)
        saved = manifest['frames'].get(str(index))
        target_path = target_root / f'{index:06d}.npz'
        if saved and saved['image_sha256'] == image_hash:
            if saved['status'] == 'rejected':
                continue
            if target_path.exists() and _sha256(target_path) == saved['target_sha256']:
                continue
        if shutil.disk_usage(target_root).free < 1024**3:
            raise RuntimeError('Less than 1 GiB disk free during depth preparation')
        record = {'image': name, 'image_sha256': image_hash}
        data = dataset.get_data(index, image_type='uint8')
        camera = dataset.cameras[index].reshape(())
        image = data['image'].numpy()
        K = camera.get_intrinsics_matrices().numpy()
        if camera.distortion_params is not None and torch.any(camera.distortion_params != 0):
            K, image, native_mask = _undistort_image(camera, camera.distortion_params.numpy(), data, image, K)
        else:
            native_mask = data.get('mask')
        frozen_image = image_lookup[str(Path(name).relative_to('images'))]
        if frames[name].get('colmap_im_id') != frozen_image.id:
            raise RuntimeError(f'Image identity changed for {name}')
        ids = np.unique(frozen_image.point3D_ids[frozen_image.point3D_ids >= 0])
        # Only original tracks of THIS training image are allowed to supply anchors.
        ids = [int(point_id) for point_id in ids if points[int(point_id)].error <= 2.0]
        xyz = np.array([points[point_id].xyz for point_id in ids], dtype=float).reshape(-1, 3)
        normalized = (xyz @ transform[:, :3].T + transform[:, 3]) * world_scale
        c2w = camera.camera_to_worlds.numpy()
        camera_xyz = (normalized - c2w[:, 3]) @ c2w[:, :3]
        camera_xyz[:, 1:] *= -1  # OpenGL camera to OpenCV camera; positive Z forward.
        z = camera_xyz[:, 2]
        projected = camera_xyz @ K.T
        uv = projected[:, :2] / np.where(z[:, None] > 0, z[:, None], 1) - 0.5
        h, w = image.shape[:2]
        usable = np.isfinite(uv).all(1) & (z > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < w - 1)
        usable &= (uv[:, 1] >= 0) & (uv[:, 1] < h - 1)
        uv, z = uv[usable], z[usable]
        if len(z) < 30:
            target_path.unlink(missing_ok=True)
            record.update(status='rejected', reason='Fewer than 30 visible frozen training tracks')
            manifest['frames'][str(index)] = record
            _write_manifest(manifest_path, manifest)
            continue
        prediction = model.infer_image(np.ascontiguousarray(image[..., :3][..., ::-1]))
        flipped = model.infer_image(np.ascontiguousarray(image[:, ::-1, :3][..., ::-1]))[:, ::-1]
        sampled = cv2.remap(prediction, uv[:, 0].astype(np.float32)[None],
                            uv[:, 1].astype(np.float32)[None], cv2.INTER_LINEAR).ravel()
        try:
            scale, shift, inliers, stats = align_inverse_depth(sampled, 1 / z)
            bins = np.floor(uv[inliers] / [w, h] * 4).astype(int)
            stats['spatial_bins_of_16'] = int(len(np.unique(bins[:, 1] * 4 + bins[:, 0])))
            if stats['spatial_bins_of_16'] < 6:
                raise ValueError('Sparse inliers cover fewer than 6/16 spatial bins')
            inverse_z = scale * prediction + shift
            flip_inverse_z = scale * flipped + shift
            lo, hi = np.quantile(sampled[inliers], [0.02, 0.98])
            valid = np.isfinite(inverse_z) & (inverse_z > 0) & (inverse_z < 65000)
            valid &= (prediction >= lo) & (prediction <= hi)
            valid &= np.abs(inverse_z - flip_inverse_z) <= 0.15 * np.maximum(inverse_z, 1e-6)
            dy, dx = np.gradient(inverse_z)
            valid &= np.maximum(abs(dx), abs(dy)) <= 0.15 * np.maximum(inverse_z, 1e-6)
            if native_mask is not None:
                valid &= np.asarray(native_mask).reshape(h, w).astype(bool)
            stats['valid_pixel_fraction'] = float(valid.mean())
            if valid.mean() < 0.15:
                raise ValueError('Fewer than 15% confident pixels')
            inverse_z = np.where(valid, inverse_z, 0).astype(np.float16)
            valid &= inverse_z > 0
            temporary = target_path.with_suffix('.npz.part')
            with temporary.open('wb') as stream:
                np.savez_compressed(stream, inverse_z=inverse_z, valid=valid)
            temporary.replace(target_path)
            record.update(status='accepted', scale=scale, shift=shift, shape=[h, w],
                          statistics=stats, target_sha256=_sha256(target_path))
        except ValueError as error:
            target_path.unlink(missing_ok=True)
            record.update(status='rejected', reason=str(error))
        manifest['frames'][str(index)] = record
        _write_manifest(manifest_path, manifest)
        print(f'Depth {index + 1}/{len(names)} {record["status"]}: {name}', flush=True)
    accepted = sum(record['status'] == 'accepted' for record in manifest['frames'].values())
    manifest['accepted_images'] = accepted
    manifest['image_coverage'] = accepted / len(names)
    manifest['status'] = 'complete' if manifest['image_coverage'] >= 0.7 else 'inadequate_coverage'
    _write_manifest(manifest_path, manifest)
    if manifest['status'] != 'complete':
        raise RuntimeError(f'Depth supervision coverage inadequate: {accepted}/{len(names)}')
    return target_root
