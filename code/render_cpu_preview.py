"""Qualitative CPU reference previews; not a substitute for native CUDA evaluation."""

import argparse
import json
from pathlib import Path
import time

import numpy as np


def compose(means2d, conics, radii, depths, opacities, colors, width, height, background):
    rgb = np.zeros((height, width, 3), dtype=np.float32)
    transmittance = np.ones((height, width), dtype=np.float32)
    visible = np.flatnonzero((radii > 0) & np.isfinite(depths))
    for i in visible[np.argsort(depths[visible])]:
        mx, my = means2d[i]
        radius = radii[i]
        x0 = max(0, int(np.floor((mx - radius) / 16)) * 16)
        y0 = max(0, int(np.floor((my - radius) / 16)) * 16)
        x1 = min(width, int(np.ceil((mx + radius) / 16)) * 16)
        y1 = min(height, int(np.ceil((my + radius) / 16)) * 16)
        if x1 <= x0 or y1 <= y0:
            continue
        remaining = transmittance[y0:y1, x0:x1]
        if not np.any(remaining > 1e-4):
            continue
        dx = np.arange(x0, x1, dtype=np.float32)[None, :] + .5 - mx
        dy = np.arange(y0, y1, dtype=np.float32)[:, None] + .5 - my
        a, b, c = conics[i]
        sigma = .5 * (a * dx * dx + 2 * b * dx * dy + c * dy * dy)
        alpha = np.minimum(.999, opacities[i] * np.exp(-np.maximum(sigma, 0)))
        alpha[(sigma < 0) | (alpha < 1 / 255) | (remaining <= 1e-4)] = 0
        weight = remaining * alpha
        rgb[y0:y1, x0:x1] += weight[..., None] * colors[i]
        remaining *= 1 - alpha
    rgb += transmittance[..., None] * background
    return np.clip(rgb, 0, 1), 1 - transmittance


def main(checkpoint, config, output):
    import cv2
    from gsplat.cuda._torch_impl import _fully_fused_projection, _quat_scale_to_covar_preci, _spherical_harmonics
    from nerfstudio.data.datamanagers.full_images_datamanager import _undistort_image
    from PIL import Image
    import torch
    import yaml

    torch.set_num_threads(4)
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    params = {key.removeprefix('_model.gauss_params.'): value
              for key, value in state['pipeline'].items() if key.startswith('_model.gauss_params.')}
    settings = yaml.load(config.read_text(), Loader=yaml.Loader)
    settings.pipeline.datamanager.dataparser.data = settings.pipeline.datamanager.data
    parsed = settings.pipeline.datamanager.dataparser.setup().get_dataparser_outputs(split='train')
    covars, _ = _quat_scale_to_covar_preci(params['quats'], params['scales'].exp(), compute_preci=False)
    coefficients = torch.cat([params['features_dc'][:, None, :], params['features_rest']], dim=1)
    opacities = params['opacities'].sigmoid().squeeze(-1).numpy()
    background = np.array([.1490, .1647, .2157], dtype=np.float32)
    output.mkdir(parents=True, exist_ok=True)
    results = []
    jobs = [(index, 'training', 0.) for index in [0, 10, 20, 30, 40, 51]]
    jobs += [(20, 'novel_left', -.03), (20, 'novel_right', .03)]
    for index, kind, offset in jobs:
        started = time.perf_counter()
        camera = parsed.cameras[index].reshape(())
        reference = np.asarray(Image.open(parsed.image_filenames[index]).convert('RGB'))
        K = camera.get_intrinsics_matrices().numpy()
        if camera.distortion_params is not None and torch.any(camera.distortion_params != 0):
            K, reference, _ = _undistort_image(camera, camera.distortion_params.numpy(), {}, reference, K)
        width, height = 480, 272
        K[0] *= width / reference.shape[1]
        K[1] *= height / reference.shape[0]
        reference = cv2.resize(reference, (width, height), interpolation=cv2.INTER_AREA)
        pose = camera.camera_to_worlds.clone()
        pose[:, 1:3] *= -1
        pose[:, 3] += offset * pose[:, 0]
        view = torch.eye(4)
        view[:3, :3] = pose[:, :3].T
        view[:3, 3] = -pose[:, :3].T @ pose[:, 3]
        with torch.no_grad():
            radii, xy, depths, conics, _ = _fully_fused_projection(
                params['means'], covars, view[None], torch.from_numpy(K)[None], width, height)
            colors = (_spherical_harmonics(3, params['means'] - pose[:, 3], coefficients) + .5).clamp_min(0).numpy()
        rgb, alpha = compose(xy[0].numpy(), conics[0].numpy(), radii[0].numpy(), depths[0].numpy(),
                             opacities, colors, width, height, background)
        assert np.isfinite(rgb).all() and np.isfinite(alpha).all()
        filename = parsed.image_filenames[index].name
        suffix = f'{index:02d}' if kind == 'training' else kind
        Image.fromarray(np.rint(rgb * 255).astype(np.uint8)).save(output / f'render_{suffix}.png')
        Image.fromarray(np.rint(alpha * 255).astype(np.uint8)).save(output / f'alpha_{suffix}.png')
        row = {'index': index, 'frame': filename, 'kind': kind, 'camera_offset_sfm_units': offset,
               'alpha_above_0p5_fraction': float(np.mean(alpha > .5)),
               'cpu_seconds': time.perf_counter() - started}
        if kind == 'training':
            Image.fromarray(reference).save(output / f'reference_{suffix}.png')
            mse = float(np.mean((rgb - reference.astype(np.float32) / 255) ** 2))
            row['psnr_db'] = float(-10 * np.log10(max(mse, 1e-12)))
        results.append(row)
        print(json.dumps(row), flush=True)
    summary = {'renderer': 'gsplat 1.4 pure-PyTorch projection and SH; NumPy reference alpha compositing',
               'resolution': [480, 272], 'split': 'six training views and two nearby virtual views; no held-out evaluation',
               'native_cuda_evaluation': False, 'views': results,
               'mean_training_view_psnr_db': float(np.mean([row['psnr_db'] for row in results if row['kind'] == 'training']))}
    assert all('psnr_db' not in row for row in results if row['kind'] != 'training')
    (output / 'metrics.json').write_text(json.dumps(summary, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    main(args.checkpoint, args.config, args.output)
