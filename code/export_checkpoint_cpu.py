"""Export the existing Splatfacto checkpoint without constructing a CUDA model."""

import argparse
import base64
from collections import OrderedDict
import json
from pathlib import Path

import numpy as np
import torch


def ply_attributes(params):
    arrays = {name: tensor.detach().cpu().numpy() for name, tensor in params.items()}
    count = len(arrays['means'])
    fields = OrderedDict()
    for i, axis in enumerate('xyz'):
        fields[axis] = arrays['means'][:, i]
    for axis in 'xyz':
        fields['n' + axis] = np.zeros(count, dtype=np.float32)
    for i in range(3):
        fields[f'f_dc_{i}'] = arrays['features_dc'][:, i]
    rest = arrays['features_rest'].transpose(0, 2, 1).reshape(count, -1)
    for i in range(rest.shape[1]):
        fields[f'f_rest_{i}'] = rest[:, i]
    fields['opacity'] = arrays['opacities'][:, 0]
    for i in range(3):
        fields[f'scale_{i}'] = arrays['scales'][:, i]
    for i in range(4):
        fields[f'rot_{i}'] = arrays['quats'][:, i]
    valid = np.ones(count, dtype=bool)
    for values in fields.values():
        valid &= np.isfinite(values)
    valid &= fields['opacity'] >= -5.5373  # Same low-opacity filter as the native exporter.
    return OrderedDict((name, values[valid].astype(np.float32)) for name, values in fields.items())


def splat_bytes(fields):
    count = len(fields['x'])
    rows = np.zeros((count, 32), dtype=np.uint8)
    floats = rows[:, :24].copy().view('<f4').reshape(count, 6)
    floats[:, :3] = np.stack([fields[axis] for axis in 'xyz'], axis=1)
    floats[:, 3:] = np.exp(np.stack([fields[f'scale_{i}'] for i in range(3)], axis=1))
    assert np.isfinite(floats).all()
    rows[:, :24] = floats.view(np.uint8).reshape(count, 24)
    colors = 0.5 + 0.28209479177387814 * np.stack([fields[f'f_dc_{i}'] for i in range(3)], axis=1)
    rows[:, 24:27] = np.rint(np.clip(colors, 0, 1) * 255).astype(np.uint8)
    rows[:, 27] = np.rint(255 / (1 + np.exp(-np.clip(fields['opacity'], -80, 80)))).astype(np.uint8)
    quats = np.stack([fields[f'rot_{i}'] for i in range(4)], axis=1)
    norm = np.linalg.norm(quats, axis=1, keepdims=True)
    if np.any(norm == 0):
        raise ValueError('Cannot encode zero-length Gaussian quaternion')
    rows[:, 28:32] = np.rint(np.clip(quats / norm * 128 + 128, 0, 255)).astype(np.uint8)
    return rows.tobytes()


def main(checkpoint, config, output):
    from nerfstudio.scripts.exporter import ExportGaussianSplat
    import yaml

    # This project-generated checkpoint also contains NumPy optimizer scalars.
    checkpoint_data = torch.load(checkpoint, map_location='cpu', weights_only=False)
    params = {key.removeprefix('_model.gauss_params.'): value
              for key, value in checkpoint_data['pipeline'].items() if key.startswith('_model.gauss_params.')}
    fields = ply_attributes(params)
    output.mkdir(parents=True, exist_ok=True)
    ply_path = output / 'splat.ply'
    if ply_path.exists():
        raise FileExistsError(f'Refusing to overwrite {ply_path}')
    ExportGaussianSplat.write_ply(str(ply_path), len(fields['x']), fields)
    packed = splat_bytes(fields)
    (output / 'model.splat').write_bytes(packed)

    settings = yaml.load(config.read_text(), Loader=yaml.Loader)
    if settings.pipeline.datamanager.data is not None:
        settings.pipeline.datamanager.dataparser.data = settings.pipeline.datamanager.data
    parser = settings.pipeline.datamanager.dataparser.setup()
    parsed = parser.get_dataparser_outputs(split='train')
    cameras = []
    for i, file_path in enumerate(parsed.image_filenames):
        pose = parsed.cameras.camera_to_worlds[i].numpy().copy()
        pose[:, 1:3] *= -1  # OpenGL camera axes to the viewer's OpenCV camera axes.
        cameras.append({'id': i, 'img_name': file_path.name, 'width': 960, 'height': 544,
                        'position': pose[:, 3].tolist(), 'rotation': pose[:, :3].tolist(),
                        'fx': float(parsed.cameras.fx[i]), 'fy': float(parsed.cameras.fy[i])})
    assert len(cameras) == 52
    (output / 'cameras.json').write_text(json.dumps(cameras, indent=2) + '\n')
    vendor = Path(__file__).parent / 'vendor/splat'
    script = (vendor / 'main.js').read_text()
    script = 'let cameras = ' + json.dumps(cameras) + ';\n' + script[script.index('let camera = cameras[0];'):]
    script = script.replace('let camera = cameras[0];',
                            'let camera = cameras[Number(new URLSearchParams(location.search).get("view") || 0) % cameras.length];', 1)
    start, end = script.index('let defaultViewMatrix = ['), script.index('let viewMatrix = defaultViewMatrix;')
    script = script[:start] + 'let defaultViewMatrix = getViewMatrix(camera);\n' + script[end:]
    script = script.replace('let carousel = true;', 'let carousel = false;', 1)
    script = script.replace('const znear = 0.2;', 'const znear = 0.01;', 1)
    start, end = script.index('    const url = new URL('), script.index('    console.log(req);')
    script = script[:start] + '''    const model = Uint8Array.from(atob(window.SCENE_B64), c => c.charCodeAt(0));
    const req = new Response(model, {headers: {"content-length": String(model.length)}});
''' + script[end:]
    script = script.replace('gl.clear(gl.COLOR_BUFFER_BIT);\n            gl.drawArraysInstanced',
                            'document.body.dataset.ready = "true";\n            gl.clear(gl.COLOR_BUFFER_BIT);\n            gl.drawArraysInstanced', 1)
    html = (vendor / 'index.html').read_text().replace('<title>WebGL Gaussian Splat Viewer</title>', '<title>Scene1 2000 step Gaussian preview</title>')
    embed = '<script>window.SCENE_B64=' + json.dumps(base64.b64encode(packed).decode()) + ';</script>\n<script>' + script + '</script>'
    html = html.replace('<script src="main.js"></script>', embed)
    html = html.replace('<h3 class="nohf">WebGL 3D Gaussian Splat Viewer</h3>',
                        '<h3>Scene1 · 2,000-step preliminary model</h3><p>Training views · DC-color WebGL preview · not native evaluation</p>')
    (output / 'preview.html').write_text(html)
    (output / 'VIEWER-LICENSE.txt').write_text((vendor / 'LICENSE').read_text())
    summary = {'checkpoint': str(checkpoint), 'step_index': checkpoint_data['step'],
               'checkpoint_gaussians': len(params['means']), 'exported_gaussians': len(fields['x']),
               'ply_bytes': ply_path.stat().st_size, 'preview_color': 'SH DC only',
               'renderer_source': 'https://github.com/antimatter15/splat',
               'scope': 'first 30 seconds, 52 training views, unmasked smoke model'}
    (output / 'export.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    main(args.checkpoint, args.config, args.output)
