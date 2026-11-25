"""Native Gaussian evaluation on the intersection of two frozen test-view sets."""

import argparse
import gc
import hashlib
import json
from pathlib import Path


def shared_test_views(frozen_baseline, localized_baseline, frozen_candidate, localized_candidate):
    frozen_baseline, frozen_candidate = set(frozen_baseline), set(frozen_candidate)
    localized_baseline, localized_candidate = set(localized_baseline), set(localized_candidate)
    if frozen_baseline != frozen_candidate:
        raise ValueError('Original frozen test-image lists differ')
    if not localized_baseline <= frozen_baseline or not localized_candidate <= frozen_candidate:
        raise ValueError('Localized test images contain a foreign image')
    common = sorted(localized_baseline & localized_candidate)
    if not common:
        raise ValueError('No shared localized test images')
    return common


def test_view_sets(config_path):
    import yaml

    config = yaml.load(config_path.read_text(), Loader=yaml.Loader)
    dataset = config.pipeline.datamanager.data
    frozen = json.loads((dataset.parent / 'splits.json').read_text())['test']
    meta = json.loads((dataset / 'transforms.json').read_text())
    localized = [Path(name).relative_to('images').as_posix() for name in meta['test_filenames']]
    assert len(set(localized)) == len(localized)
    return dataset, frozen, localized


def evaluate(config_path, dataset, common, image_dir):
    import torch
    from nerfstudio.utils.eval_utils import eval_setup

    _, pipeline, checkpoint, _ = eval_setup(config_path, test_mode='test')
    filenames = [path.relative_to(dataset / 'images').as_posix()
                 for path in pipeline.datamanager.eval_dataset.image_filenames]
    indices = {name: i for i, name in enumerate(filenames)}
    assert len(indices) == len(filenames) and set(common) <= set(indices)
    # Preserve the native undistortion, camera ordering and metric implementations.
    full_loader = pipeline.datamanager.fixed_indices_eval_dataloader
    loader = [full_loader[indices[name]] for name in common]
    with torch.no_grad():
        metrics = pipeline.get_average_image_metrics(loader, 'common_test', output_path=image_dir,
                                                       get_std=len(common) > 1)
    result = {'checkpoint': str(checkpoint), 'metrics': metrics}
    del loader, full_loader, pipeline
    gc.collect()
    torch.cuda.empty_cache()
    return result


def compare(baseline_config, candidate_config, output):
    import torch

    assert torch.cuda.is_available(), 'Common-view native rendering requires the host GPU.'
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite comparison: {output}')
    baseline_dataset, frozen, old_names = test_view_sets(baseline_config)
    candidate_dataset, new_frozen, new_names = test_view_sets(candidate_config)
    common = shared_test_views(frozen, old_names, new_frozen, new_names)
    for name in common:
        old_hash = hashlib.sha256((baseline_dataset / 'images' / name).read_bytes()).digest()
        new_hash = hashlib.sha256((candidate_dataset / 'images' / name).read_bytes()).digest()
        if old_hash != new_hash:
            raise ValueError(f'Test source-image bytes differ: {name}')
    output.parent.mkdir(parents=True, exist_ok=True)
    result = {'scope': 'native rendered image metrics on identical source test frames; not metric geometry validation',
              'common_test_filenames': common, 'common_count': len(common), 'original_test_count': len(frozen),
              'localized_counts': {'baseline': len(old_names), 'candidate': len(new_names)},
              'failed_test_filenames': {'baseline': sorted(set(frozen) - set(old_names)),
                                       'candidate': sorted(set(frozen) - set(new_names))}}
    for label, config, dataset in [('baseline', baseline_config, baseline_dataset),
                                    ('candidate', candidate_config, candidate_dataset)]:
        result[label] = evaluate(config, dataset, common, output.parent / 'common_test_images' / label)
    result['candidate_minus_baseline'] = {key: result['candidate']['metrics'][key] - result['baseline']['metrics'][key]
                                          for key in ['psnr', 'ssim', 'lpips']}
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'common_count': len(common), 'original_test_count': len(frozen),
                      'localized_counts': result['localized_counts'],
                      'candidate_minus_baseline': result['candidate_minus_baseline']}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-config', type=Path, required=True)
    parser.add_argument('--candidate-config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    compare(args.baseline_config, args.candidate_config, args.output)
