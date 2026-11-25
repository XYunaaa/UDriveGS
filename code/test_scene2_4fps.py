import csv
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

import numpy as np


def module(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(name + '.py'))
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


class FourFpsTests(unittest.TestCase):
    def test_host_pipeline_requests_gpu_sift_1920_8192_and_isolated_outputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            code = root / 'code'
            code.mkdir()
            for name in ['run_scene2_4fps_host.sh', 'run_scene2_training.sh']:
                shutil.copy(Path(__file__).with_name(name), code / name)
            binaries = root / '.envs/splatfacto/bin'
            binaries.mkdir(parents=True)
            for name in ['python', 'colmap', 'ns-train', 'ns-eval', 'ns-export', 'ns-render']:
                path = binaries / name
                path.write_text('#!/usr/bin/python3\nimport json,os,sys\n'
                                'from pathlib import Path\n'
                                'source=sys.stdin.read() if Path(sys.argv[0]).name=="python" and sys.argv[1]=="-" else ""\n'
                                'with open(os.environ["COMMAND_LOG"],"a") as f: f.write(json.dumps({"args":sys.argv,"source":source,"cc":os.getenv("CC"),"cache":os.getenv("TORCH_EXTENSIONS_DIR")})+"\\n")\n')
                path.chmod(0o755)
            log = root / 'commands.jsonl'
            subprocess.run(['bash', str(code / 'run_scene2_4fps_host.sh')],
                           env=dict(os.environ, COMMAND_LOG=str(log), CC='/wrong/gcc15',
                                    TORCH_EXTENSIONS_DIR='/wrong/cache'), check=True,
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            commands = [json.loads(line) for line in log.read_text().splitlines()]
            extract = next(c['args'] for c in commands if 'feature_extractor' in c['args'])
            for option, value in [('--SiftExtraction.use_gpu', '1'), ('--SiftExtraction.max_image_size', '1920'),
                                  ('--SiftExtraction.max_num_features', '8192')]:
                self.assertEqual(extract[extract.index(option) + 1], value)
            match = next(c['args'] for c in commands if 'matches_importer' in c['args'])
            self.assertEqual(match[match.index('--SiftMatching.use_gpu') + 1], '1')
            train = next(c['args'] for c in commands if Path(c['args'][0]).name == 'ns-train')
            self.assertEqual(train[train.index('--data') + 1], str(root / 'data/scene2_4fps_splatfacto/processed'))
            self.assertEqual(train[train.index('--experiment-name') + 1], 'scene2_4fps_big')
            self.assertTrue(any('finish(model, dataset_root=root)' in c['source'] for c in commands))
            compare_call = next(c for c in commands if any(a.endswith('compare_reconstructions.py') for a in c['args']))
            self.assertEqual(compare_call['cc'], '/usr/bin/gcc-12')
            self.assertEqual(compare_call['cache'], str(root / '.cache/torch_extensions_gcc12'))
            comparison = compare_call['args']
            self.assertEqual(comparison[comparison.index('--output') + 1],
                             str(root / 'results/splatfacto/scene2_4fps/common_test_metrics.json'))
            self.assertFalse((root / 'data/scene2_splatfacto').exists())

    def test_training_entrypoint_uses_variant_paths_without_touching_baseline(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'code').mkdir()
            entry = root / 'code/run_scene2_training.sh'
            shutil.copy(Path(__file__).with_name(entry.name), entry)
            binaries = root / '.envs/splatfacto/bin'
            binaries.mkdir(parents=True)
            for name in ['python', 'ns-train', 'ns-eval', 'ns-export', 'ns-render']:
                p = binaries / name
                p.write_text('#!/usr/bin/python3\nimport json,os,sys\n'
                             'from pathlib import Path\n'
                             'if Path(sys.argv[0]).name=="python": sys.stdin.read()\n'
                             'with open(os.environ["COMMAND_LOG"],"a") as f: f.write(json.dumps(sys.argv)+"\\n")\n')
                p.chmod(0o755)
            dataset, results = root / 'data/variant/processed', root / 'results/variant'
            log = root / 'commands.jsonl'
            subprocess.run(['bash', str(entry), str(dataset), str(results), 'variant_big'],
                           env=dict(os.environ, COMMAND_LOG=str(log)), check=True,
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            commands = [json.loads(line) for line in log.read_text().splitlines()]
            train = next(c for c in commands if Path(c[0]).name == 'ns-train')
            self.assertEqual(train[train.index('--data') + 1], str(dataset))
            self.assertEqual(train[train.index('--output-dir') + 1], str(results / 'runs'))
            self.assertEqual(train[train.index('--experiment-name') + 1], 'variant_big')
            evaluate = next(c for c in commands if Path(c[0]).name == 'ns-eval')
            self.assertEqual(evaluate[evaluate.index('--load-config') + 1],
                             str(results / 'runs/variant_big/splatfacto/quality_30k_v1/config.yml'))
            self.assertFalse((root / 'results/splatfacto/scene2').exists())

    def test_sampling_interval_is_configurable_and_validated(self):
        prep = module('prepare_scene')
        self.assertEqual(prep.sample_indices([0., .25, .5, .75, 1.], interval_seconds=.25), [0, 1, 2, 3, 4])
        self.assertEqual(prep.sample_indices([0., .25, .5, .75, 1.]), [0, 2, 4])
        for interval in [0., -1., float('nan')]:
            with self.subTest(interval=interval), self.assertRaises(ValueError):
                prep.sample_indices([0., 1.], interval_seconds=interval)

    def test_relative_sharpness_rejects_blur_and_preserves_local_flat_surfaces(self):
        prep = module('prepare_multivideo')
        self.assertTrue(hasattr(prep, 'sharpness_selection'), 'Sharpness selection is absent')
        times = np.arange(12) * .25
        decisions = prep.sharpness_selection(times, [100.] * 3 + [5.] + [100.] * 8)
        self.assertEqual(decisions[3]['decision'], 'rejected_blur')
        self.assertEqual(sum(d['keep'] for d in decisions), 11)
        scores = [100.] * 8 + [1.] * 16 + [100.] * 8
        decisions = prep.sharpness_selection(np.arange(32) * .25, scores)
        self.assertTrue(all(d['keep'] for d in decisions[12:20]))
        decisions = prep.sharpness_selection(times, [100.] * 4 + [5., 5.] + [100.] * 6)
        self.assertTrue(any(d['decision'] == 'coverage_fallback' for d in decisions[4:6]))
        for index in range(0, 12, 2):
            self.assertTrue(any(d['keep'] for d in decisions[index:index + 2]))

    def test_pairs_preserve_time_horizon_at_four_fps_and_exclude_holdouts(self):
        prep = module('prepare_multivideo')
        rows = [{'image': f'{clip}/{i:04d}.png', 'video': clip, 'relative_seconds': i * .25, 'split': split}
                for clip, split in [('a', 'train'), ('b', 'train'), ('heldout', 'test')] for i in range(20)]
        pairs = prep.matching_pairs(rows, temporal_seconds=4., anchor_seconds=2.)
        self.assertIn(('a/0000.png', 'a/0016.png'), pairs)
        self.assertNotIn(('a/0000.png', 'a/0017.png'), pairs)
        self.assertIn(('a/0000.png', 'b/0008.png'), pairs)
        self.assertTrue(all(not n.startswith('heldout/') for pair in pairs for n in pair))

    def test_variant_retains_identical_holdout_images_and_does_not_mutate_baseline(self):
        prep = module('prepare_multivideo')
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / 'source'
            source.mkdir()
            sample = source / 'IMG_9763.MOV'
            subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-f', 'lavfi',
                            '-i', 'testsrc=size=32x24:rate=4:duration=2', '-c:v', 'mpeg4', str(sample)], check=True)
            for name in ['IMG_9765.MOV', 'IMG_9780.MOV']:
                (source / name).write_bytes(sample.read_bytes())
            baseline = root / 'baseline'
            prep.prepare_many(source, baseline)
            original = (baseline / 'manifest.json').read_bytes()
            output = root / 'variant'
            prep.prepare_many(source, output, fps=4, filter_sharpness=True, baseline=baseline)
            old = json.loads((baseline / 'splits.json').read_text())
            new = json.loads((output / 'splits.json').read_text())
            self.assertGreater(len(new['train']), len(old['train']))
            for split in ['val', 'test']:
                self.assertEqual(new[split], old[split])
                for name in old[split]:
                    self.assertEqual((output / 'images' / name).read_bytes(), (baseline / 'images' / name).read_bytes())
            self.assertEqual((baseline / 'manifest.json').read_bytes(), original)
            with (output / 'quality/selection.csv').open() as stream:
                audit = list(csv.DictReader(stream))
            self.assertEqual(len(audit), 8)
            self.assertEqual({r['video'] for r in audit}, {'IMG_9763.MOV'})
            manifest = json.loads((output / 'manifest.json').read_text())
            self.assertEqual(manifest['colmap_parameters']['max_image_size'], 1920)
            self.assertEqual(manifest['colmap_parameters']['max_num_features'], 8192)


if __name__ == '__main__':
    unittest.main()
