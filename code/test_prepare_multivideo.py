import csv
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest


class MultiVideoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).with_name('prepare_multivideo.py')
        if path.exists():
            spec = importlib.util.spec_from_file_location('prepare_multivideo', path)
            cls.module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(cls.module)
        else:
            cls.module = None

    def test_pair_graph_links_training_clips_and_excludes_holdouts(self):
        self.assertIsNotNone(self.module, 'Multi-video preparation is absent')
        rows = [{'image': f'{clip}/frame_{i:06d}.png', 'video': clip + '.MOV',
                 'relative_seconds': i * .5, 'split': split}
                for clip, split in [('a', 'train'), ('b', 'train'), ('val', 'val'), ('test', 'test')]
                for i in range(10)]
        pairs = self.module.matching_pairs(rows)
        self.assertIn(('a/frame_000000.png', 'b/frame_000004.png'), pairs)
        self.assertIn(('a/frame_000000.png', 'a/frame_000008.png'), pairs)
        self.assertNotIn(('a/frame_000000.png', 'a/frame_000009.png'), pairs)
        self.assertTrue(all(not name.startswith(('val/', 'test/')) for pair in pairs for name in pair))
        self.assertEqual(len(pairs), len(set(pairs)))

    def test_real_extraction_namespaces_frames_and_freezes_whole_videos(self):
        self.assertIsNotNone(self.module, 'Multi-video preparation is absent')
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / 'source'
            source.mkdir()
            sample = source / 'IMG_9763.MOV'
            subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-f', 'lavfi',
                            '-i', 'testsrc=size=32x24:rate=4:duration=2', '-c:v', 'mpeg4',
                            str(sample)], check=True)
            for name in ['IMG_9765.MOV', 'IMG_9780.MOV']:
                (source / name).write_bytes(sample.read_bytes())
            output = root / 'prepared'
            self.module.prepare_many(source, output)
            splits = json.loads((output / 'splits.json').read_text())
            self.assertEqual({k: len(v) for k, v in splits.items()}, {'train': 4, 'val': 4, 'test': 4})
            self.assertTrue(all(n.startswith('IMG_9765/') for n in splits['val']))
            self.assertTrue(all(n.startswith('IMG_9780/') for n in splits['test']))
            names = [n for split in splits.values() for n in split]
            self.assertEqual(len(set(names)), 12)
            self.assertTrue(all((output / 'images' / n).is_file() for n in names))
            with (output / 'frames.csv').open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 12)
            self.assertEqual(len((output / 'pairs.txt').read_text().splitlines()), 6)
            before = (output / 'manifest.json').read_bytes()
            with self.assertRaises(FileExistsError):
                self.module.prepare_many(source, output)
            self.assertEqual((output / 'manifest.json').read_bytes(), before)


if __name__ == '__main__':
    unittest.main()
