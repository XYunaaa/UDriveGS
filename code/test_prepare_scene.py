"""Run with: python3 -m unittest discover -s code -p 'test_*.py'."""

import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest


SCRIPT = Path(__file__).with_name('prepare_scene.py')


class SamplingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if SCRIPT.exists():
            spec = importlib.util.spec_from_file_location('prepare_scene', SCRIPT)
            cls.module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(cls.module)
        else:
            cls.module = None

    def test_nearest_real_pts_and_no_duplicates(self):
        self.assertIsNotNone(self.module, 'prepare_scene.py is not implemented')
        self.assertEqual(self.module.sample_indices([2.0, 2.2, 2.7, 3.05, 3.48]), [0, 2, 3])
        self.assertEqual(self.module.sample_indices([0.0, 1.0, 2.0]), [0, 1, 2])

    def test_invalid_pts_fail_before_extraction(self):
        self.assertIsNotNone(self.module, 'prepare_scene.py is not implemented')
        for timestamps in [[], [0, 0], [1, 0], [0, float('nan')]]:
            with self.subTest(timestamps=timestamps), self.assertRaises(ValueError):
                self.module.sample_indices(timestamps)

    def test_two_second_blocks_have_one_split(self):
        self.assertIsNotNone(self.module, 'prepare_scene.py is not implemented')
        for second in range(120):
            expected = 'val' if (second // 2) % 10 == 8 else 'test' if (second // 2) % 10 == 9 else 'train'
            self.assertEqual(self.module.split_for_time(second + 0.25), expected)
        self.assertEqual(self.module.split_for_time(15.999), 'train')
        self.assertEqual(self.module.split_for_time(16), 'val')
        self.assertEqual(self.module.split_for_time(18), 'test')
        self.assertEqual(self.module.split_for_time(20), 'train')

    def test_extract_video_and_refuse_overwrite(self):
        self.assertIsNotNone(self.module, 'prepare_scene.py is not implemented')
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            video = root / 'sample.mkv'
            subprocess.run([
                'ffmpeg', '-hide_banner', '-loglevel', 'error', '-f', 'lavfi',
                '-i', 'testsrc=size=32x24:rate=4:duration=2', '-c:v', 'ffv1', str(video),
            ], check=True)
            output = root / 'prepared'
            result = self.module.prepare(video, output)
            self.assertEqual(result['candidates'], 4)
            self.assertEqual(sorted(p.name for p in (output / 'images').glob('*.png')),
                             [f'frame_{i:06d}.png' for i in (0, 2, 4, 6)])
            manifest_before = (output / 'manifest.json').read_bytes()
            self.assertEqual(json.loads(manifest_before)['source_frame_count'], 8)
            with self.assertRaises(FileExistsError):
                self.module.prepare(video, output)
            self.assertEqual((output / 'manifest.json').read_bytes(), manifest_before)


if __name__ == '__main__':
    unittest.main()
