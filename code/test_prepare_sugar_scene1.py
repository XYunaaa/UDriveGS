import importlib.util
from pathlib import Path
import unittest

import numpy as np


class SugarAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location('adapter', Path(__file__).with_name('prepare_sugar_scene1.py'))
        cls.adapter = importlib.util.module_from_spec(spec)
        if spec.origin and Path(spec.origin).is_file():
            spec.loader.exec_module(cls.adapter)

    def test_camera_axes_and_world_position_match_nerfstudio(self):
        self.assertTrue(hasattr(self.adapter, 'camera_entry'), 'Camera adapter is absent')
        pose = np.c_[np.eye(3), [1., 2., 3.]]
        K = np.array([[10., 0., 4.], [0., 12., 3.], [0., 0., 1.]])
        entry = self.adapter.camera_entry(pose, K, 8, 6, 'frame_000015.png', 1)
        np.testing.assert_allclose(entry['position'], [1., 2., 3.])
        R = np.array(entry['rotation'])
        np.testing.assert_allclose(R, np.diag([1., -1., -1.]))
        # A point in front of the OpenGL camera must be in front of the OpenCV camera.
        np.testing.assert_allclose(R.T @ (np.array([1., 2., 1.]) - entry['position']), [0., 0., 2.])
        self.assertEqual(entry['img_name'], 'frame_000015')

    def test_centering_preserves_pinhole_rays(self):
        self.assertTrue(hasattr(self.adapter, 'center_image'), 'Image centering is absent')
        image = np.zeros((6, 8, 3), dtype=np.uint8)
        image[2, 3] = 255
        K = np.array([[10., 0., 3.], [0., 12., 2.], [0., 0., 1.]])
        centered, new_K = self.adapter.center_image(image, K)
        np.testing.assert_array_equal(centered[3, 4], [255, 255, 255])
        np.testing.assert_allclose(new_K[:2, 2], [4., 3.])
        np.testing.assert_allclose(np.linalg.inv(K) @ [3., 2., 1.], np.linalg.inv(new_K) @ [4., 3., 1.])

    def test_non_rigid_camera_is_rejected(self):
        self.assertTrue(hasattr(self.adapter, 'camera_entry'), 'Camera adapter is absent')
        with self.assertRaises(ValueError):
            self.adapter.camera_entry(np.c_[np.eye(3) * 2, [0., 0., 0.]], np.eye(3), 8, 6, 'frame.png', 0)

    def test_video_prefix_prevents_duplicate_frame_names(self):
        self.assertTrue(hasattr(self.adapter, 'image_name'), 'Nested image naming is absent')
        root = Path('/data/images')
        first = self.adapter.image_name(root / 'IMG_9763/frame_000000.png', root)
        second = self.adapter.image_name(root / 'IMG_9764/frame_000000.png', root)
        self.assertNotEqual(first, second)
        self.assertEqual(first, 'IMG_9763__frame_000000.png')
        self.assertEqual(self.adapter.image_name(root / 'frame_000000.png', root), 'frame_000000.png')


if __name__ == '__main__':
    unittest.main()
