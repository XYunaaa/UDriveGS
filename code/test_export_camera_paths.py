import importlib.util
from pathlib import Path
import unittest

import numpy as np
from scipy.spatial.transform import Rotation


class CameraPathTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).with_name('export_camera_paths.py')
        if path.exists():
            spec = importlib.util.spec_from_file_location('camera_paths', path)
            cls.module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(cls.module)
        else:
            cls.module = None

    def test_rotation_and_translation_scale_match_the_model(self):
        self.assertIsNotNone(self.module, 'Camera path exporter is absent')
        affine = np.c_[Rotation.from_euler('z', 90, degrees=True).as_matrix(), [2., 3., 4.]]
        rotation, center = self.module.transform_pose(np.eye(3), np.array([1., 0., 0.]), affine, .5)
        basis = Rotation.from_euler('x', 90, degrees=True).as_matrix()
        np.testing.assert_allclose(center, basis @ np.array([1., 2., 2.]), atol=1e-8)
        np.testing.assert_allclose(rotation, basis @ affine[:, :3], atol=1e-8)
        np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-8)

    def test_interpolation_preserves_duration_endpoints_and_rotation(self):
        self.assertIsNotNone(self.module, 'Camera path exporter is absent')
        poses = [{'time': t, 'position': [t, 0., 0.],
                  'rotation': Rotation.from_euler('z', t * 90, degrees=True).as_matrix().tolist(),
                  'width': 1920, 'height': 1080, 'fx': 1400., 'fy': 1400., 'img_name': f'frame{t}'}
                 for t in [0., 1.]]
        result = self.module.sample_track(poses, fps=2)
        self.assertEqual(len(result), 3)
        np.testing.assert_allclose(result[1]['position'], [.5, 0., 0.])
        np.testing.assert_allclose(result[1]['rotation'], Rotation.from_euler('z', 45, degrees=True).as_matrix())
        np.testing.assert_allclose(result[-1]['position'], poses[-1]['position'])
        self.assertEqual([r['id'] for r in result], [1, 2, 3])
        with self.assertRaises(ValueError):
            self.module.sample_track([poses[0], poses[0]])


if __name__ == '__main__':
    unittest.main()
