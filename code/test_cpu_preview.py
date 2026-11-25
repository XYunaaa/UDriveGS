import importlib.util
from pathlib import Path
import unittest

import numpy as np


class CpuPreviewTests(unittest.TestCase):
    def test_front_to_back_compositing_and_empty_background(self):
        path = Path(__file__).with_name('render_cpu_preview.py')
        self.assertTrue(path.exists(), 'CPU reference compositor is not implemented')
        spec = importlib.util.spec_from_file_location('cpu_preview', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        rgb, alpha = module.compose(
            np.array([[.5, .5], [.5, .5]]), np.array([[1., 0., 1.], [1., 0., 1.]]),
            np.array([1., 1.]), np.array([1., 2.]), np.array([.5, .5]),
            np.array([[1., 0., 0.], [0., 0., 1.]]), 1, 1, np.zeros(3))
        np.testing.assert_allclose(rgb[0, 0], [.5, 0, .25])
        np.testing.assert_allclose(alpha[0, 0], .75)
        rgb, alpha = module.compose(
            np.empty((0, 2)), np.empty((0, 3)), np.array([]), np.array([]),
            np.array([]), np.empty((0, 3)), 1, 1, np.array([.1, .2, .3]))
        np.testing.assert_allclose(rgb[0, 0], [.1, .2, .3])
        self.assertEqual(alpha[0, 0], 0.)


if __name__ == '__main__':
    unittest.main()
