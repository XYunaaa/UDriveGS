import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np


class FrozenSceneTests(unittest.TestCase):
    def test_heldout_localization_may_only_add_camera_poses(self):
        path = Path(__file__).with_name('finish_scene2.py')
        self.assertTrue(path.exists(), 'Scene2 localization/conversion is absent')
        spec = importlib.util.spec_from_file_location('finish_scene2', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        def model():
            return ({1: SimpleNamespace(params=np.array([100., 32., 24., 0.]))},
                    {1: SimpleNamespace(name='train.png', qvec=np.array([1., 0., 0., 0.]), tvec=np.zeros(3))},
                    {1: SimpleNamespace(xyz=np.ones(3), rgb=np.array([20, 30, 40]))})
        before, after = model(), model()
        after[1][2] = SimpleNamespace(name='test.png')
        module.assert_frozen(before, after)
        for group, field in [(0, 'params'), (1, 'qvec'), (1, 'tvec'), (2, 'xyz'), (2, 'rgb')]:
            after = model()
            getattr(after[group][1], field)[0] += 1
            with self.subTest(field=field), self.assertRaises(AssertionError):
                module.assert_frozen(before, after)
        after = model()
        after[2][2] = after[2][1]
        with self.assertRaises(AssertionError):
            module.assert_frozen(before, after)


if __name__ == '__main__':
    unittest.main()
