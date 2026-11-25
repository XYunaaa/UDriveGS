import importlib.util
from pathlib import Path
import unittest


class LocalizationPairsTests(unittest.TestCase):
    def test_only_temporal_neighbors_inside_the_frozen_map_span(self):
        path = Path(__file__).with_name('localize_heldout.py')
        self.assertTrue(path.exists(), 'Held-out localization is not implemented')
        spec = importlib.util.spec_from_file_location('localize', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        pairs = module.temporal_pairs(['eval5', 'eval12'], ['train0', 'train10'],
                                      {'eval5': 5., 'eval12': 12., 'train0': 0., 'train10': 10.})
        self.assertEqual(set(pairs), {('eval5', 'train0'), ('eval5', 'train10')})


if __name__ == '__main__':
    unittest.main()
