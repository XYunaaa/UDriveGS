import importlib.util
from pathlib import Path
import unittest


class CommonViewTests(unittest.TestCase):
    def test_compare_only_shared_localized_views_and_reject_changed_test_split(self):
        path = Path(__file__).with_name('compare_reconstructions.py')
        self.assertTrue(path.exists(), 'Common-view evaluation is absent')
        spec = importlib.util.spec_from_file_location('compare_reconstructions', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        frozen = {'a.png', 'b.png', 'c.png'}
        result = module.shared_test_views(frozen, {'a.png', 'b.png'}, frozen, {'b.png', 'c.png'})
        self.assertEqual(result, ['b.png'])
        with self.assertRaises(ValueError):
            module.shared_test_views(frozen, {'a.png'}, {'a.png'}, {'a.png'})
        with self.assertRaises(ValueError):
            module.shared_test_views(frozen, {'a.png'}, frozen, {'c.png'})
        with self.assertRaises(ValueError):
            module.shared_test_views(frozen, {'foreign.png'}, frozen, {'a.png'})


if __name__ == '__main__':
    unittest.main()
