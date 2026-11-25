import importlib.util
from pathlib import Path
import unittest

import torch


class CheckpointExportTests(unittest.TestCase):
    def test_checkpoint_parameters_keep_original_scale_opacity_and_sh_order(self):
        script = Path(__file__).with_name('export_checkpoint_cpu.py')
        self.assertTrue(script.exists(), 'CPU checkpoint exporter is not implemented')
        spec = importlib.util.spec_from_file_location('cpu_export', script)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        params = {
            'means': torch.tensor([[1., 2., 3.], [4., 5., 6.]]),
            'features_dc': torch.ones(2, 3),
            'features_rest': torch.arange(90.).reshape(2, 15, 3),
            'opacities': torch.tensor([[0.], [-10.]]),
            'scales': torch.tensor([[0., 1., 2.], [0., 0., 0.]]),
            'quats': torch.tensor([[1., 0., 0., 0.], [1., 0., 0., 0.]]),
        }
        fields = module.ply_attributes(params)
        self.assertEqual(len(fields['x']), 1)
        self.assertEqual(fields['x'][0], 1.)
        self.assertEqual(fields['scale_1'][0], 1.)
        self.assertEqual(fields['opacity'][0], 0.)
        self.assertEqual(fields['f_rest_1'][0], 3.)
        self.assertEqual(fields['rot_0'][0], 1.)
        self.assertEqual(len(module.splat_bytes(fields)), 32)


if __name__ == '__main__':
    unittest.main()
