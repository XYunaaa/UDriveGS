import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import numpy as np
from plyfile import PlyData, PlyElement


class QualityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).with_name('scene2_quality.py')
        cls.module = None
        if path.exists():
            spec = importlib.util.spec_from_file_location('scene2_quality', path)
            cls.module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(cls.module)

    def test_cleaning_preserves_retained_fields_and_original(self):
        self.assertIsNotNone(self.module, 'Quality runner is absent')
        data = np.zeros(3, dtype=[(name, 'f4') for name in ['x','y','z','opacity','f_dc_0','f_rest_0','scale_0','rot_0']])
        data['opacity'] = np.log(np.array([.005,.02,.5]) / (1-np.array([.005,.02,.5])))
        data['f_rest_0'] = [3, 8, 17]
        data['rot_0'] = 1
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, output = root/'source.ply', root/'clean.ply'
            PlyData([PlyElement.describe(data, 'vertex')], comments=['Vertical Axis: z']).write(source)
            original = source.read_bytes()
            result = self.module.clean_ply(source, output, .02)
            cleaned = PlyData.read(output)
            np.testing.assert_array_equal(cleaned['vertex'].data, data[1:])
            self.assertEqual(cleaned.comments, ['Vertical Axis: z'])
            self.assertEqual(source.read_bytes(), original)
            self.assertEqual(result['retained'], 2)
            with self.assertRaises(FileExistsError):
                self.module.clean_ply(source, output, .02)
            with self.assertRaises(ValueError):
                self.module.clean_ply(source, root/'bad.ply', float('nan'))

    def test_three_ablations_change_exactly_one_setting(self):
        self.assertIsNotNone(self.module, 'Quality runner is absent')
        model = NS(camera_optimizer=NS(mode='off'), use_scale_regularization=False,
                   max_gauss_ratio=10., cull_alpha_thresh=.005, densify_grad_thresh=.0005)
        base = NS(pipeline=NS(model=model), machine=NS(seed=42), max_num_iterations=30000)
        expected = {'pose': ('SO3xR3',False,.005), 'scale': ('off',True,.005), 'prune': ('off',False,.02)}
        for label, values in expected.items():
            cfg = self.module.make_ablation_config(base, label)
            m = cfg.pipeline.model
            self.assertEqual((m.camera_optimizer.mode,m.use_scale_regularization,m.cull_alpha_thresh),values)
            self.assertEqual(m.max_gauss_ratio,10.)
            self.assertEqual(m.densify_grad_thresh,.0005)
            self.assertEqual(cfg.machine.seed,42)
            self.assertEqual(cfg.max_num_iterations,30000)
        self.assertEqual(model.camera_optimizer.mode,'off')

    def test_selection_requires_complete_matching_validation_and_guards_quality(self):
        self.assertIsNotNone(self.module, 'Quality runner is absent')
        def record(lpips, psnr=23., ssim=.87):
            return {'split':'val','filenames':['a','b'], 'metrics':{'lpips':lpips,'psnr':psnr,'ssim':ssim}}
        rows={'baseline':record(.2), 'pose':record(.18), 'scale':record(.15,22.), 'prune':record(.199)}
        self.assertEqual(self.module.choose_on_validation(rows,['baseline','pose','scale','prune']),'pose')
        with self.assertRaises(ValueError):
            self.module.choose_on_validation(rows,['baseline','depth'])
        rows['pose']['split']='test'
        with self.assertRaises(ValueError):
            self.module.choose_on_validation(rows,['baseline','pose'])
        rows['pose']=record(.18)
        rows['pose']['filenames']=['a','foreign']
        with self.assertRaises(ValueError):
            self.module.choose_on_validation(rows,['baseline','pose'])

    def test_compact_checkpoint_preserves_pipeline_and_step(self):
        self.assertIsNotNone(self.module, 'Quality runner is absent')
        import torch
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'step-000029999.ckpt'
            torch.save({'step':29999,'pipeline':{'x':torch.tensor([1.,2.])},
                        'optimizers':{'x':torch.ones(100)},'schedulers':{},'scalers':{}},path)
            self.module.compact_checkpoint(path)
            actual=torch.load(path,map_location='cpu',weights_only=False)
            torch.testing.assert_close(actual['pipeline']['x'],torch.tensor([1.,2.]))
            self.assertEqual(actual['step'],29999)
            self.assertTrue(actual['inference_only'])
            self.assertNotIn('optimizers',actual)

    def test_partial_export_is_regenerated_and_bound_to_checkpoint(self):
        self.assertIsNotNone(self.module, 'Quality runner is absent')
        import torch
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            checkpoint=root/'step-000029999.ckpt'
            fields={'means':torch.tensor([[1.,2.,3.],[4.,5.,6.]]),
                    'features_dc':torch.zeros(2,3),'features_rest':torch.zeros(2,15,3),
                    'opacities':torch.zeros(2,1),'scales':torch.zeros(2,3),
                    'quats':torch.tensor([[1.,0.,0.,0.],[1.,0.,0.,0.]])}
            torch.save({'step':29999,'pipeline':{'_model.gauss_params.'+k:v for k,v in fields.items()}},checkpoint)
            output=root/'splat.ply'
            output.write_bytes(b'interrupted export')
            with patch.object(self.module,'checkpoint_path',return_value=checkpoint):
                self.module.export_checkpoint_ply(root/'config.yml',output)
                self.assertTrue(output.read_bytes().startswith(b'ply\n'),'Partial export was silently reused')
                self.assertEqual(len(PlyData.read(output)['vertex']),2)
                expected=output.read_bytes()
                output.write_bytes(expected[:100])
                self.module.export_checkpoint_ply(root/'config.yml',output)
                self.assertEqual(output.read_bytes(),expected)

    def test_busy_gpu_waits_without_starting_or_stopping_other_jobs(self):
        self.assertIsNotNone(self.module, 'Quality runner is absent')
        self.assertTrue(hasattr(self.module,'wait_for_gpu'),'Stage-specific GPU waiting is absent')
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            with patch('torch.cuda.mem_get_info',side_effect=[(7*1024**3,24*1024**3),(14*1024**3,24*1024**3)]), \
                    patch.object(self.module.time,'sleep') as sleep:
                self.module.wait_for_gpu(root,12,'train_pose')
            sleep.assert_called_once_with(30)
            status=json.loads((root/'status.json').read_text())
            self.assertEqual(status['status'],'gpu_memory_ready')
            self.assertEqual(status['stage'],'train_pose')


if __name__=='__main__':
    unittest.main()
