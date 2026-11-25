"""CPU checks for calibrated training-only depth supervision."""

import unittest
import json
import hashlib
from pathlib import Path
import tempfile
from unittest.mock import patch

import numpy as np
import torch

from scene2_depth import (CalibratedDepthModel, CalibratedDepthConfig,
                          align_inverse_depth, depth_model_config, masked_inverse_depth_loss)
from nerfstudio.models.splatfacto import SplatfactoModel, SplatfactoModelConfig


class DepthTests(unittest.TestCase):
    def test_alignment_recovers_positive_scale_shift_despite_outliers(self):
        x = np.linspace(0.1, 3, 200)
        y = 0.7 * x + 0.3
        y[::7] += 20
        x[1], y[2], y[3] = np.nan, np.inf, -1
        scale, shift, valid, stats = align_inverse_depth(x, y)
        np.testing.assert_allclose([scale, shift], [0.7, 0.3], atol=1e-5)
        self.assertGreater(valid.sum(), 150)
        self.assertFalse(valid[:4].any())
        self.assertLess(stats['median_relative_error'], 1e-5)

    def test_alignment_rejects_low_support_flat_negative_and_bad_fit(self):
        rng = np.random.default_rng(42)
        for x, y in [(np.arange(10), np.ones(10)),
                     (np.ones(100), np.linspace(1, 2, 100)),
                     (np.linspace(1, 2, 100), np.linspace(2, 1, 100)),
                     (rng.random(200), rng.uniform(0.1, 10, 200))]:
            with self.subTest(), self.assertRaises(ValueError):
                align_inverse_depth(x, y)

    def test_loss_uses_only_valid_inverse_z_and_has_gradient(self):
        z = torch.tensor([[2., 4.], [float('nan'), 0.]], requires_grad=True)
        target = torch.tensor([[1., 0.25], [float('nan'), 3.]])
        mask = torch.tensor([[True, True], [False, False]])
        loss = masked_inverse_depth_loss(z, target, mask)
        self.assertAlmostEqual(loss.item(), 0.4, places=6)
        loss.backward()
        self.assertTrue(torch.isfinite(z.grad).all())
        self.assertNotEqual(z.grad[0, 0].item(), 0)
        self.assertTrue((z.grad[1] == 0).all())

    def test_no_valid_pixels_has_finite_zero_loss_and_zero_gradient(self):
        z = torch.tensor([float('nan'), 0., -2.], requires_grad=True)
        loss = masked_inverse_depth_loss(z, torch.ones(3), torch.zeros(3, dtype=torch.bool))
        self.assertEqual(loss.item(), 0)
        loss.backward()
        self.assertTrue((z.grad == 0).all())

    def test_config_refuses_incomplete_depth_preparation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'manifest.json').write_text(json.dumps({'status': 'preparing'}))
            with self.assertRaises(ValueError):
                depth_model_config(SplatfactoModelConfig(), root)

    def test_complete_config_preserves_base_settings_and_checks_target_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / '000000.npz'
            np.savez_compressed(target, inverse_z=np.ones((2, 2), dtype=np.float16),
                                valid=np.ones((2, 2), dtype=bool))
            manifest = {'status': 'complete', 'image_coverage': 1.,
                        'identity': {'training_images': ['images/train.png']},
                        'frames': {'0': {'status': 'accepted',
                                         'target_sha256': hashlib.sha256(target.read_bytes()).hexdigest()}}}
            (root / 'manifest.json').write_text(json.dumps(manifest))
            base = SplatfactoModelConfig(use_scale_regularization=True, max_gauss_ratio=8., cull_alpha_thresh=0.02)
            config = depth_model_config(base, root)
            for name, value in vars(base).items():
                if name not in {'_target', 'output_depth_during_training'}:
                    self.assertEqual(getattr(config, name), value, name)
            self.assertTrue(config.output_depth_during_training)
            self.assertIs(config._target, CalibratedDepthModel)
            target.write_bytes(b'changed')
            with self.assertRaises(ValueError):
                depth_model_config(base, root)

    def test_model_keeps_rgb_loss_and_only_supervises_training_after_warmup(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            np.savez_compressed(root / '000000.npz', inverse_z=np.ones((8, 8), dtype=np.float16),
                                valid=np.ones((8, 8), dtype=bool))
            model = object.__new__(CalibratedDepthModel)
            torch.nn.Module.__init__(model)
            model.config = CalibratedDepthConfig(depth_root=root)
            depth = torch.full((4, 4, 1), 2., requires_grad=True)
            main_loss = torch.tensor(3.)
            with patch.object(SplatfactoModel, 'get_loss_dict', side_effect=lambda *args: {'main_loss': main_loss}):
                model.step = 999
                self.assertNotIn('calibrated_depth_loss', model.get_loss_dict({'depth': depth}, {'image_idx': 0}))
                model.step = 1000
                losses = model.get_loss_dict({'depth': depth}, {'image_idx': 0})
                self.assertIs(losses['main_loss'], main_loss)
                self.assertAlmostEqual(losses['calibrated_depth_loss'].item(), 0.005)
                losses['calibrated_depth_loss'].backward()
                self.assertGreater(depth.grad.abs().sum().item(), 0)
                model.eval()
                self.assertNotIn('calibrated_depth_loss', model.get_loss_dict({'depth': depth}, {'image_idx': 0}))

    def test_missing_accepted_depth_is_error_not_silently_skipped(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'manifest.json').write_text(json.dumps({'frames': {'0': {'status': 'accepted'}}}))
            model = object.__new__(CalibratedDepthModel)
            torch.nn.Module.__init__(model)
            model.config = CalibratedDepthConfig(depth_root=root)
            model.step = 1000
            with patch.object(SplatfactoModel, 'get_loss_dict', return_value={}):
                with self.assertRaises(FileNotFoundError):
                    model.get_loss_dict({}, {'image_idx': 0})


if __name__ == '__main__':
    unittest.main()
