"""Exercise the launcher's real mesh export block without a GPU."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import open3d as o3d


class SugarMeshExportTests(unittest.TestCase):
    def test_colored_mesh_exports_and_reloads_as_glb(self):
        self.check_export('scene1', 'segment_a_mesh.glb')

    def test_scene2_mesh_has_its_own_output_name(self):
        self.check_export('scene2', 'scene2_mesh.glb')

    def check_export(self, scene, output_name):
        script = Path(__file__).with_name('run_scene1_sugar_host.sh').read_text()
        marker = '"$env_prefix/bin/python" - "$mesh_dir" "$result_dir" <<\'PY\'\n'
        block = script.split(marker, 1)[1].split('\nPY\n', 1)[0]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            mesh_dir = root / 'mesh'
            mesh_dir.mkdir()
            (root / 'input').mkdir()
            original = root / 'original.ply'
            original.write_bytes(b'Original Gaussian remains untouched')
            (root / 'input/manifest.json').write_text(json.dumps({
                'scene': scene, 'source_ply': str(original), 'source_ply_sha256': hashlib.sha256(original.read_bytes()).hexdigest()}))
            mesh = o3d.geometry.TriangleMesh.create_box()
            mesh.paint_uniform_color([.2, .4, .6])
            self.assertTrue(o3d.io.write_triangle_mesh(str(mesh_dir / 'sample.ply'), mesh))
            result = subprocess.run([sys.executable, '-', str(mesh_dir), str(root)], input=block,
                                    text=True, capture_output=True,
                                    env={**os.environ, 'OMP_NUM_THREADS': '8', 'OPENBLAS_NUM_THREADS': '8'})
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            summary = json.loads((root / 'mesh_summary.json').read_text())
            self.assertEqual(summary['triangles'], 12)
            self.assertTrue(summary['original_ply_unchanged'])
            self.assertTrue((mesh_dir / output_name).is_file())


if __name__ == '__main__':
    unittest.main()
