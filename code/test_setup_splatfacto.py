"""Check host setup's environment boundary without downloads or a GPU."""

import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


class SetupTests(unittest.TestCase):
    def test_setup_selects_own_cuda_and_checks_gsplat(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            code = root / 'code'
            code.mkdir()
            shutil.copy(Path(__file__).with_name('setup_splatfacto.sh'), code)
            shutil.copy(Path(__file__).with_name('run_splatfacto_smoke.sh'), code)
            shutil.copy(Path(__file__).with_name('splatfacto-constraints.txt'), code)
            prefix = root / '.envs/splatfacto'
            (prefix / 'conda-meta').mkdir(parents=True)
            bins = prefix / 'bin'
            bins.mkdir()
            fake_bins = root / 'fake-bin'
            fake_bins.mkdir()
            commands = {
                fake_bins / 'nvidia-smi': '#!/bin/sh\nexit 0\n',
                fake_bins / 'conda': '#!/bin/sh\nexit 0\n',
                bins / 'colmap': '#!/bin/sh\nexit 0\n',
                bins / 'nvcc': '#!/bin/sh\necho "release 12.4"\n',
                bins / 'ns-train': '#!/bin/sh\nexit 0\n',
                bins / 'python': '''#!/bin/sh
if [ "$1" = "-" ]; then
    printf '%s\\n%s\\n%s\\n%s\\n%s\\n%s\\n%s\\n%s\\n' "${CUDA_HOME:-}" "$PATH" "${PYTHONNOUSERSITE:-}" "${PYTHONPATH:-}" "${CC:-}" "${CXX:-}" "${TORCH_CUDA_ARCH_LIST:-}" "${TORCH_EXTENSIONS_DIR:-}" > "$SETUP_TEST_ROOT/environment.txt"
    cat > "$SETUP_TEST_ROOT/check.py"
fi
''',
            }
            for path, content in commands.items():
                path.write_text(content)
                path.chmod(0o755)
            env = dict(os.environ, PATH=str(fake_bins) + ':/usr/bin:/bin',
                       CONDA_EXE=str(fake_bins / 'conda'), SETUP_TEST_ROOT=str(root),
                       CUDA_HOME='/wrong/host/toolkit', PYTHONPATH='/external/project',
                       CC='/wrong/gcc-15', CXX='/wrong/g++-15', TORCH_CUDA_ARCH_LIST='9.0')
            subprocess.run(['bash', str(code / 'setup_splatfacto.sh')], env=env,
                           check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            lines = (root / 'environment.txt').read_text().splitlines()
            self.assertEqual(lines[0], str(prefix))
            self.assertEqual(lines[1].split(':')[0], str(bins))
            self.assertEqual(lines[2], '1')
            self.assertEqual(lines[3], '')
            self.assertEqual(lines[4], '/usr/bin/gcc-12')
            self.assertEqual(lines[5], '/usr/bin/g++-12')
            self.assertEqual(lines[6], '9.0')
            self.assertEqual(lines[7], str(root / '.cache/torch_extensions_gcc12'))
            check = (root / 'check.py').read_text()
            self.assertIn('rasterization', check)
            self.assertIn('.backward()', check)
            subprocess.run(['bash', str(code / 'run_splatfacto_smoke.sh')], env=env,
                           check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            training_lines = (root / 'environment.txt').read_text().splitlines()
            self.assertEqual(training_lines, lines)


if __name__ == '__main__':
    unittest.main()
