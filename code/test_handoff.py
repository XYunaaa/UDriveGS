"""Cross-machine handoff checks; no GPU or training required."""
import hashlib
import importlib
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch


class HandoffTests(unittest.TestCase):
    def test_relative_config_paths_resolve_inside_checkout(self):
        import scene2_quality as quality
        cfg = NS(data=Path('data/scene2'), output_dir=Path('results/runs'), load_dir=None,
                 pipeline=NS(datamanager=NS(data=Path('data/scene2'),
                     dataparser=NS(data=Path('data/scene2')))))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            p = root/'config.yml'; p.write_text('local trusted config')
            with patch.object(quality, 'PROJECT', root), patch('yaml.load', return_value=cfg):
                actual = quality.load_config(p)
            self.assertEqual(actual.data, root/'data/scene2')
            self.assertEqual(actual.pipeline.datamanager.dataparser.data, actual.data)
            self.assertEqual(actual.output_dir, root/'results/runs')

    def test_rgb_inputs_do_not_require_historical_checkpoint_or_ply(self):
        import scene2_quality as quality
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); dataset = root/'dataset'; dataset.mkdir()
            meta = {}
            for split, count in [('train',474),('val',41),('test',30)]:
                names = [f'{split}_{i}.png' for i in range(count)]
                for name in names: (dataset/name).write_bytes(b'fixture')
                meta[f'{split}_filenames'] = names
            (dataset/'transforms.json').write_text(json.dumps(meta))
            (dataset/'sparse_pc.ply').write_bytes(b'frozen sparse points')
            meta['ply_file_path']='sparse_pc.ply'
            (dataset/'transforms.json').write_text(json.dumps(meta))
            cfg = root/'config.yml'; cfg.write_text('portable config')
            config = NS(pipeline=NS(datamanager=NS(data=dataset)))
            with patch.object(quality,'BASE_CONFIG',cfg), patch.object(quality,'load_config',return_value=config), \
                    patch.object(quality,'checkpoint_path',side_effect=AssertionError('must not load baseline')):
                result = quality.initialize(root/'run', include_baseline=False)
            self.assertNotIn('base_checkpoint', result['fingerprint'])
            self.assertEqual(result['split_counts'],{'train':474,'val':41,'test':30})
            self.assertEqual(result['fingerprint']['sparse_initialization'],
                             hashlib.sha256(b'frozen sparse points').hexdigest())

    def test_transfer_roundtrip_corruption_and_conflicting_destination(self):
        transfer = importlib.import_module('scene2_transfer')
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); src = root/'source'; src.mkdir()
            (src/'data').mkdir(); (src/'data/image.png').write_bytes(bytes(range(256))*20)
            records = [{'path':'data/image.png','size':5120,'sha256':hashlib.sha256((src/'data/image.png').read_bytes()).hexdigest()}]
            bundle = root/'bundle'
            transfer.pack_files(src, bundle, records, {}, chunk_bytes=4096)
            transfer.verify(bundle)
            dest = root/'different absolute path'; dest.mkdir()
            transfer.unpack(bundle,dest)
            self.assertEqual((dest/'data/image.png').read_bytes(),(src/'data/image.png').read_bytes())
            transfer.unpack(bundle,dest)  # Identical files can safely be reused.
            (dest/'data/image.png').write_bytes(b'user data')
            with self.assertRaisesRegex(ValueError,'conflict'):
                transfer.unpack(bundle,dest)
            self.assertEqual((dest/'data/image.png').read_bytes(),b'user data')
            first = sorted(bundle.glob('*.part-*'))[0]
            first.write_bytes(b'corrupt')
            with self.assertRaises(ValueError): transfer.verify(bundle)

    def test_transfer_rejects_escape_paths(self):
        transfer = importlib.import_module('scene2_transfer')
        for name in ['../outside','/absolute','data/../../outside','data\\outside']:
            with self.assertRaises(ValueError): transfer.safe_path(name)

    def test_transfer_detects_blocked_parent_before_writing_any_file(self):
        transfer=importlib.import_module('scene2_transfer')
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);src=root/'src';src.mkdir();records=[]
            for name in ['data/image.png','blocked/image.png']:
                p=src/name;p.parent.mkdir();p.write_bytes(b'fixture')
                records.append({'path':name,'size':7,'sha256':hashlib.sha256(b'fixture').hexdigest()})
            bundle=root/'bundle';transfer.pack_files(src,bundle,records,{})
            dest=root/'dest';dest.mkdir();(dest/'blocked').write_text('user file')
            with self.assertRaisesRegex(ValueError,'conflict'):transfer.unpack(bundle,dest)
            self.assertFalse((dest/'data/image.png').exists())
            self.assertEqual((dest/'blocked').read_text(),'user file')

    def test_git_guard_checks_staged_blobs_even_when_worktree_is_small(self):
        guard = Path(__file__).with_name('check_git_size.py').resolve()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            subprocess.run(['git','init','-q',str(root)],check=True)
            (root/'docs').mkdir()
            p=root/'docs/large.txt'; p.write_bytes(b'x'*(5*1024**2+1))
            subprocess.run(['git','-C',str(root),'add','docs/large.txt'],check=True)
            p.write_text('small working tree file')
            result=subprocess.run(['python3',str(guard)],cwd=root,capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0)
            self.assertIn('large.txt',result.stdout+result.stderr)
            subprocess.run(['git','-c','user.name=Test','-c','user.email=test@example.invalid',
                            'commit','-qm','large historical blob'],cwd=root,check=True)
            subprocess.run(['git','rm','--cached','docs/large.txt'],cwd=root,check=True,capture_output=True)
            result=subprocess.run(['python3',str(guard)],cwd=root,capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0,'Deleted historical blobs still travel in Git pushes')
            self.assertIn('large.txt',result.stdout+result.stderr)

    def test_git_guard_rejects_forced_small_data_files(self):
        guard=Path(__file__).with_name('check_git_size.py').resolve()
        for name in ['data/source.png','code/.env.production','code/test-private.pem','configs/server.key']:
            with self.subTest(name=name),tempfile.TemporaryDirectory() as temporary:
                root=Path(temporary);subprocess.run(['git','init','-q',str(root)],check=True)
                p=root/name;p.parent.mkdir();p.write_bytes(b'fake sensitive fixture')
                subprocess.run(['git','add','-f',name],cwd=root,check=True)
                result=subprocess.run(['python3',str(guard)],cwd=root,capture_output=True,text=True)
                self.assertNotEqual(result.returncode,0)
                self.assertIn(name,result.stdout+result.stderr)


if __name__ == '__main__':
    unittest.main()
