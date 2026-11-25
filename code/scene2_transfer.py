"""Pack, verify and restore the exact processed scene2 inputs without ML packages."""
import argparse
from contextlib import contextmanager
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import tempfile

PROJECT = Path(__file__).resolve().parents[1]
DATASET = Path('data/scene2_splatfacto/processed')


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024**2),b''): h.update(block)
    return h.hexdigest()


def safe_path(name):
    p = PurePosixPath(name)
    if not name or '\\' in name or p.is_absolute() or '..' in p.parts or str(p) != name:
        raise ValueError(f'Unsafe archive path: {name}')
    return p


def collect_inputs(project):
    dataset = project/DATASET
    meta = json.loads((dataset/'transforms.json').read_text())
    names = [name for split in ('train','val','test') for name in meta[f'{split}_filenames']]
    if len(names) != len(set(names)) or len(names) != 545:
        raise ValueError('Expected 545 distinct frozen train/val/test images')
    provenance = json.loads((dataset/'provenance.json').read_text())
    sparse = Path(provenance['source_training_model'])
    sparse = (sparse if sparse.is_absolute() else project/sparse).resolve()
    sparse_relative = sparse.relative_to(project.resolve())
    provenance['source_training_model'] = sparse_relative.as_posix()
    generated = {(DATASET/'provenance.json').as_posix():(json.dumps(provenance,indent=2)+'\n').encode()}
    paths = [DATASET/safe_path(name) for name in names]
    paths += [DATASET/'transforms.json',DATASET/'provenance.json',DATASET/safe_path(meta['ply_file_path'])]
    paths += [sparse_relative/f'{name}.bin' for name in ('cameras','images','points3D')]
    records = []
    for path in sorted(set(paths)):
        name = path.as_posix(); safe_path(name)
        source = project/path
        if not source.resolve().is_relative_to(project.resolve()) or not source.is_file():
            raise ValueError(f'Input outside project or missing: {name}')
        data = generated.get(name)
        record = {'path':name,'size':len(data) if data is not None else source.stat().st_size,
                  'sha256':hashlib.sha256(data).hexdigest() if data is not None else digest(source)}
        # shortcut: restored provenance has a new audit hash; repack on the original source host.
        if data is not None: record['original_source_sha256'] = digest(source)
        records.append(record)
    return records, generated


def pack_files(project, output, records, generated, chunk_bytes=512*1024**2):
    output.mkdir(parents=True,exist_ok=False)
    archive = output/'.inputs.tar.partial'
    with tarfile.open(archive,'w',dereference=True) as tar:
        for record in records:
            name = str(safe_path(record['path']))
            data = generated.get(name)
            source = project/name
            info = tarfile.TarInfo(name)
            info.size = record['size']; info.mode = 0o644; info.mtime = 0
            with io.BytesIO(data) if data is not None else source.open('rb') as stream:
                tar.addfile(info,stream)
    parts = []
    with archive.open('rb') as stream:
        index = 0
        while True:
            block = stream.read(min(chunk_bytes,1024**2))
            if not block: break
            part = output/f'scene2-inputs.tar.part-{index:03d}'
            h=hashlib.sha256(); size=0
            with part.open('xb') as target:
                while block:
                    target.write(block);h.update(block);size+=len(block)
                    if size == chunk_bytes: break
                    block=stream.read(min(chunk_bytes-size,1024**2))
            parts.append({'name':part.name,'size':size,'sha256':h.hexdigest()}); index+=1
    manifest={'schema':1,'scope':'exact processed RGB inputs; no videos or trained models',
              'files':records,'parts':parts,'raw_bytes':sum(r['size'] for r in records)}
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    archive.unlink()
    return manifest


@contextmanager
def checked_archive(bundle, expected=None):
    manifest=json.loads((bundle/'manifest.json').read_text())
    if expected is not None and expected.exists() and json.loads(expected.read_text()) != manifest:
        raise ValueError('Transfer manifest differs from the versioned input inventory')
    if manifest.get('schema') != 1 or not manifest['parts']:
        raise ValueError('Unsupported or empty transfer manifest')
    files={str(safe_path(r['path'])):r for r in manifest['files']}
    if len(files) != len(manifest['files']): raise ValueError('Duplicate input paths')
    seen_parts=set()
    with tempfile.TemporaryDirectory(prefix='udrive-verify-') as temporary:
        archive=Path(temporary)/'inputs.tar'
        with archive.open('xb') as joined:
            for part in manifest['parts']:
                name=str(safe_path(part['name']))
                if '/' in name or name in seen_parts: raise ValueError('Invalid/duplicate part name')
                seen_parts.add(name);source=bundle/name
                if source.stat().st_size != part['size'] or digest(source) != part['sha256']:
                    raise ValueError(f'Transfer part checksum/size mismatch: {name}')
                with source.open('rb') as stream: shutil.copyfileobj(stream,joined)
        with tarfile.open(archive,'r:') as tar:
            members=tar.getmembers()
            if len(members)!=len(files) or {m.name for m in members}!=set(files):
                raise ValueError('Archive file set differs from manifest')
            for member in members:
                if not member.isfile() or member.size != files[member.name]['size']:
                    raise ValueError(f'Invalid archive member: {member.name}')
                h=hashlib.sha256()
                with tar.extractfile(member) as stream:
                    for block in iter(lambda:stream.read(1024**2),b''):h.update(block)
                if h.hexdigest()!=files[member.name]['sha256']:
                    raise ValueError(f'Input checksum mismatch: {member.name}')
            yield tar,manifest


def verify(bundle, expected=None):
    with checked_archive(bundle,expected) as (_,manifest): return manifest


def unpack(bundle, project):
    project=project.resolve()
    with checked_archive(bundle,project/'metadata/scene2/inputs.json') as (tar,manifest):
        for record in manifest['files']:
            target=project/record['path']
            if not target.resolve().is_relative_to(project): raise ValueError(f'Path escapes checkout: {target}')
            if any(p.is_symlink() for p in [target,*target.parents] if p.is_relative_to(project)):
                raise ValueError(f'Symlink destination refused: {target}')
            if any(p.exists() and not p.is_dir() for p in target.parents if p.is_relative_to(project)):
                raise ValueError(f'Destination parent conflict; no files overwritten: {target}')
            if target.exists() and (not target.is_file() or digest(target)!=record['sha256']):
                raise ValueError(f'Destination conflict; no files overwritten: {target}')
        for record in manifest['files']:
            target=project/record['path']
            if target.exists():continue
            target.parent.mkdir(parents=True,exist_ok=True)
            with tar.extractfile(record['path']) as stream, \
                    tempfile.NamedTemporaryFile(dir=target.parent,prefix='.restore-',delete=False) as output:
                shutil.copyfileobj(stream,output);temporary=Path(output.name)
            temporary.replace(target)
        return manifest


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['pack','verify','unpack'])
    parser.add_argument('--project',type=Path,default=PROJECT)
    parser.add_argument('--bundle',type=Path,required=True)
    args=parser.parse_args()
    if args.action=='pack':
        records,generated=collect_inputs(args.project.resolve())
        result=pack_files(args.project.resolve(),args.bundle,records,generated)
    elif args.action=='verify':result=verify(args.bundle,args.project/'metadata/scene2/inputs.json')
    else:result=unpack(args.bundle,args.project)
    print(json.dumps({'files':len(result['files']),'raw_bytes':result['raw_bytes'],
                      'parts':len(result['parts']),'status':'verified' if args.action!='pack' else 'packed'}))
