"""Reject staged data/results and blobs larger than 5 MiB, without ML dependencies."""
from pathlib import PurePosixPath
import subprocess
import sys


def check():
    entries = subprocess.check_output(['git','ls-files','--stage','-z']).split(b'\0')
    blobs = []
    errors = []
    allowed = {'code','configs','docs','metadata','.githooks'}
    root_files = {'.gitignore','.gitattributes','README.md'}
    binary = {'.ckpt','.pth','.pt','.ply','.splat','.npz','.bin','.so','.o',
              '.mov','.mp4','.webm','.zip','.tar','.gz','.pdf','.png','.jpg','.jpeg','.pem','.key'}
    for entry in filter(None,entries):
        info, raw_name = entry.split(b'\t',1)
        mode, sha, stage = info.split()
        name = raw_name.decode('utf-8',errors='surrogateescape')
        path = PurePosixPath(name)
        if (path.parts[0] not in allowed and name not in root_files) or \
                path.suffix.lower() in binary or name.startswith('code/third_party/') or \
                any(part.startswith('.env') for part in path.parts) or mode == b'160000':
            errors.append(f'Excluded asset in index: {name}')
        if stage != b'0':
            errors.append(f'Unresolved merge: {name}')
        blobs.append((sha.decode(),name))
    if blobs:
        result = subprocess.run(['git','cat-file','--batch-check=%(objectsize)'],
            input=''.join(sha+'\n' for sha,_ in blobs),capture_output=True,text=True,check=True)
        sizes = [int(line) for line in result.stdout.splitlines()]
        for (_,name),size in zip(blobs,sizes):
            if size > 5*1024**2:
                errors.append(f'Blob exceeds 5 MiB: {name} ({size} bytes)')
        if sum(sizes) > 20*1024**2:
            errors.append('Tracked snapshot exceeds 20 MiB; keep transfer assets outside Git')
    # Deleted files remain in history and would still be uploaded by git push.
    history=subprocess.check_output(['git','rev-list','--objects','--all'],text=True).splitlines()
    if history:
        result=subprocess.run(['git','cat-file','--batch-check=%(objecttype) %(objectsize)'],
            input=''.join(line.split(' ',1)[0]+'\n' for line in history),
            capture_output=True,text=True,check=True)
        for item,line in zip(history,result.stdout.splitlines()):
            kind,size=line.split()
            if kind=='blob' and int(size)>5*1024**2:
                errors.append(f'Historical blob exceeds 5 MiB: {item}')
    if errors:
        print('\n'.join(errors),file=sys.stderr)
        return 1
    print(f'Git asset guard passed: {len(blobs)} tracked files')
    return 0


if __name__ == '__main__':
    sys.exit(check())
