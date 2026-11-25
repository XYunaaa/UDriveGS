"""Extract fixed scene1 candidates from real video timestamps, without overwriting data."""

import argparse
from bisect import bisect_left
import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess
import tempfile

from PIL import Image, ImageDraw


def sample_indices(timestamps, interval_seconds=0.5):
    if not math.isfinite(interval_seconds) or interval_seconds <= 0:
        raise ValueError('Sampling interval must be finite and positive')
    if not timestamps or any(not math.isfinite(t) for t in timestamps):
        raise ValueError('Frame timestamps must be nonempty and finite')
    if any(b <= a for a, b in zip(timestamps, timestamps[1:])):
        raise ValueError('Frame timestamps must be strictly increasing')
    indices = []
    duration = timestamps[-1] - timestamps[0]
    for step in range(math.floor(duration / interval_seconds) + 1):
        target = timestamps[0] + step * interval_seconds
        right = bisect_left(timestamps, target)
        options = [i for i in (right - 1, right) if 0 <= i < len(timestamps)]
        index = min(options, key=lambda i: (abs(timestamps[i] - target), i))
        if not indices or index != indices[-1]:
            indices.append(index)
    return indices


def split_for_time(timestamp):
    block = math.floor(timestamp / 2) % 10
    return 'val' if block == 8 else 'test' if block == 9 else 'train'


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def prepare(video, output, interval_seconds=0.5):
    if not video.is_file():
        raise FileNotFoundError(video)
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite {output}')
    metadata = json.loads(subprocess.check_output([
        'ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_frames',
        '-show_streams', '-show_format', '-show_entries',
        'frame=best_effort_timestamp_time:stream=codec_name,width,height,avg_frame_rate,nb_frames:format=duration,size',
        '-of', 'json', str(video),
    ]))
    timestamps = [float(frame['best_effort_timestamp_time']) for frame in metadata.pop('frames')]
    selected = sample_indices(timestamps, interval_seconds=interval_seconds)
    rows = []
    for index in selected:
        relative = timestamps[index] - timestamps[0]
        rows.append({
            'image': f'frame_{index:06d}.png', 'source_index': index,
            'pts_seconds': timestamps[index], 'relative_seconds': relative,
            'split': split_for_time(relative), 'selection': 'candidate',
        })
    digest = hashlib.sha256()
    with video.open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f'.{output.name}-', dir=output.parent) as temporary:
        stage = Path(temporary) / 'dataset'
        images = stage / 'images'
        images.mkdir(parents=True)
        expression = '+'.join(f'eq(n,{i})' for i in selected)
        command = [
            'ffmpeg', '-hide_banner', '-loglevel', 'error', '-i', str(video),
            '-map', '0:v:0', '-vf', f'select={expression}', '-vsync', '0',
            '-compression_level', '3', str(images / 'selected_%06d.png'),
        ]
        # The filter parser uses commas as separators; quote each select expression.
        command[command.index('-vf') + 1] = f"select='{expression}'"
        subprocess.run(command, check=True)
        extracted = sorted(images.glob('selected_*.png'))
        if len(extracted) != len(rows):
            raise RuntimeError(f'Expected {len(rows)} frames, decoded {len(extracted)}')
        width, height = metadata['streams'][0]['width'], metadata['streams'][0]['height']
        for path, row in zip(extracted, rows):
            with Image.open(path) as frame:
                if frame.size != (width, height):
                    raise ValueError(f'Unexpected image size: {path}')
                frame.verify()
            path.rename(images / row['image'])
        with (stage / 'frames.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        splits = {name: [row['image'] for row in rows if row['split'] == name]
                  for name in ('train', 'val', 'test')}
        all_names = [name for names in splits.values() for name in names]
        if len(set(all_names)) != len(rows):
            raise ValueError('Duplicate or missing split assignment')
        write_json(stage / 'splits.json', splits)
        smoke = [row['image'] for row in rows if row['split'] == 'train' and row['relative_seconds'] < 30]
        write_json(stage / 'smoke_train.json', smoke)
        for name, names in {**splits, 'smoke_train': smoke}.items():
            (stage / f'{name}.txt').write_text('\n'.join(names) + '\n')
        contacts = stage / 'quality'
        contacts.mkdir()
        training = [row for row in rows if row['split'] == 'train']
        for start in range(0, len(training), 48):
            page = training[start:start + 48]
            sheet = Image.new('RGB', (1440, 8 * 160), 'white')
            draw = ImageDraw.Draw(sheet)
            for tile, row in enumerate(page):
                x, y = (tile % 6) * 240, (tile // 6) * 160
                with Image.open(images / row['image']) as frame:
                    frame.thumbnail((240, 136))
                    sheet.paste(frame.convert('RGB'), (x, y))
                draw.text((x + 3, y + 137), f"{row['image']} {row['relative_seconds']:.2f}s", fill='black')
            sheet.save(contacts / f'train_{start // 48 + 1:02d}.jpg', quality=90)
        write_json(stage / 'manifest.json', {
            'status': 'candidates_extracted_quality_review_pending',
            'source': str(video.resolve()), 'sha256': digest.hexdigest(),
            'metadata': metadata, 'source_frame_count': len(timestamps),
            'sampling': {'interval_seconds': interval_seconds, 'selection': 'nearest source PTS; earlier frame wins ties'},
            'split_rule': 'floor(relative_seconds / 2) % 10: 8=val, 9=test, otherwise train',
            'counts': {name: len(names) for name, names in splits.items()},
            'smoke_train_count': len(smoke), 'source_index_base': 0,
            'ffmpeg_version': subprocess.check_output(['ffmpeg', '-version'], text=True).splitlines()[0],
            'extraction_command': command,
        })
        stage.rename(output)
    return {'output': str(output), 'candidates': len(rows),
            'splits': {name: len(names) for name, names in splits.items()}, 'smoke_train': len(smoke)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video', type=Path, default=Path('data/scene1'))
    parser.add_argument('--output', type=Path, default=Path('data/scene1_splatfacto'))
    parser.add_argument('--interval-seconds', type=float, default=0.5)
    args = parser.parse_args()
    print(json.dumps(prepare(args.video, args.output, args.interval_seconds), ensure_ascii=False, indent=2))
