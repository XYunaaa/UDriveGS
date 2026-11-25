"""Prepare scene2 with full-video holdouts and cross-video training matches."""

import argparse
import csv
import hashlib
from itertools import combinations
import json
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np

from prepare_scene import prepare, sample_indices, write_json


VALIDATION_VIDEO = 'IMG_9765.MOV'
TEST_VIDEO = 'IMG_9780.MOV'


def matching_pairs(rows, temporal_seconds=None, anchor_seconds=None):
    clips = {}
    for row in rows:
        if row['split'] == 'train':
            clips.setdefault(row['video'], []).append(row)
    pairs = set()
    anchors = []
    for clip, frames in sorted(clips.items()):
        frames = sorted(frames, key=lambda r: float(r['relative_seconds']))
        for i, row in enumerate(frames):
            neighbors = frames[i + 1:i + 9] if temporal_seconds is None else frames[i + 1:]
            for other in neighbors:
                if temporal_seconds is not None and float(other['relative_seconds']) - float(row['relative_seconds']) > temporal_seconds:
                    break
                pairs.add(tuple(sorted((row['image'], other['image']))))
        indices = (range(0, len(frames), 4) if anchor_seconds is None else
                   sample_indices([float(r['relative_seconds']) for r in frames], interval_seconds=anchor_seconds))
        anchors.extend((clip, frames[i]['image']) for i in indices)
    for (clip_a, name_a), (clip_b, name_b) in combinations(anchors, 2):
        if clip_a != clip_b:
            pairs.add(tuple(sorted((name_a, name_b))))
    return sorted(pairs)


def sharpness_score(path):
    import cv2

    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f'Cannot decode frame for sharpness scoring: {path}')
    if image.shape[1] > 960:
        image = cv2.resize(image, (960, round(image.shape[0] * 960 / image.shape[1])), interpolation=cv2.INTER_AREA)
    return float(cv2.Laplacian(image, cv2.CV_64F).var())


def sharpness_selection(times, scores):
    times, scores = np.asarray(times, dtype=float), np.asarray(scores, dtype=float)
    if times.ndim != 1 or times.shape != scores.shape or not len(times) or not np.isfinite(times).all() or not np.isfinite(scores).all():
        raise ValueError('Expected matching, nonempty finite timestamp and sharpness arrays')
    if np.any(np.diff(times) <= 0) or np.any(scores < 0):
        raise ValueError('Timestamps must increase and sharpness must be nonnegative')
    medians = [float(np.median(scores[np.abs(times - t) <= 1.])) for t in times]
    result = [{'score': float(score), 'local_median': median, 'threshold': .5 * median,
               'keep': bool(score >= .5 * median),
               'decision': 'kept' if score >= .5 * median else 'rejected_blur'}
              for score, median in zip(scores, medians)]
    # shortcut: local sharpness is a heuristic; retain a best frame per0.5sec slot until a validated blur model is needed.
    slots = np.floor((times - times[0] + .125) / .5).astype(int)
    for slot in np.unique(slots):
        indices = np.flatnonzero(slots == slot)
        if not any(result[i]['keep'] for i in indices):
            best = max(indices, key=lambda i: (scores[i], -i))
            result[best].update(keep=True, decision='coverage_fallback')
    return result


def prepare_many(source, output, fps=2, filter_sharpness=False, baseline=None):
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite {output}')
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError('FPS must be finite and positive')
    if (fps != 2 or filter_sharpness) and baseline is None:
        raise ValueError('A baseline is required to freeze held-out image bytes for this variant')
    videos = sorted(p for p in source.iterdir() if p.is_file() and p.suffix.lower() == '.mov')
    names = {p.name for p in videos}
    if not {VALIDATION_VIDEO, TEST_VIDEO} < names:
        raise ValueError('Both frozen held-out clips and at least one training clip are required')
    if len({p.stem for p in videos}) != len(videos):
        raise ValueError('Duplicate video stems would collide')
    baseline_rows, baseline_sources = {}, {}
    if baseline is not None:
        if baseline.resolve() in output.resolve().parents:
            raise ValueError('Variant output must be outside the baseline dataset')
        with (baseline / 'frames.csv').open() as stream:
            for row in csv.DictReader(stream):
                baseline_rows.setdefault(row['video'], []).append(row)
        baseline_sources = {v['video']: v for v in json.loads((baseline / 'manifest.json').read_text())['sources']}
        if set(baseline_sources) != names:
            raise ValueError('Source video set differs from the frozen baseline')
        for video in videos:
            digest = hashlib.sha256()
            with video.open('rb') as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                    digest.update(chunk)
            if digest.hexdigest() != baseline_sources[video.name]['sha256']:
                raise ValueError(f'Source video changed since baseline: {video}')
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f'.{output.name}-', dir=output.parent) as temporary:
        stage = Path(temporary) / 'dataset'
        (stage / 'clips').mkdir(parents=True)
        rows, sources, audit = [], [], []
        for video in videos:
            split = 'val' if video.name == VALIDATION_VIDEO else 'test' if video.name == TEST_VIDEO else 'train'
            if baseline is not None and split != 'train':
                sources.append(baseline_sources[video.name])
                for original in baseline_rows[video.name]:
                    assert original['split'] == split
                    destination = stage / 'images' / original['image']
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(baseline / 'images' / original['image'], destination)
                    rows.append(original.copy())
                print(f'Preserved frozen {split} frames: {video.name}', flush=True)
                continue
            clip = stage / 'clips' / video.stem
            prepare(video, clip, interval_seconds=1. / fps)
            manifest = json.loads((clip / 'manifest.json').read_text())
            sources.append({'video': video.name, 'split': split, 'sha256': manifest['sha256'],
                            'metadata': manifest['metadata'], 'source_frame_count': manifest['source_frame_count']})
            destination = stage / 'images' / video.stem
            destination.mkdir(parents=True)
            with (clip / 'frames.csv').open() as stream:
                originals = list(csv.DictReader(stream))
            decisions = ([{'keep': True, 'decision': 'kept'}] * len(originals))
            if filter_sharpness:
                decisions = sharpness_selection(
                    [float(r['relative_seconds']) for r in originals],
                    [sharpness_score(clip / 'images' / r['image']) for r in originals])
            for original, decision in zip(originals, decisions):
                image = original['image']
                if filter_sharpness:
                    audit.append({'image': f'{video.stem}/{image}', 'video': video.name,
                                  'relative_seconds': original['relative_seconds'], **decision})
                if decision['keep']:
                    os.link(clip / 'images' / image, destination / image)
                    rows.append({**original, 'image': f'{video.stem}/{image}', 'video': video.name, 'split': split})
            print(f"Extracted {video.name}: {sum(r['video'] == video.name for r in rows)} {split} frames", flush=True)
        splits = {split: [r['image'] for r in rows if r['split'] == split] for split in ['train', 'val', 'test']}
        assert sum(map(len, splits.values())) == len({r['image'] for r in rows}) == len(rows)
        for split, filenames in splits.items():
            (stage / f'{split}.txt').write_text('\n'.join(filenames) + '\n')
        write_json(stage / 'splits.json', splits)
        with (stage / 'frames.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        pairs = (matching_pairs(rows, temporal_seconds=4., anchor_seconds=2.) if baseline is not None else matching_pairs(rows))
        training = set(splits['train'])
        assert all(a in training and b in training for a, b in pairs)
        (stage / 'pairs.txt').write_text('\n'.join(f'{a} {b}' for a, b in pairs) + '\n')
        if filter_sharpness:
            (stage / 'quality').mkdir()
            with (stage / 'quality/selection.csv').open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(audit[0]))
                writer.writeheader()
                writer.writerows(audit)
        if baseline is not None:
            frozen = json.loads((baseline / 'splits.json').read_text())
            for split in ['val', 'test']:
                assert splits[split] == frozen[split]
        write_json(stage / 'manifest.json', {
            'status': 'extracted_for_training_quality_review', 'sources': sources,
            'sampling': f'nearest actual training PTS every{1. / fps}sec; full resolution PNG; frozen baseline holdouts when provided',
            'split_rule': f'whole videos: val={VALIDATION_VIDEO}, test={TEST_VIDEO}, all others train',
            'counts': {k: len(v) for k, v in splits.items()}, 'training_pair_count': len(pairs),
            'matching': ('4sec temporal horizon plus cross-clip anchors every2sec' if baseline is not None else
                         '8 next temporal training frames per clip plus all cross-clip anchors sampled every4 candidates'),
            'anchor_interval_seconds': 2. if baseline is not None else None,
            'colmap_parameters': {'max_image_size': 1920 if baseline is not None else 1600,
                                  'max_num_features': 8192 if baseline is not None else 4096},
            'baseline': str(baseline.resolve()) if baseline is not None else None,
            'sharpness_filter': {'enabled': filter_sharpness, 'candidate_count': len(audit),
                                 'rejected': sum(not r['keep'] for r in audit),
                                 'coverage_fallback': sum(r['decision'] == 'coverage_fallback' for r in audit),
                                 'rule': 'score<0.5x temporal+/-1sec median; preserve best frame per0.5sec slot'},
            'authoritative_split_files': ['splits.json', 'frames.csv', 'train.txt', 'val.txt', 'test.txt'],
            'clip_metadata_note': 'clips/* temporal split files are unused extraction artifacts; outer lists override them',
        })
        stage.rename(output)
    return {'output': str(output), 'frames': len(rows), 'pairs': len(pairs),
            'counts': {k: len(v) for k, v in splits.items()}}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('data/scene2'))
    parser.add_argument('--output', type=Path, default=Path('data/scene2_splatfacto'))
    parser.add_argument('--fps', type=float, default=2.)
    parser.add_argument('--filter-sharpness', action='store_true')
    parser.add_argument('--baseline', type=Path)
    args = parser.parse_args()
    print(json.dumps(prepare_many(args.source, args.output, args.fps, args.filter_sharpness, args.baseline), indent=2))
