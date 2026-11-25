"""Ordered, isolated scene2 cleanup and controlled reconstruction experiments."""

import argparse
import copy
import fcntl
import gc
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
from plyfile import PlyData, PlyElement

PROJECT = Path(__file__).resolve().parents[1]
BASE_CONFIG = PROJECT / 'configs/scene2/splatfacto_big.yml'
SOURCE_PLY = PROJECT / 'results/splatfacto/scene2/exports/scene2_big_30k/splat.ply'
DEFAULT_ROOT = PROJECT / 'results/splatfacto/scene2_quality'
ABLATIONS = ['pose', 'scale', 'prune']
CLEAN_THRESHOLDS = {'clean_001': .01, 'clean_002': .02, 'clean_005': .05}


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def clean_ply(source, output, threshold):
    if not math.isfinite(threshold) or not 0 < threshold < 1:
        raise ValueError('Opacity threshold must be finite and between0 and1')
    if output.exists():
        raise FileExistsError(output)
    ply = PlyData.read(source, mmap=True)
    vertices = ply['vertex'].data
    if len(ply.elements) != 1 or not {'opacity','f_dc_0','scale_0','rot_0'} <= set(vertices.dtype.names):
        raise ValueError('Expected a Gaussian PLY with full Gaussian attributes')
    if not np.isfinite(vertices['opacity']).all():
        raise ValueError('Nonfinite Gaussian opacity')
    cutoff = np.asarray(math.log(threshold / (1 - threshold)), dtype=vertices['opacity'].dtype)
    retained = vertices[vertices['opacity'] >= cutoff]
    if not len(retained):
        raise ValueError('Threshold would remove every Gaussian')
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix('.ply.tmp')
    PlyData([PlyElement.describe(retained, 'vertex')], text=ply.text, byte_order=ply.byte_order,
            comments=ply.comments, obj_info=ply.obj_info).write(temporary)
    temporary.replace(output)
    return {'threshold': threshold, 'source_count': len(vertices), 'retained': len(retained),
            'removed': len(vertices) - len(retained), 'ply': str(output), 'sha256': sha256(output),
            'scope': 'opacity cleanup only; retained attributes unchanged; not a floater classifier'}


def make_ablation_config(base, label):
    config = copy.deepcopy(base)
    model = config.pipeline.model
    if label == 'pose':
        model.camera_optimizer.mode = 'SO3xR3'
    elif label == 'scale':
        model.use_scale_regularization = True
    elif label == 'prune':
        model.cull_alpha_thresh = .02
    elif label != 'baseline':
        raise ValueError(f'Unknown ablation: {label}')
    return config


def choose_on_validation(records, required):
    missing = set(required) - records.keys()
    if missing:
        raise ValueError(f'Incomplete validation evidence: {sorted(missing)}')
    baseline = records['baseline']
    reference = baseline['metrics']
    for label in required:
        row = records[label]
        if row['split'] != 'val' or row['filenames'] != baseline['filenames']:
            raise ValueError('Selection requires identical ordered validation images')
        if not all(math.isfinite(row['metrics'][k]) for k in ['psnr','ssim','lpips']):
            raise ValueError('Selection metrics must be finite')
    winner = 'baseline'
    for label in required:
        scores = records[label]['metrics']
        if (scores['psnr'] >= reference['psnr'] - .3 and scores['ssim'] >= reference['ssim'] - .003
                and scores['lpips'] < records[winner]['metrics']['lpips']
                and scores['lpips'] <= reference['lpips'] - .002):
            winner = label
    return winner


def compact_checkpoint(path):
    import torch
    state = torch.load(path, map_location='cpu', weights_only=False)
    if state.get('inference_only'):
        return
    if state['step'] != 29999:
        raise ValueError('Only completed30k checkpoints may be compacted')
    compact = {'step': state['step'], 'pipeline': state['pipeline'], 'inference_only': True}
    temporary = path.with_suffix('.compact.tmp')
    torch.save(compact, temporary)
    reloaded = torch.load(temporary, map_location='cpu', weights_only=False)
    if set(reloaded['pipeline']) != set(state['pipeline']):
        raise ValueError('Compacted checkpoint lost model fields')
    for name, value in state['pipeline'].items():
        if torch.is_tensor(value):
            torch.testing.assert_close(reloaded['pipeline'][name], value, rtol=0, atol=0, equal_nan=True)
    temporary.replace(path)


def load_config(path):
    import yaml
    config = yaml.load(path.read_text(), Loader=yaml.Loader)
    manager = config.pipeline.datamanager
    for owner, name in [(config,'data'),(config,'output_dir'),(config,'load_dir'),
                        (manager,'data'),(manager.dataparser,'data')]:
        value = getattr(owner,name,None)
        if value is not None and not value.is_absolute():
            setattr(owner,name,PROJECT/value)
    return config


def checkpoint_path(config_path):
    config = load_config(config_path)
    return config.get_checkpoint_dir() / 'step-000029999.ckpt'


def initialize(root, include_baseline=True):
    config = load_config(BASE_CONFIG)
    dataset = config.pipeline.datamanager.data
    transforms = dataset / 'transforms.json'
    meta = json.loads(transforms.read_text())
    names = {s: meta[f'{s}_filenames'] for s in ['train','val','test']}
    assert {s:len(v) for s,v in names.items()} == {'train':474,'val':41,'test':30}
    assert set(names['train']).isdisjoint(names['val'] + names['test'])
    assert set(names['val']).isdisjoint(names['test'])
    fingerprint = {'base_config': sha256(BASE_CONFIG), 'transforms': sha256(transforms),
                   'sparse_initialization':sha256(dataset/meta['ply_file_path']),
                   'images': {name: sha256(dataset / name) for s in names for name in names[s]}}
    if include_baseline:
        fingerprint.update(base_checkpoint=sha256(checkpoint_path(BASE_CONFIG)),source_ply=sha256(SOURCE_PLY))
    manifest = {'base_config': str(BASE_CONFIG), 'dataset': str(dataset), 'fingerprint': fingerprint,
                'split_counts': {s:len(v) for s,v in names.items()}, 'test_candidates':38,
                'unlocalized_test_views':8, 'seed':42, 'iterations':30000,
                'selection': 'validation LPIPS, PSNR baseline-0.3dB and SSIM baseline-0.003 guards; minimum0.002 LPIPS gain',
                'checkpoint_policy': 'only NEW final checkpoints compacted to inference-only; cannot resume optimizer'}
    path = root / 'manifest.json'
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError('Experiment inputs changed; use a new experiment directory')
    write_json(path, manifest)
    return manifest


def run_rgb(root, check_only=False):
    manifest = initialize(root, include_baseline=False)
    if check_only:
        print(json.dumps({'status':'inputs_verified','split_counts':manifest['split_counts'],
                          'dataset':manifest['dataset'],'training_started':False}),flush=True)
        return
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('RGB experiments require CUDA; no CPU fallback')
    contract = {'runner_sha256':sha256(Path(__file__)),'fingerprint':manifest['fingerprint']}
    path = root/'rgb_contract.json'
    if path.exists() and json.loads(path.read_text()) != contract:
        raise ValueError('RGB code or inputs changed; preserve this run and use a new root')
    write_json(path,contract)
    records = {}
    for label in ABLATIONS:
        write_json(root/'status.json',{'status':'training_rgb','candidate':label})
        config = ensure_trained(root,label)
        records[label] = native_evaluate(root,label,config,'val')
        write_json(root/'rgb_comparison.json',{'validation':records,'selection_split':'val',
                   'visual_review':'pending','test_used_for_selection':False})
    write_json(root/'status.json',{'status':'rgb_experiments_complete_visual_review_pending',
                                  'completed':ABLATIONS,'depth_training_started':False})


def cleanup(root):
    manifest = initialize(root)
    records = {}
    path = root / 'cleanup.json'
    previous = json.loads(path.read_text()) if path.exists() else {}
    for label, threshold in CLEAN_THRESHOLDS.items():
        output = root / 'cleaned' / f'{label}.ply'
        if output.exists():
            record = previous.get(label)
            if record is None or sha256(output) != record['sha256']:
                raise ValueError(f'Unverified existing cleanup output: {output}')
            records[label] = record
        else:
            records[label] = clean_ply(SOURCE_PLY, output, threshold)
            write_json(path, records)
    assert sha256(SOURCE_PLY) == manifest['fingerprint']['source_ply']
    write_json(path, records)
    return records


def export_checkpoint_ply(config_path, output):
    import torch
    from export_checkpoint_cpu import ply_attributes
    from nerfstudio.scripts.exporter import ExportGaussianSplat
    checkpoint = checkpoint_path(config_path)
    identity = {'checkpoint_sha256':sha256(checkpoint)}
    record_path = output.with_suffix('.manifest.json')
    if output.exists() and record_path.exists():
        previous = json.loads(record_path.read_text())
        if previous['identity'] == identity and previous['sha256'] == sha256(output):
            return
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    params = {k.removeprefix('_model.gauss_params.'): v for k,v in state['pipeline'].items()
              if k.startswith('_model.gauss_params.')}
    fields = ply_attributes(params)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix('.ply.part')
    ExportGaussianSplat.write_ply(str(temporary), len(fields['x']), fields)
    vertices = PlyData.read(temporary,mmap=True)['vertex'].data
    if not len(vertices) or set(vertices.dtype.names) != set(fields):
        raise ValueError('Export is empty or missing Gaussian attributes')
    for name,values in fields.items():
        np.testing.assert_array_equal(vertices[name],values)
    record = {'identity':identity,'sha256':sha256(temporary),'gaussians':len(vertices)}
    temporary.replace(output)
    write_json(record_path,record)


def native_evaluate(root, label, config_path, split, ply_override=None):
    import torch
    from PIL import Image
    from nerfstudio.utils.eval_utils import eval_setup
    path = root / 'evaluation' / label / f'{split}.json'
    manifest = json.loads((root / 'manifest.json').read_text())
    signature = {'config':sha256(config_path), 'checkpoint':sha256(checkpoint_path(config_path)),
                 'ply':sha256(ply_override) if ply_override else None,
                 'input_fingerprint':manifest['fingerprint'], 'evaluator':sha256(Path(__file__))}
    if path.exists():
        previous = json.loads(path.read_text())
        if previous.get('signature') == signature:
            return previous
    _, pipeline, _, _ = eval_setup(config_path, test_mode=split)
    model = pipeline.model
    if ply_override:
        vertices = PlyData.read(ply_override, mmap=True)['vertex'].data
        values = {'means':np.stack([vertices[a] for a in 'xyz'],axis=1),
                  'features_dc':np.stack([vertices[f'f_dc_{i}'] for i in range(3)],axis=1),
                  'features_rest':np.stack([vertices[f'f_rest_{i}'] for i in range(45)],axis=1).reshape(-1,3,15).transpose(0,2,1),
                  'opacities':vertices['opacity'][:,None],
                  'scales':np.stack([vertices[f'scale_{i}'] for i in range(3)],axis=1),
                  'quats':np.stack([vertices[f'rot_{i}'] for i in range(4)],axis=1)}
        model.gauss_params = torch.nn.ParameterDict({k:torch.nn.Parameter(
            torch.as_tensor(np.array(v,copy=True),device=model.device),requires_grad=False) for k,v in values.items()})
    loader = pipeline.datamanager.fixed_indices_eval_dataloader
    filenames = [str(p.relative_to(config_path_dataset(config_path)))
                 for p in pipeline.datamanager.eval_dataset.image_filenames]
    expected = json.loads((Path(manifest['dataset'])/'transforms.json').read_text())[f'{split}_filenames']
    if set(filenames) != set(expected) or len(filenames) != len(expected):
        raise ValueError('Native evaluator changed the frozen view set')
    # Reorder all variants to the original frozen image list.
    by_name = {name: pair for name,pair in zip(filenames,loader)}
    loader = [by_name[name] for name in expected]
    image_dir = path.parent / split
    image_dir.mkdir(parents=True,exist_ok=True)
    rows = []
    def save_image(tensor, target):
        array = (tensor.detach().clamp(0,1).cpu().numpy()*255).round().astype(np.uint8)
        image = Image.fromarray(array)
        image.thumbnail((1920,1080))
        image.save(target, quality=90)
    with torch.no_grad():
        for index,(camera,batch) in enumerate(loader):
            if camera.metadata and 'cam_idx' in camera.metadata:
                raise ValueError('Held-out cameras must not use training pose corrections')
            outputs = model.get_outputs_for_camera(camera)
            metrics, images = model.get_image_metrics_and_images(outputs,batch)
            rows.append(metrics)
            if index in {0,5,10,20,23,len(loader)-1}:
                save_image(images['img'],image_dir/f'view_{index:03d}.jpg')
        if split == 'val':
            for index in [5,23]:
                for offset in [-.06,-.03,.03,.06]:
                    camera = copy.deepcopy(loader[index][0])
                    camera.camera_to_worlds[..., :,3] += offset * camera.camera_to_worlds[..., :,0]
                    outputs = model.get_outputs_for_camera(camera)
                    save_image(outputs['rgb'],image_dir/f'virtual_{index:03d}_{offset:+.2f}.jpg')
    result = {'label':label,'split':split,'filenames':expected,
              'metrics':{k:float(np.mean([r[k] for r in rows])) for k in ['psnr','ssim','lpips']},
              'per_image':rows,'signature':signature,
              'scope':'native full-resolution metrics; virtual offsets in normalized SfM units have no ground truth'}
    write_json(path,result)
    del outputs,loader,model,pipeline
    gc.collect()
    torch.cuda.empty_cache()
    return result


def config_path_dataset(path):
    return load_config(path).pipeline.datamanager.data


def wait_for_gpu(root, needed_gib, stage):
    import torch
    while True:
        free,_ = torch.cuda.mem_get_info()
        free_gib = free / 1024**3
        if free_gib >= needed_gib:
            write_json(root/'status.json',{'status':'gpu_memory_ready','stage':stage,'free_gib':free_gib})
            return
        write_json(root/'status.json',{'status':'waiting_gpu_memory','stage':stage,
                                     'needed_gib':needed_gib,'free_gib':free_gib})
        print(f'Waiting for GPU memory: {stage}, need{needed_gib}GiB, free{free_gib:.2f}GiB; other jobs left running',flush=True)
        time.sleep(30)


def train(root, label, base_label):
    import torch
    from nerfstudio.scripts.train import train_loop
    assert torch.cuda.is_available(), 'Training requires the host GPU'
    assert shutil.disk_usage(root).free >= 4*1024**3, 'Need4GiB free for the next training checkpoint'
    config = make_ablation_config(load_config(BASE_CONFIG), base_label if label=='depth' else label)
    config.output_dir = root/'runs'
    config.experiment_name = label
    config.timestamp = 'quality_30k_v1'
    config.max_num_iterations = 30000
    config.machine.seed = 42
    config.vis = 'tensorboard'
    config.steps_per_save = 30000
    config.steps_per_eval_image = 0
    config.steps_per_eval_all_images = 0
    config.load_dir = config.load_config = config.load_checkpoint = config.load_step = None
    if label == 'depth':
        from scene2_depth import depth_model_config
        config.pipeline.model = depth_model_config(config.pipeline.model,root/'depth')
    run_dir = config.get_base_dir()
    if run_dir.exists():
        raise FileExistsError(f'Partial training run requires inspection: {run_dir}')
    config.save_config()
    train_loop(local_rank=0,world_size=1,config=config)


def ensure_trained(root,label,base_label='baseline'):
    import torch
    # Evaluation/depth inference runs in this parent; return its cached VRAM before spawning training.
    gc.collect()
    torch.cuda.empty_cache()
    config = root/'runs'/label/'splatfacto/quality_30k_v1/config.yml'
    if not config.exists() or not checkpoint_path(config).exists():
        wait_for_gpu(root,12,f'train_{label}')
        subprocess.run([sys.executable,str(Path(__file__)),'train','--root',str(root),
                        '--label',label,'--base-label',base_label],check=True)
    compact_checkpoint(checkpoint_path(config))
    export_checkpoint_ply(config,root/'exports'/label/'splat.ply')
    return config


def run(root):
    import torch
    cleaned = cleanup(root)
    if not torch.cuda.is_available():
        write_json(root/'status.json',{'status':'pending_host_gpu','cleanup_complete':True,
            'pending':['native cleanup evaluation','pose30k','scale30k','prune30k','depth preparation/training','final selection'],
            'reason':'CUDA inaccessible to current process; no CPU training fallback'})
        write_json(root/'recommendation.json',{'status':'incomplete','best_configuration':None,
            'reason':'No new native GPU evaluation or training has completed; cannot choose a winner'})
        print('Cleanup completed. GPU stages require: bash code/run_scene2_quality.sh',flush=True)
        return
    # Keep evaluation/depth inference within5.5GiB while another host job is running.
    _,total = torch.cuda.mem_get_info()
    torch.cuda.set_per_process_memory_fraction(5.5*1024**3/total)
    wait_for_gpu(root,6,'cleanup_evaluation')
    contract = {'runner_sha256':sha256(Path(__file__)),
                'depth_sha256':sha256(Path(__file__).with_name('scene2_depth.py')),
                'baseline_fingerprint':json.loads((root/'manifest.json').read_text())['fingerprint']}
    contract_path = root/'gpu_contract.json'
    if contract_path.exists() and json.loads(contract_path.read_text()) != contract:
        raise ValueError('GPU experiment code or inputs changed; preserve these results and use a new root')
    write_json(contract_path,contract)
    records = {}
    candidates = {'baseline':(BASE_CONFIG,None)}
    candidates.update({label:(BASE_CONFIG,Path(row['ply'])) for label,row in cleaned.items()})
    for label,(config,ply) in candidates.items():
        write_json(root/'status.json',{'status':'evaluating_cleanup','candidate':label})
        records[label] = native_evaluate(root,label,config,'val',ply)
    for label in ABLATIONS:
        write_json(root/'status.json',{'status':'training_rgb','candidate':label})
        config = ensure_trained(root,label)
        candidates[label] = (config,None)
        records[label] = native_evaluate(root,label,config,'val')
    rgb_choice = choose_on_validation(records,['baseline']+ABLATIONS)
    write_json(root/'rgb_choice.json',{'validation_selected':rgb_choice,'test_metrics_used':False})
    write_json(root/'status.json',{'status':'preparing_depth','rgb_control':rgb_choice})
    from scene2_depth import prepare_depths
    wait_for_gpu(root,6,'depth_preparation')
    prepare_depths(BASE_CONFIG,root/'depth')
    config = ensure_trained(root,'depth',rgb_choice)
    candidates['depth'] = (config,None)
    records['depth'] = native_evaluate(root,'depth',config,'val')
    winner = choose_on_validation(records,list(candidates))
    recommendation = {'status':'validation_selected_visual_review_pending','best_configuration':winner,
        'selection_split':'val','validation_metrics':{k:v['metrics'] for k,v in records.items()},
        'visual_review':'pending; inspect identical virtual views and floater/hole tradeoffs before calling this visually best',
        'test_metrics_used_for_selection':False,'source_config':str(candidates[winner][0]),
        'source_ply':str(candidates[winner][1] or (SOURCE_PLY if winner=='baseline' else root/'exports'/winner/'splat.ply'))}
    write_json(root/'recommendation.json',recommendation)
    # Selection is locked before any new test evaluation.
    test_records = {}
    for label,(config,ply) in candidates.items():
        write_json(root/'status.json',{'status':'final_test_evaluation','candidate':label,'validation_choice':winner})
        test_records[label] = native_evaluate(root,label,config,'test',ply)
    write_json(root/'comparison.json',{'validation':records,'test':test_records,
        'original_test_count':38,'localized_test_count':30,'failed_test_localizations':8})
    best = root/'best.ply'
    if best.is_symlink():
        best.unlink()
    elif best.exists():
        raise FileExistsError(best)
    best.symlink_to(recommendation['source_ply'])
    write_json(root/'status.json',{'status':'gpu_experiments_complete_visual_review_pending',
                                 'validation_selected':winner,'best_ply':str(best)})


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['clean','run','rgb','train'])
    parser.add_argument('--root',type=Path,default=DEFAULT_ROOT)
    parser.add_argument('--label',choices=ABLATIONS+['depth'])
    parser.add_argument('--base-label',choices=['baseline']+ABLATIONS,default='baseline')
    parser.add_argument('--check-only',action='store_true',help='Verify RGB inputs without GPU work')
    args = parser.parse_args()
    if args.check_only and args.action != 'rgb':
        parser.error('--check-only is only valid with rgb')
    root = args.root.resolve()
    root.mkdir(parents=True,exist_ok=True)
    if args.action=='train':
        train(root,args.label,args.base_label)
    else:
        with (root/'.lock').open('w') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            try:
                if args.action=='clean':
                    cleanup(root)
                elif args.action=='rgb':
                    run_rgb(root,args.check_only)
                else:
                    run(root)
            except Exception as exc:
                write_json(root/'status.json',{'status':'failed','error':str(exc)})
                raise
