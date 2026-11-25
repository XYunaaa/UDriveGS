# Scene2 quality experiments implementation plan

Goal: execute clean copies → three controlled Splatfacto runs → calibrated depth regularization, then select the best configuration from actual validation evidence. User explicitly authorized autonomous execution and previously opted out of questions. Current process cannot access GPU; finish all possible CPU work and prepare a single resumable host command without bypassing that restriction.

Scope: preserve baseline dataset, checkpoints and PLY. New root results/splatfacto/scene2_quality_20261125. Reuse the original474 training /41 validation /30 localized test views, seed42, full resolution, splatfacto-big and30k iterations. Report the8 failed test localizations separately. Do not use test images for hyperparameter selection or depth alignment. Work in place: no usable project Git repo.

## Decisions

- Cleanup: copy the original full-SH PLY, keeping alpha>=0.01/0.02/0.05. Preserve every retained attribute and source hash; do not claim opacity is a floater detector.
- Three independent from-scratch changes: camera SO3xR3 only; scale regularizer True/max ratio10 only; cull threshold0.02 only. Other original model/training settings unchanged.
- Ruling: implement depth supervision as a small local Splatfacto subclass, using an existing Depth Anything V2 model and calibrated inverse-Z loss. This isolates the effect of depth, unlike switching renderer/trainer simultaneously. Train from scratch with the best RGB configuration chosen on validation only. No RGB masks or guessed geometry are introduced.
- Depth maps: training images only, native undistortion and normalized camera coordinates; robust positive scale/shift alignment to training-only sparse COLMAP geometry. Invalid/poorly supported maps skipped and coverage recorded, fail when supervision coverage is inadequate. Never infer/alignment-fit validation/test images. Confidence is a data-quality heuristic, not truth, particularly for glass/screens.
- Storage: only12GiB free. Save a final checkpoint, compact NEW experiment checkpoints to inference-only (same pipeline tensors/step, remove optimizer state) after training; retain manifests/PLY/metrics. Original checkpoints untouched. Check free disk before each stage. Do not fabricate resumable optimizer state.
- Metrics: native full-resolution PSNR/SSIM/LPIPS with fixed frozen cameras and no exposure correction. Cache results by source artifact hash, save a small fixed set of qualitative panels and same virtual-camera offsets for all candidates. Novel views without ground truth get no PSNR/SSIM claims.
- Selection: validation LPIPS primary, requiring finite metrics, identical41 validation images, PSNR>=original baseline-0.3dB and SSIM>=baseline-0.003; improvement>=0.002 LPIPS, otherwise retain baseline. Geometry/floater visual judgment remains explicitly pending until panels are inspected. Once selection locks, report test metrics and30/38 coverage; never select on test scores. If stages are absent, report incomplete, not a winner.

## Tasks and files

1. [x] code/scene2_quality.py + code/test_scene2_quality.py: lossless cleanup, exact ablation config generation, native evaluation, compact inference checkpoints, resumable ordered orchestration and evidence-based report; preserve pending status without CUDA. Write failing tests first for masks/attributes, config differences, incomplete report and selection guards.
2. [x] code/scene2_depth.py + code/test_scene2_depth.py: train-only depth preparation and small model subclass; CPU tests for alignment/invalid data and gradient-bearing masked inverse-depth loss. Interface prepare_depths(base_config:Path, output_root:Path)->Path; depth_model_config(base_model_config, depth_root:Path) returns config with _target and get_outputs/get_loss_dict overrides. No test-data fitting. Record source/model provenance.
3. [x] code/run_scene2_quality.sh: host environment, sequential GPU work and single-run lock; no automatic CPU training fallback. Report stage/errors and reuse verified completed stages. CPU cleanup runs now; actual training/evaluation requires accessible CUDA.
4. [ ] Actual cleanup files/hashes; tests, import/CLI checks, input audit, fresh final review; results README/progress/status and exact unfinished stages. If host GPUs become normally accessible, execute queue and inspect results before final selection.

Preflight interfaces: native evaluation loads local subclass via PYTHONPATH=project/code; inference-only checkpoints retain every model tensor; source frame order/checksums fixed; depth maps match native cached undistortion dimensions; all GPU stages run one process at a time to release CUDA memory. A failed/partial run must not be silently reused as complete.

CPU execution complete: three cleaning files generated and every retained field checked;39 tests pass; source hash unchanged. Independent review fixed/confirmed atomic verified export. GPU stages have not run because CUDA is inaccessible; no winner selected, depth weights unavailable. Task4 remains open for real GPU execution and visual selection.
