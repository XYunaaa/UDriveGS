# Scene2 4FPS + sharpness +1920/8192 experiment

> **For agentic workers:** Use superpowers:executing-plans inline. User has authorized execution without questions; respect the later request to run COLMAP on the host, not this CPU-only session.

**Goal:** Prepare an isolated second scene2 reconstruction using4FPS candidates, documented clarity filtering and1920px/8192-feature GPU SIFT while preserving the original experiment.

**Architecture:** Extend existing PTS extraction with a default-preserving interval parameter. Extend multi-video preparation for training-only relative sharpness filtering and byte-identical baseline holdouts. Generate train-only pairs covering the same4sec temporal neighborhood and approximately2sec cross-clip anchor spacing. Reuse frozen-map localization and training entrypoints with an explicit dataset/output location.

**Tech Stack:** Existing FFmpeg, Python, OpenCV, NumPy, COLMAP3.9.1 and splatfacto-big30k. No new dependency or training hyperparameter changes.

**Spec:** User selected the combined4FPS+sharpness and1600→1920/4096→8192 change. This tests a combined preprocessing recipe, not individual-factor attribution.

## Constraints and decisions

- Baseline data/scene2_splatfacto and results/splatfacto/scene2 remain intact; use data/scene2_4fps_splatfacto and results/splatfacto/scene2_4fps.
- Validation IMG_9765 and test IMG_9780 keep exactly the old41/38 candidate frames and original bytes. Do not expand/filter heldouts. Retain localization failure denominators; compare common localized test frames as well as coverage.
- Candidate interval0.25sec, nearest actual PTS. Full1920x1080 PNG images; source video hashes must match the baseline.
- Sharpness = variance of grayscale Laplacian after resizing to at most960px width. Flag score<0.5×the temporal±1sec local median. Preserve at least one candidate per0.5sec slot, selecting the sharpest if necessary; explicitly label coverage fallback. This is a heuristic, not proven optimal, and no percentile quota forces deletions.
- Preserve all candidate images and an audit CSV, including scores, local medians and keep/drop decisions. Only selected training frames enter train.txt/pairs/SfM.
- GPU SIFT extraction/matching on host, size1920/features8192; COLMAP3.9.1 Mapper/BA remains CPU and must be labelled as such. Do not start it here.
- Same model preset,30k iterations, seed42,1080p input and camera optimization off. Changes to sampling/matching/calibration alter SfM naturally; do not hide this as a single-factor experiment.

## Tasks

- [x] RED/GREEN tests: default extraction unchanged,4FPS real PTS, sharpness/coverage selection, frozen holdout bytes and leakage-free time-based pairs.
- [x] Implement preparation, extract and inspect actual scene2 candidate/filter outcomes.
- [x] Reuse finishing/training with new roots and1920/8192 settings; prepare host GPU entrypoint and validate its command boundary.
- [x] Independent review, full relevant tests and exact data audit; deliver host command and actual status.
