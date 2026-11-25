# Scene2 PLY camera playback

> **For agentic workers:** Use superpowers:executing-plans inline; user previously opted out of questions.

**Goal:** Import the scene2 Gaussian PLY plus aligned capture-camera paths into SuperSplat and play each clip with its existing Play/Pause controls.

**Architecture:** Reuse SuperSplat's full-SH renderer and Timeline rather than adding a second renderer. Export INRIA camera JSON per source clip. Apply the saved Nerfstudio dataparser transform and translation scale to frozen COLMAP poses, then align the Z-up model with SuperSplat's Y-up camera. Resample each clip independently to30fps to preserve approximately real capture duration; disable loops for open paths.

**Tech Stack:** Existing Python/NumPy/SciPy/Nerfstudio; SuperSplat native camera importer. No package installation or model changes.

**Spec:** User requests PLY import, own camera positions and a play button that moves through the model. Use latest scene2 model and preserve all original training artifacts. No upload/publishing authorization inferred.

## Tasks

- [x] Verify current PLY, saved run transform and official importer conventions.
- [x] Observe failing tests for pose transform, interpolation timing and invalid timestamps, then implement camera exporter.
- [x] Export individual tracks and independently compare normalized poses against the installed training dataparser.
- [x] Provide exact PLY/track import, model orientation, timeline length/FPS/loop settings; disclose roll/browser validation limitations.

## Verified conventions

- Official SuperSplat source4069712370df3425c386610f28a814403e835543 reads camera JSON as OpenCV camera-to-world, uses forward third column, negates world X/Y, and assigns sequential timeline frames.
- Official PLY loader uses180deg Z rotation. For Nerfstudio Z-up PLY, model Euler rotation must be X90,Y0,Z180; export camera coordinates pre-rotated Rx90 so both have the same scene transform and the scene is Y-up after import.
- Camera JSON import uses position/forward target rather than full camera roll. This preserves path and viewing direction but is not an exact native-rasterization replay.
- Original PTS sampling is about2fps; direct sparse-key import at30fps would be too fast. Export dense30fps pose samples per clip; never interpolate between independent videos.
