# Git and scene2 handoff implementation plan

> **For agentic workers:** Use superpowers:executing-plans inline. The user already opted out of questions and authorized preparation and push to XYunaaa/UDriveGS.

**Goal:** Push a small code repository and provide a byte-preserving, verifiable scene2 transfer bundle for three independent 30k controls on another Linux NVIDIA machine.

**Architecture:** Git contains first-party code, portable configuration, small evidence and instructions. An ignored transfer directory contains only processed scene2 inputs and frozen COLMAP geometry, split into 512 MiB parts with SHA256. Existing source videos, results and environments stay local. No training starts during this task.

**Tech Stack:** Existing Python/Nerfstudio/PyTorch, Python standard library transfer tools, Git and shell.

**Spec:** User request in this conversation: manage core code, handoff documents and large data/results for reproduction, then start the three experiments on a different computer. Remote: https://github.com/XYunaaa/UDriveGS.git.

## Global Constraints

- Preserve original image/geometry bytes and fixed 474 train / 41 val / 30 localized test split; eight original test candidates remain unlocalized.
- Three controls: pose SO3xR3, scale regularization, opacity pruning 0.02; seed42, 30k, same original splatfacto-big settings.
- No training, source deletion, environment reinstallation or force push here.
- Do not publish videos, scene images, checkpoints, point clouds, model weights, media documents or credentials.
- Existing intermittent PyTorch import failure is unresolved and must be disclosed; do not silently claim cross-machine GPU training was tested.

## Review Focus

- Clone into a different absolute path: configuration and train-only sparse geometry resolve inside the new project.
- Fresh clone has no historical model: RGB-only queue must not require baseline PLY/checkpoint.
- Transfer corruption/traversal/overwriting: verify all parts and archive entries before writing, preserve conflicting destination files.
- Forced staging of a large or data/result file: repository guard rejects it.
- Remote authentication: preserve any remote history; never imply push succeeded without checking remote SHA.

## Tasks

### 1. Repository boundary and portable RGB queue

- [x] Add root allowlist ignore rules and staged-blob size/path guard with commit/push hooks.
- [x] Pin a portable copy of the actual baseline configuration; add path resolution and an RGB-only queue that requires no original trained artifacts.
- [x] Remove machine-specific Conda default and forced 4090 architecture from setup/quality entrypoints; keep user overrides and compiler compatibility.
- [x] First add regression tests for portable paths, RGB-only initialization and repository guard, then implement and run them.

### 2. Minimal data transfer and evidence

- [x] Add tested pack/verify/unpack commands with 512 MiB parts, SHA256, safe paths and no overwrite of conflicting files.
- [x] Build the actual ~1 GiB processed-input bundle; verify original source bytes and unpack in an isolated temporary path.
- [x] Record compact source inventory, dependency versions, third-party revisions and completed cleanup validation metrics in Git.

### 3. Handoff, verification and push

- [x] Write Chinese README and migration instructions with exact setup, restore, preflight, RGB queue and optional full/depth requirements.
- [x] Run CPU suite, shell syntax checks, clean-clone checks, Git ignore/size audit and an independent review.
- [x] Initialize/fetch the empty remote in this folder (no existing Git history/worktree to isolate), commit only reviewed small files and push without forcing when existing authentication permits.
- [x] Verify remote commit; otherwise leave the exact ready-to-push commit and report the authentication blocker.

## Execution ledger

- Recon: no .git exists; remote reports public/empty, default master. Connected GitHub account YANG-Deep has pull but no push permission; native Git authentication is being checked.
- Ruling: work in place while bootstrapping Git; no pre-existing tracked branch or worktree exists, and input/result files are protected by an explicit root allowlist.
- Recon: processed inputs are545 images /1,009,075,364 bytes, plus sparse_pc.ply and~53 MiB frozen COLMAP binaries. Raw videos and old optimizer checkpoints are unnecessary for the three fresh controls.

- Verification:47tests passed (with compile threads1); independent review once hit a SciPy/inspect import error, recorded alongside prior PyTorch worker instability. Actual551-file transfer verified, restored under a different absolute project path, RGB check-only passed without training, original and portable loaded configs equal. Shell syntax checks passed. Staged assets66files /~739KB; no source data, outputs, environments, third-party repos or obvious token/private-key patterns in staged assets.
- Review: Important forced-credential staging gap observed RED and fixed GREEN; independent reviewer confirmed .env*/.pem/.key rejection and returned Ship. Vendored previewer whitespace is deliberately preserved. Source-host repacking provenance audit limit documented.
- Ruling: use repository-local XYunaaa GitHub noreply identity matching the authenticated SSH account; do not inherit the shared machine global author identity.

- Final environment audit: final whole-suite rerun reported40tests /1importerror in test_scene2_depth (TorchInductor/hipify Tokenizer TypeError), despite earlier47-test pass. Exact final output saved in metadata/environment/latest-cpu-suite.txt. Ruling: preserve/disclose both results, do not retry until green or claim healthy GPU/CPU environment; deliver requested code/data handoff, with destination fresh-environment preflight mandatory. No new production code defect was identified.

- Publication: initial code commit d1ae243a58fc04b8a1353b20b79772857caecdfd pushed to XYunaaa/UDriveGS main via authenticated SSH. Focused7 handoff tests passed after the credential fix. Source data and outputs remained ignored; no training started. Final follow-up commit only updates this ledger and verification metadata.
