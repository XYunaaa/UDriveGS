# Scene2 SuGaR preview implementation plan

Goal: reuse the working scene1 SuGaR script for scene2_big_30k, keeping scene1 as its backward-compatible default. User requests direct execution without questions. Work in place because the project root has no usable Git repository.

Scope: parameterize code/prepare_sugar_scene1.py and code/run_scene1_sugar_host.sh with scene1/scene2; let setup_sugar.sh accept a destination log directory. Preserve Gaussian copies, exact checkpoint verification, camera calibration and output coordinate transforms. Scene2 nested video frame names must be flattened uniquely. Do not change SuGaR extraction parameters or retrain anything. Current process has no CUDA access; prepare CPU inputs and one host launch command.

- [x] Test nested-name collision handling and scene2 GLB output while retaining scene1 behavior.
- [x] Add scene selection, remove scene1-only count assumptions, prepare scene2 inputs, keep shared dependency setup logs in the selected result directory.
- [x] Run tests, real input audit, shared dependency imports and host preflight; document precise remaining GPU work.

Inputs: results/splatfacto/scene2/exports/scene2_big_30k/splat.ply and results/splatfacto/scene2/runs/scene2_big/splatfacto/quality_30k_v1. Outputs: results/mesh/scene2_sugar/input and mesh_vanilla_preview_v1/scene2_mesh.glb. Command: bash code/run_scene1_sugar_host.sh scene2. No arguments must retain the prior scene1 paths and segment_a_mesh.glb output name.

Verification: 6 tests pass, real scene2 input audit passes, shared CUDA imports pass, host preflight stops explicitly at inaccessible CUDA. Independent review found no important defects. Scene1 completed mesh/status retained. No scene2 mesh generated yet.
