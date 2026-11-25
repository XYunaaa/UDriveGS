# Scene1 segment_a SuGaR mesh preview

User requests an experimental mesh of the existing segment_a_30k Gaussian PLY using Anttwo/SuGaR. They previously authorized direct execution without questions. Reuse the exact PLY and its training run; do not retrain COLMAP or alter the model. Work in place because this workspace has no usable Git repository; isolate outputs under results/mesh/scene1_segment_a_sugar and packages in .envs/sugar.

## Scope and decisions

- First experiment: upstream extract_mesh.py --use_vanilla_3dgs True, no surface regularization/refinement. This is SuGaR's vanilla extraction ablation, not a fully optimized SuGaR model or relightable material recovery.
- Adapt 84 training cameras from the actual Nerfstudio parser into OpenCV camera-to-world coordinates, with no SuperSplat transforms. Undistort source images and center their principal points for SuGaR's centered projection. Verify the PLY against its original checkpoint.
- Preview budget: one density level 0.3, a decimation target of100k triangles per foreground/background mesh, at most approximately200k before projection/cleanup, 1M sampled surface points, 200k query batch, Poisson depth9. Upstream calls its decimation argument vertices, but passes it to Open3D as a triangle count. Leave both foreground and background extraction intact; skip the optional surface projection for this initial preview.
- Source: SuGaR commit7c10c4ae4a267dece512f5c7f40ed212a0a2ab44. Preserve the 3-line preview budget patch. Use isolated overlay venv with existing torch2.4/cu124; compile upstream rasterizer, simple-knn and PyTorch3D V0.7.8 for SM89/GCC12.
- Current tool session has no CUDA access; do not bypass this restriction or ask again. Prepare and validate all CPU work and dependencies; run GPU stages only if normally available. Otherwise provide one host command with logs and explicit incomplete status.

## Tasks

1. [x] Test and implement camera/image/checkpoint adapter; stage exact input with manifests and hashes.
2. [x] Clone source, create isolated overlay venv, attempt dependencies; create fail-fast host setup and extraction launcher; preserve upstream patch and provenance. Installation remains incomplete: pip could not obtain iopath, CUDA extensions not built.
3. [x] Four relevant CPU tests, Shell syntax, input audit and host preflight checked. GPU inaccessible: no real scene mesh produced. Fresh review found no critical/important code defects. Unrestricted native export test segfaulted once; test/installer now use the same thread limits already set by the launcher. Cause of that native failure not confirmed.

Interfaces: adapter produces scene/images, gs/cameras.json, gs/point_cloud/iteration_30000/point_cloud.ply and manifest.json. SuGaR requires a trailing slash on gs output path. GPU launcher uses --eval False because adapter includes training views only. Mesh output coordinates match original Z-up Gaussian PLY.
