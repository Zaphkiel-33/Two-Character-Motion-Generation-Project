
# Two-Character Motion Generation Project

This project is a monocular two-person motion pipeline for a small-scale MVP.

Given a video with two interacting people, it does three things:

1. detects and tracks both people
2. estimates 2D pose for each person
3. reconstructs a usable 3D skeleton sequence and exports a minimum delivery package

The current repository is built around one verified demo input: `video2.mp4`.

## Current State

The pipeline has been run end to end on `video2.mp4`, and the repository already includes:

- source code for the 2D, 3D, and export stages
- a reference 2D result in `outputs/video_rtmpose/`
- a published 3D result in `outputs/video_3d/`
- a separate copy of the improved 3D result in `outputs_improved/video_3d/`
- an MVP delivery package in `mvp_delivery/video2/`

Note:

- `outputs/video_3d/` is already using the improved 3D result
- `outputs_improved/video_3d/` is kept as a separate copy for comparison or later reuse

## Pipeline

The current workflow is:

`video -> detection and tracking -> role assignment -> 2D pose estimation -> temporal smoothing -> local 3D reconstruction -> MVP export`

In practice, the stages are:

1. detect person boxes with YOLO
2. track them with BoT-SORT
3. keep identities stable as `character_A` and `character_B`
4. run RTMPose on each role crop
5. smooth and export 2D pose sequences
6. reconstruct 3D skeleton motion
7. export BVH, glTF, and JSON assets

## Repository Structure

```text
.
├── README.md
├── environment.macos-arm64.yml
├── video2.mp4
├── two_character_rtmpose_pipeline.py
├── two_character_3d_reconstruction.py
├── export_dual_interaction_mvp.py
├── visualize_pose_sequences.py
├── visualize_motion3d.py
├── run_two_character_pipeline.sh
├── run_two_character_3d.sh
├── run_mvp_delivery.sh
├── outputs/
│   ├── video_rtmpose/
│   └── video_3d/
├── outputs_improved/
│   └── video_3d/
└── mvp_delivery/
    └── video2/
```

## Environment

This project was tested on macOS Apple Silicon with Conda.

Create the environment with:

```bash
conda env create -f environment.macos-arm64.yml
conda activate twochar_rtmpose
```

Main dependencies:

- `ultralytics`
- `opencv-python`
- `numpy`
- `onnxruntime-silicon`
- `rtmlib`
- `torch`
- `torchvision`

The detector weights are not required to be stored in the repository. If they are missing locally, `ultralytics` can download them automatically on first run.

## Quick Start

To run the full pipeline with the default demo video:

```bash
./run_mvp_delivery.sh
```

Default paths:

- input video: `video2.mp4`
- 2D output: `outputs/video_rtmpose/`
- improved 3D output: `outputs_improved/video_3d/`
- delivery package: `mvp_delivery/video2/`

If you want to regenerate the delivery package only:

```bash
./run_mvp_delivery.sh --force-mvp
```

If you want to rerun everything from scratch:

```bash
./run_mvp_delivery.sh --force-2d --force-3d --force-mvp
```

## Step-by-Step Run

### 1. Run the 2D stage

```bash
./run_two_character_pipeline.sh \
  --video-path video2.mp4 \
  --output-root outputs/video_rtmpose \
  --save-video
```

### 2. Run the 3D stage

```bash
./run_two_character_3d.sh \
  --sequence-json outputs/video_rtmpose/pose_sequences.json \
  --metadata-json outputs/video_rtmpose/metadata.json \
  --output-root outputs_improved/video_3d \
  --use-smoothed
```

### 3. Export the MVP package

```bash
python export_dual_interaction_mvp.py \
  --motion-json outputs_improved/video_3d/motion3d_sequences.json \
  --output-root mvp_delivery/video2 \
  --source-label video2_mvp
```

## Included Outputs

### `outputs/video_rtmpose/`

This is the reference 2D result for `video2.mp4`.

Main files:

- `metadata.json`
- `pose_sequences.json`
- `pose_sequences.csv`
- `debug_overlay.mp4`
- `pose_review_overlay.mp4`

### `outputs/video_3d/`

This is the published 3D result included for inspection. In the current repository, this folder already uses the improved 3D reconstruction result.

Main files:

- `motion3d_sequences.json`
- `motion3d_sequences.csv`
- `motion3d_review.mp4`
- `contacts.json`
- `metrics.json`
- `wham_tracking_observations.pth`

### `outputs_improved/video_3d/`

This is a separate copy of the improved 3D result.

### `mvp_delivery/video2/`

This is the exported minimum delivery package.

It includes:

- `skeletons/`
- `full_sequence/`
- `clips/`
- `metadata/`
- `package_manifest.json`
- `export_manifest.json`

Each clip folder contains:

- `character_A.bvh`
- `character_B.bvh`
- `character_A_animation.json`
- `character_B_animation.json`
- `dual_scene.gltf`
- `clip.meta.json`

## Verified Result on `video2.mp4`

The current repository state has been checked end to end on `video2.mp4`.

Improved 3D metrics:

- `character_A`: `987/987` valid frames
- `character_B`: `987/987` valid frames
- `character_A mean_bone_length_error_m`: `0.0037`
- `character_B mean_bone_length_error_m`: `0.0048`
- `hand_hand_count`: `185`
- `hand_body_count`: `123`
- `collision_correction_count`: `88`

MVP export result:

- 6 clips generated successfully
- both roles are present in all stages
- each clip includes BVH, JSON, and glTF outputs
- package metadata is complete

## Useful Options

### 2D pipeline

`two_character_rtmpose_pipeline.py` supports:

- `--pose-mode lightweight|balanced|performance`
- `--tracking-preset default|occlusion`
- `--save-video`
- `--save-crops`
- `--max-frames`

Example for a more difficult video with stronger occlusion handling:

```bash
./run_two_character_pipeline.sh \
  --video-path /absolute/path/to/video.mp4 \
  --output-root /absolute/path/to/output/video_rtmpose \
  --tracking-preset occlusion \
  --pose-mode performance \
  --save-video
```

### 3D pipeline

`two_character_3d_reconstruction.py` supports:

- `--use-smoothed`
- `--reconstruction-mode improved|legacy`
- `--visibility-mode legacy|occlusion_aware`
- `--depth-alpha`
- `--foot-lock-min-frames`
- `--foot-lock-blend`

To reproduce a legacy-style baseline:

```bash
./run_two_character_3d.sh \
  --sequence-json outputs/video_rtmpose/pose_sequences.json \
  --metadata-json outputs/video_rtmpose/metadata.json \
  --output-root outputs/video_3d_legacy \
  --use-smoothed \
  --reconstruction-mode legacy \
  --visibility-mode legacy
```

## Notes

- Some generated JSON files still contain absolute local paths from the original macOS workspace.
- The repository includes outputs so the project can be inspected without rerunning everything immediately.
- If you only want to reproduce the pipeline, the essential files are the source code and `environment.macos-arm64.yml`.

## Limitations

- This is an MVP pipeline, not a production mocap system.
- The current 3D stage outputs usable skeleton motion, not a full SMPL mesh pipeline.
- In difficult videos, identity stability under heavy occlusion is still the main bottleneck.
- Official WHAM + SMPL integration is not included here because it depends on external licensed assets.
