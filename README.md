# Two-Character Motion Generation Project

This repository is a practical MVP pipeline for extracting dual-character motion
from a monocular video. It focuses on a student-project scenario: get a
two-person interaction video running end to end, produce stable 2D tracking,
reconstruct usable 3D motion, and export a minimum delivery package that can be
inspected or taken into downstream animation tools.

## What This Repository Already Includes

- Source code for the 2D tracking, 3D reconstruction, and MVP export stages
- A verified example input video: `video2.mp4`
- Reference 2D and 3D outputs in `outputs/`
- A standalone improved 3D result in `outputs_improved/`
- A minimum delivery package in `mvp_delivery/video2/`

For GitHub browsing convenience, `outputs/video_3d/` has already been replaced
with the improved 3D result. In other words:

- `outputs/video_rtmpose/` is the reference 2D frontend output
- `outputs/video_3d/` is the recommended improved 3D result
- `outputs_improved/video_3d/` keeps the improved 3D result as a separate
  explicit folder as well

## Pipeline Overview

The current end-to-end flow is:

`video -> person detection/tracking -> role locking -> RTMPose -> temporal smoothing -> improved local 3D reconstruction -> MVP asset export`

Main stages:

1. detect persons with YOLO
2. track them with BoT-SORT
3. lock stable identities as `character_A` and `character_B`
4. run RTMPose on each role crop
5. smooth and export 2D sequences
6. reconstruct dual-character 3D motion
7. export BVH / glTF / JSON assets for MVP delivery

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

Tested locally on macOS Apple Silicon with Conda.

Create the environment:

```bash
conda env create -f environment.macos-arm64.yml
conda activate twochar_rtmpose
```

Core dependencies used by the pipeline:

- `ultralytics`
- `opencv-python`
- `numpy`
- `onnxruntime-silicon`
- `rtmlib`
- `torch`
- `torchvision`

Model weights such as `yolov8n.pt` and `yolov8m.pt` are not required to be
committed to GitHub. `ultralytics` can download the needed detector weights on
first run if they are missing locally.

## Quick Start

If you want the fastest verified end-to-end run in this repository:

```bash
./run_mvp_delivery.sh
```

Default behavior:

- input video: `video2.mp4`
- 2D output: `outputs/video_rtmpose/`
- improved 3D output: `outputs_improved/video_3d/`
- MVP package: `mvp_delivery/video2/`

## Manual Reproduction

### 1. Run the 2D frontend

```bash
./run_two_character_pipeline.sh \
  --video-path video2.mp4 \
  --output-root outputs/video_rtmpose \
  --save-video
```

### 2. Run the improved 3D stage

```bash
./run_two_character_3d.sh \
  --sequence-json outputs/video_rtmpose/pose_sequences.json \
  --metadata-json outputs/video_rtmpose/metadata.json \
  --output-root outputs_improved/video_3d \
  --use-smoothed
```

### 3. Export the MVP delivery package

```bash
python export_dual_interaction_mvp.py \
  --motion-json outputs_improved/video_3d/motion3d_sequences.json \
  --output-root mvp_delivery/video2 \
  --source-label video2_mvp
```

## Included Reference Results

### `outputs/video_rtmpose/`

Reference 2D result for the included demo video. Important files:

- `metadata.json`
- `pose_sequences.json`
- `pose_sequences.csv`
- `debug_overlay.mp4`
- `pose_review_overlay.mp4`

### `outputs/video_3d/`

Recommended 3D result for GitHub inspection. This folder now uses the improved
3D reconstruction result rather than the old legacy baseline. Important files:

- `motion3d_sequences.json`
- `motion3d_sequences.csv`
- `motion3d_review.mp4`
- `contacts.json`
- `metrics.json`
- `wham_tracking_observations.pth`

### `outputs_improved/video_3d/`

Standalone copy of the improved 3D run, kept separately for explicit comparison
or reuse in later experiments.

### `mvp_delivery/video2/`

Minimum deliverable package exported from the improved 3D result. It includes:

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

## Current Validated Status

The current repository state has been rechecked end to end on `video2.mp4`.

Current improved 3D metrics:

- `character_A`: `987/987` valid frames, `valid_ratio = 1.0`
- `character_B`: `987/987` valid frames, `valid_ratio = 1.0`
- `character_A mean_bone_length_error_m`: `0.0037`
- `character_B mean_bone_length_error_m`: `0.0048`
- `hand_hand_count`: `185`
- `hand_body_count`: `123`
- `collision_correction_count`: `88`

Current MVP package status:

- 6 clips exported successfully
- both roles included in every stage
- per-clip `BVH + JSON + glTF` generated successfully
- package metadata generated successfully

## Useful Script Notes

### `two_character_rtmpose_pipeline.py`

2D frontend with:

- YOLO person detection
- BoT-SORT tracking
- role locking
- RTMPose pose estimation
- recovery for short occlusion or missing segments

Useful flags:

- `--pose-mode lightweight|balanced|performance`
- `--tracking-preset default|occlusion`
- `--save-video`
- `--save-crops`
- `--max-frames`

### `two_character_3d_reconstruction.py`

Local dual-character 3D reconstruction backend.

Useful flags:

- `--use-smoothed`
- `--reconstruction-mode improved|legacy`
- `--visibility-mode legacy|occlusion_aware`
- `--depth-alpha`
- `--foot-lock-min-frames`
- `--foot-lock-blend`

To reproduce the earlier legacy-style baseline more closely:

```bash
./run_two_character_3d.sh \
  --sequence-json outputs/video_rtmpose/pose_sequences.json \
  --metadata-json outputs/video_rtmpose/metadata.json \
  --output-root outputs/video_3d_legacy \
  --use-smoothed \
  --reconstruction-mode legacy \
  --visibility-mode legacy
```

## Notes for GitHub Viewers

- Some generated JSON files contain absolute local paths because the assets were
  produced on a local macOS workspace.
- The included outputs are meant to make the project easy to inspect on GitHub
  without rerunning everything immediately.
- If you only want to reproduce the pipeline, the source code plus
  `environment.macos-arm64.yml` are the essential parts.

## Limitations

- This is an MVP-oriented pipeline, not a final production mocap system.
- The current 3D backend exports usable skeleton motion, not a full SMPL mesh
  pipeline.
- For heavily occluded videos, front-end identity stability is still the main
  bottleneck.
- Official WHAM + SMPL integration is not bundled here because of external
  licensed assets.
