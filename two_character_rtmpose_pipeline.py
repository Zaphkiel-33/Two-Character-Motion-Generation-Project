#!/usr/bin/env python3
"""
Two-character tracking + RTMPose pipeline for motion asset generation.

This script keeps the original `video_detection.py` untouched and builds a new,
more production-oriented flow:

1. YOLO person detection + BoT-SORT tracking with ReID enabled
2. Role locking on top of tracker IDs to keep stable `character_A/B`
3. RTMPose top-down inference on each character crop
4. Lightweight temporal post-processing
5. JSON / CSV export plus optional debug video and crops

Example:
    python code/two_character_rtmpose_pipeline.py \
        --video-path video/1.mp4 \
        --output-root outputs/two_character_run \
        --display
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

try:
    import torch
except ImportError:
    torch = None

try:
    import onnxruntime as ort
except ImportError:
    ort = None

MMPoseInferencer = None

try:
    from rtmlib import RTMPose as RTMLibPose
except Exception:
    RTMLibPose = None


DEFAULT_ROLE_NAMES = ("character_A", "character_B")

KEYPOINT_NAMES: Dict[int, List[str]] = {
    17: [
        "nose",
        "left_eye",
        "right_eye",
        "left_ear",
        "right_ear",
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "left_wrist",
        "right_wrist",
        "left_hip",
        "right_hip",
        "left_knee",
        "right_knee",
        "left_ankle",
        "right_ankle",
    ],
    26: [
        "nose",
        "left_eye",
        "right_eye",
        "left_ear",
        "right_ear",
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "left_wrist",
        "right_wrist",
        "left_hip",
        "right_hip",
        "left_knee",
        "right_knee",
        "left_ankle",
        "right_ankle",
        "head",
        "neck",
        "hip",
        "left_big_toe",
        "right_big_toe",
        "left_small_toe",
        "right_small_toe",
        "left_heel",
        "right_heel",
    ],
}

ROLE_COLORS = {
    "character_A": (50, 205, 50),
    "character_B": (30, 144, 255),
}

RTMLIB_POSE_MODELS: Dict[str, Dict[str, Dict[str, object]]] = {
    "body17": {
        "lightweight": {
            "pose": "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-s_simcc-body7_pt-body7_420e-256x192-acd4a1ef_20230504.zip",
            "pose_input_size": (192, 256),
        },
        "balanced": {
            "pose": "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.zip",
            "pose_input_size": (192, 256),
        },
        "performance": {
            "pose": "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-x_simcc-body7_pt-body7_700e-384x288-71d7b7e9_20230629.zip",
            "pose_input_size": (288, 384),
        },
    },
    "body26": {
        "lightweight": {
            "pose": "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-s_simcc-body7_pt-body7-halpe26_700e-256x192-7f134165_20230605.zip",
            "pose_input_size": (192, 256),
        },
        "balanced": {
            "pose": "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-m_simcc-body7_pt-body7-halpe26_700e-256x192-4d3e73dd_20230605.zip",
            "pose_input_size": (192, 256),
        },
        "performance": {
            "pose": "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-x_simcc-body7_pt-body7-halpe26_700e-384x288-7fb6e239_20230606.zip",
            "pose_input_size": (288, 384),
        },
    },
}


@dataclass
class Detection:
    tracker_id: int
    bbox: np.ndarray
    confidence: float

    @property
    def center(self) -> np.ndarray:
        x1, y1, x2, y2 = self.bbox
        return np.array([(x1 + x2) * 0.5, (y1 + y2) * 0.5], dtype=np.float32)

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.bbox
        return max(0.0, float(x2 - x1)) * max(0.0, float(y2 - y1))


@dataclass
class RoleState:
    name: str
    tracker_id: Optional[int] = None
    last_bbox: Optional[np.ndarray] = None
    last_center: Optional[np.ndarray] = None
    velocity: Optional[np.ndarray] = None
    last_seen_frame: int = -1
    missing_frames: int = 0
    initialized: bool = False
    appearance_feature: Optional[np.ndarray] = None
    appearance_updates: int = 0
    anchor_appearance_feature: Optional[np.ndarray] = None
    anchor_appearance_updates: int = 0


@dataclass
class InteractionAnchor:
    feature: Optional[np.ndarray] = None
    bbox: Optional[np.ndarray] = None
    center: Optional[np.ndarray] = None
    velocity: Optional[np.ndarray] = None
    frame_index: int = -1


@dataclass
class PoseResult:
    keypoints: np.ndarray
    scores: np.ndarray


@dataclass
class FrameRoleRecord:
    frame_index: int
    role: str
    visible: bool
    tracker_id: Optional[int]
    tracker_bbox: Optional[List[float]]
    pose_crop_bbox: Optional[List[int]]
    raw_keypoints: Optional[List[List[float]]] = None
    raw_scores: Optional[List[float]] = None
    smoothed_keypoints: Optional[List[List[float]]] = None
    source_track_id_changed: bool = False
    is_recovered: bool = False


@dataclass
class DetectionCandidate:
    detection: Detection
    pose_crop_bbox: Tuple[int, int, int, int]
    pose_result: Optional["PoseResult"]
    pose_mean_score: float
    confident_keypoint_count: int
    appearance_feature: Optional[np.ndarray] = None
    is_recovered: bool = False


@dataclass
class DetectionTrackHistory:
    centers: List[np.ndarray] = field(default_factory=list)
    bbox_heights: List[float] = field(default_factory=list)
    last_frame_index: int = -1


@dataclass
class TrackProbationState:
    consecutive_frames: int = 0
    last_seen_frame: int = -1
    approved: bool = False


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir

    parser = argparse.ArgumentParser(
        description="Two-character RTMPose pipeline for motion asset generation."
    )
    parser.add_argument("--video-path", default=None)
    parser.add_argument("--yolo-weights", default="yolov8n.pt")
    parser.add_argument("--output-root", default=str(project_root / "outputs" / "two_character_rtmpose"))
    parser.add_argument("--tracking-preset", choices=("default", "occlusion"), default="default")
    parser.add_argument("--pose-backend", choices=("rtmlib", "mmpose"), default="rtmlib")
    parser.add_argument("--pose-alias", default="body26")
    parser.add_argument("--pose-mode", choices=("lightweight", "balanced", "performance"), default="balanced")
    parser.add_argument("--pose-config", default=None)
    parser.add_argument("--pose-checkpoint", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--display", action="store_true")
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--save-crops", action="store_true")
    parser.add_argument("--keep-temp-pose-inputs", action="store_true")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--yolo-conf", type=float, default=0.35)
    parser.add_argument("--yolo-iou", type=float, default=0.5)
    parser.add_argument("--bbox-padding", type=float, default=0.2)
    parser.add_argument("--min-keypoint-score", type=float, default=0.2)
    parser.add_argument("--ema-alpha", type=float, default=0.6)
    parser.add_argument("--interpolate-gap", type=int, default=6)
    parser.add_argument("--max-missed-frames", type=int, default=45)
    parser.add_argument("--track-buffer", type=int, default=60)
    parser.add_argument("--match-thresh", type=float, default=0.8)
    parser.add_argument("--appearance-thresh", type=float, default=0.35)
    parser.add_argument("--proximity-thresh", type=float, default=0.5)
    parser.add_argument("--disable-detection-filter", action="store_true")
    parser.add_argument("--filter-min-bbox-height-ratio", type=float, default=0.12)
    parser.add_argument("--filter-min-bbox-area-ratio", type=float, default=0.01)
    parser.add_argument("--filter-relative-min-bbox-height-ratio", type=float, default=0.38)
    parser.add_argument("--filter-relative-min-bbox-area-ratio", type=float, default=0.18)
    parser.add_argument("--filter-min-bbox-aspect-ratio", type=float, default=0.3)
    parser.add_argument("--filter-max-bbox-aspect-ratio", type=float, default=1.2)
    parser.add_argument("--filter-min-pose-mean-score", type=float, default=0.45)
    parser.add_argument("--filter-pose-keypoint-score", type=float, default=0.35)
    parser.add_argument("--filter-min-pose-keypoints", type=int, default=6)
    parser.add_argument("--filter-static-history", type=int, default=6)
    parser.add_argument("--filter-static-motion-threshold", type=float, default=12.0)
    parser.add_argument("--filter-static-height-jitter-ratio", type=float, default=0.08)
    parser.add_argument("--appearance-alpha", type=float, default=0.25)
    parser.add_argument("--appearance-weight", type=float, default=2.2)
    parser.add_argument("--appearance-min-pose-score", type=float, default=0.55)
    parser.add_argument("--anchor-appearance-alpha", type=float, default=0.08)
    parser.add_argument("--anchor-appearance-weight", type=float, default=2.8)
    parser.add_argument("--anchor-min-pose-score", type=float, default=0.72)
    parser.add_argument("--interaction-iou-threshold", type=float, default=0.08)
    parser.add_argument("--interaction-center-distance-ratio", type=float, default=0.9)
    parser.add_argument("--interaction-cooldown-frames", type=int, default=18)
    parser.add_argument("--interaction-side-prior-scale", type=float, default=0.18)
    parser.add_argument("--interaction-anchor-motion-weight", type=float, default=2.6)
    parser.add_argument("--interaction-anchor-distance-weight", type=float, default=2.4)
    parser.add_argument("--interaction-tracker-bonus-scale", type=float, default=0.28)
    parser.add_argument("--interaction-anchor-velocity-scale", type=float, default=0.9)
    parser.add_argument("--tracker-role-memory-frames", type=int, default=42)
    parser.add_argument("--tracker-role-min-history", type=int, default=3)
    parser.add_argument("--tracker-role-bonus", type=float, default=0.45)
    parser.add_argument("--tracker-role-penalty", type=float, default=2.2)
    parser.add_argument("--interaction-tracker-role-penalty-scale", type=float, default=1.35)
    parser.add_argument("--recovery-max-gap", type=int, default=12)
    parser.add_argument("--recovery-extra-padding", type=float, default=0.35)
    parser.add_argument("--recovery-velocity-scale", type=float, default=0.75)
    parser.add_argument("--recovery-min-pose-mean-score", type=float, default=0.2)
    parser.add_argument("--recovery-min-keypoints", type=int, default=4)
    parser.add_argument("--recovery-min-appearance-similarity", type=float, default=0.2)
    parser.add_argument("--recovery-appearance-margin", type=float, default=0.03)
    parser.add_argument("--recovery-max-center-distance-ratio", type=float, default=0.65)
    parser.add_argument("--new-track-probation-frames", type=int, default=3)
    parser.add_argument("--probation-extra-frames", type=int, default=2)
    parser.add_argument("--probation-low-pose-threshold", type=float, default=0.62)
    parser.add_argument("--probation-min-appearance-similarity", type=float, default=0.34)
    args = parser.parse_args()
    apply_tracking_preset(args, sys.argv[1:])
    return args


def apply_tracking_preset(args: argparse.Namespace, argv: Sequence[str]) -> None:
    if args.tracking_preset == "default":
        return

    provided_flags = {
        item.split("=", 1)[0]
        for item in argv
        if item.startswith("--")
    }

    def set_if_default(flag: str, attr: str, value: object) -> None:
        if flag not in provided_flags:
            setattr(args, attr, value)

    if args.tracking_preset == "occlusion":
        # Prefer recall and long identity memory for close-contact videos.
        set_if_default("--yolo-weights", "yolo_weights", "yolov8m.pt")
        set_if_default("--pose-mode", "pose_mode", "performance")
        set_if_default("--interpolate-gap", "interpolate_gap", 18)
        set_if_default("--ema-alpha", "ema_alpha", 0.5)
        set_if_default("--yolo-conf", "yolo_conf", 0.18)
        set_if_default("--yolo-iou", "yolo_iou", 0.62)
        set_if_default("--bbox-padding", "bbox_padding", 0.28)
        set_if_default("--max-missed-frames", "max_missed_frames", 120)
        set_if_default("--track-buffer", "track_buffer", 180)
        set_if_default("--match-thresh", "match_thresh", 0.72)
        set_if_default("--appearance-thresh", "appearance_thresh", 0.25)
        set_if_default("--proximity-thresh", "proximity_thresh", 0.35)
        set_if_default("--filter-min-bbox-height-ratio", "filter_min_bbox_height_ratio", 0.07)
        set_if_default("--filter-min-bbox-area-ratio", "filter_min_bbox_area_ratio", 0.004)
        set_if_default(
            "--filter-relative-min-bbox-height-ratio",
            "filter_relative_min_bbox_height_ratio",
            0.18,
        )
        set_if_default(
            "--filter-relative-min-bbox-area-ratio",
            "filter_relative_min_bbox_area_ratio",
            0.06,
        )
        set_if_default("--filter-min-pose-mean-score", "filter_min_pose_mean_score", 0.28)
        set_if_default("--filter-pose-keypoint-score", "filter_pose_keypoint_score", 0.25)
        set_if_default("--filter-min-pose-keypoints", "filter_min_pose_keypoints", 4)
        set_if_default("--filter-static-history", "filter_static_history", 10)
        set_if_default("--filter-static-motion-threshold", "filter_static_motion_threshold", 6.0)
        set_if_default("--appearance-weight", "appearance_weight", 2.8)
        set_if_default("--appearance-min-pose-score", "appearance_min_pose_score", 0.42)
        set_if_default("--anchor-appearance-weight", "anchor_appearance_weight", 3.4)
        set_if_default("--anchor-min-pose-score", "anchor_min_pose_score", 0.55)
        set_if_default("--interaction-cooldown-frames", "interaction_cooldown_frames", 36)
        set_if_default("--interaction-tracker-bonus-scale", "interaction_tracker_bonus_scale", 0.18)
        set_if_default("--tracker-role-memory-frames", "tracker_role_memory_frames", 120)
        set_if_default("--tracker-role-min-history", "tracker_role_min_history", 2)
        set_if_default("--tracker-role-bonus", "tracker_role_bonus", 0.8)
        set_if_default("--tracker-role-penalty", "tracker_role_penalty", 3.0)
        set_if_default("--interaction-tracker-role-penalty-scale", "interaction_tracker_role_penalty_scale", 1.7)
        set_if_default("--recovery-max-gap", "recovery_max_gap", 48)
        set_if_default("--recovery-extra-padding", "recovery_extra_padding", 0.55)
        set_if_default("--recovery-velocity-scale", "recovery_velocity_scale", 0.5)
        set_if_default("--recovery-min-pose-mean-score", "recovery_min_pose_mean_score", 0.12)
        set_if_default("--recovery-min-keypoints", "recovery_min_keypoints", 3)
        set_if_default("--recovery-min-appearance-similarity", "recovery_min_appearance_similarity", 0.12)
        set_if_default("--recovery-appearance-margin", "recovery_appearance_margin", 0.0)
        set_if_default("--recovery-max-center-distance-ratio", "recovery_max_center_distance_ratio", 0.95)
        set_if_default("--new-track-probation-frames", "new_track_probation_frames", 2)
        set_if_default("--probation-extra-frames", "probation_extra_frames", 1)
        set_if_default("--probation-low-pose-threshold", "probation_low_pose_threshold", 0.45)
        set_if_default(
            "--probation-min-appearance-similarity",
            "probation_min_appearance_similarity",
            0.18,
        )


def select_device(device_arg: str, pose_backend: str) -> str:
    if device_arg != "auto":
        return device_arg
    if pose_backend == "rtmlib" and ort is not None:
        providers = ort.get_available_providers()
        if "CoreMLExecutionProvider" in providers or "MPSExecutionProvider" in providers:
            return "mps"
    if torch is not None and torch.cuda.is_available():
        return "cuda:0"
    if (
        pose_backend == "mmpose"
        and torch is not None
        and hasattr(torch, "backends")
        and hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        return "mps"
    return "cpu"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_input_video_path(video_arg: Optional[str], project_root: Path) -> Path:
    if video_arg:
        return Path(video_arg).expanduser().resolve()

    video_dir = project_root / "video"
    candidate_paths = [
        video_dir / "3.mp4",
        video_dir / "1.mp4",
    ]
    for candidate in candidate_paths:
        if candidate.exists():
            return candidate.resolve()

    if video_dir.exists():
        for pattern in ("*.mp4", "*.mov", "*.m4v", "*.avi"):
            matches = sorted(video_dir.glob(pattern))
            if matches:
                return matches[0].resolve()

    raise SystemExit(
        "No input video was found. Pass --video-path /path/to/video.mp4 "
        "or place a video file under ./video/."
    )


def resolve_yolo_weights_source(weights_arg: str, project_root: Path) -> str:
    weights_path = Path(weights_arg).expanduser()
    if weights_path.exists():
        return str(weights_path.resolve())

    if not weights_path.is_absolute():
        for local_candidate in (
            project_root / weights_arg,
            project_root / "weights" / weights_arg,
        ):
            if local_candidate.exists():
                return str(local_candidate.resolve())

    return weights_arg


def select_yolo_device(device: str) -> str:
    if device == "mps":
        if (
            torch is not None
            and hasattr(torch, "backends")
            and hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
        ):
            return "mps"
        return "cpu"

    if device.startswith("cuda"):
        if torch is not None and torch.cuda.is_available():
            return device
        return "cpu"

    return device


def build_tracker_yaml(args: argparse.Namespace, output_root: Path) -> Path:
    tracker_path = output_root / "botsort_reid.yaml"
    tracker_path.write_text(
        "\n".join(
            [
                "tracker_type: botsort",
                "track_high_thresh: 0.25",
                "track_low_thresh: 0.1",
                "new_track_thresh: 0.25",
                f"track_buffer: {args.track_buffer}",
                f"match_thresh: {args.match_thresh}",
                "fuse_score: True",
                "gmc_method: sparseOptFlow",
                f"proximity_thresh: {args.proximity_thresh}",
                f"appearance_thresh: {args.appearance_thresh}",
                "with_reid: True",
                "model: auto",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return tracker_path


def bbox_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    union_area = (
        max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        + max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        - inter_area
    )
    return inter_area / union_area if union_area > 0 else 0.0


def expand_bbox(
    bbox: np.ndarray,
    frame_shape: Sequence[int],
    padding_ratio: float,
) -> Tuple[int, int, int, int]:
    height, width = frame_shape[:2]
    x1, y1, x2, y2 = bbox.astype(np.float32)
    bw = x2 - x1
    bh = y2 - y1
    pad_x = bw * padding_ratio
    pad_y = bh * padding_ratio
    nx1 = max(0, int(round(x1 - pad_x)))
    ny1 = max(0, int(round(y1 - pad_y)))
    nx2 = min(width, int(round(x2 + pad_x)))
    ny2 = min(height, int(round(y2 + pad_y)))
    return nx1, ny1, nx2, ny2


def clip_bbox_to_frame(
    bbox: np.ndarray,
    frame_shape: Sequence[int],
) -> np.ndarray:
    height, width = frame_shape[:2]
    x1, y1, x2, y2 = bbox.astype(np.float32)
    x1 = min(max(0.0, x1), max(0.0, float(width - 1)))
    y1 = min(max(0.0, y1), max(0.0, float(height - 1)))
    x2 = min(max(x1 + 1.0, x2), float(width))
    y2 = min(max(y1 + 1.0, y2), float(height))
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def normalize_feature(feature: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(feature))
    if norm <= 1e-8:
        return feature.astype(np.float32, copy=True)
    return (feature / norm).astype(np.float32, copy=False)


def compute_appearance_feature(
    crop_bgr: np.ndarray,
    pose_result: Optional[PoseResult],
    pose_score_threshold: float,
) -> Optional[np.ndarray]:
    if crop_bgr.size == 0:
        return None

    crop_h, crop_w = crop_bgr.shape[:2]
    sample = crop_bgr

    if pose_result is not None and len(pose_result.keypoints) >= 13:
        torso_indices = [5, 6, 11, 12]
        valid_points = []
        for idx in torso_indices:
            if idx >= len(pose_result.keypoints):
                continue
            if idx < len(pose_result.scores) and pose_result.scores[idx] < pose_score_threshold:
                continue
            x, y = pose_result.keypoints[idx]
            if not np.isfinite(x) or not np.isfinite(y):
                continue
            valid_points.append((float(x), float(y)))

        if len(valid_points) >= 2:
            xs = [point[0] for point in valid_points]
            ys = [point[1] for point in valid_points]
            torso_x1 = max(0, int(round(min(xs) - max(8.0, 0.18 * crop_w))))
            torso_y1 = max(0, int(round(min(ys) - max(8.0, 0.12 * crop_h))))
            torso_x2 = min(crop_w, int(round(max(xs) + max(8.0, 0.18 * crop_w))))
            torso_y2 = min(crop_h, int(round(max(ys) + max(8.0, 0.2 * crop_h))))
            if torso_x2 > torso_x1 and torso_y2 > torso_y1:
                sample = crop_bgr[torso_y1:torso_y2, torso_x1:torso_x2]

    if sample.size == 0:
        return None

    hsv = cv2.cvtColor(sample, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [12, 12, 6], [0, 180, 0, 256, 0, 256])
    if hist is None:
        return None
    return normalize_feature(hist.reshape(-1))


def appearance_similarity(feature_a: Optional[np.ndarray], feature_b: Optional[np.ndarray]) -> float:
    if feature_a is None or feature_b is None:
        return 0.0
    return float(np.dot(feature_a, feature_b))


def predict_recovery_bbox(
    state: RoleState,
    frame_shape: Sequence[int],
    gap_frames: int,
    velocity_scale: float,
) -> Optional[np.ndarray]:
    if state.last_bbox is None or state.last_center is None:
        return None

    bbox = state.last_bbox.astype(np.float32).copy()
    width = float(bbox[2] - bbox[0])
    height = float(bbox[3] - bbox[1])
    shift = np.zeros((2,), dtype=np.float32)

    if state.velocity is not None:
        shift = state.velocity.astype(np.float32) * float(min(gap_frames, 3)) * float(velocity_scale)
        shift_norm = float(np.linalg.norm(shift))
        max_shift = max(width, height) * (0.35 + 0.1 * float(min(gap_frames, 3)))
        if shift_norm > max_shift and shift_norm > 1e-6:
            shift *= max_shift / shift_norm

    predicted_center = state.last_center.astype(np.float32) + shift
    predicted_bbox = np.array(
        [
            predicted_center[0] - 0.5 * width,
            predicted_center[1] - 0.5 * height,
            predicted_center[0] + 0.5 * width,
            predicted_center[1] + 0.5 * height,
        ],
        dtype=np.float32,
    )
    return clip_bbox_to_frame(predicted_bbox, frame_shape)


def role_side_prior(role_name: str, detection: Detection, frame_width: int) -> float:
    center_x = float(detection.center[0]) / max(1.0, float(frame_width))
    if role_name == "character_A":
        return 1.0 - center_x
    return center_x


class RoleLockManager:
    def __init__(
        self,
        role_names: Sequence[str],
        max_missed_frames: int,
        appearance_alpha: float,
        appearance_weight: float,
        appearance_min_pose_score: float,
        anchor_appearance_alpha: float,
        anchor_appearance_weight: float,
        anchor_min_pose_score: float,
        interaction_iou_threshold: float,
        interaction_center_distance_ratio: float,
        interaction_cooldown_frames: int,
        interaction_side_prior_scale: float,
        interaction_anchor_motion_weight: float,
        interaction_anchor_distance_weight: float,
        interaction_tracker_bonus_scale: float,
        interaction_anchor_velocity_scale: float,
        tracker_role_memory_frames: int,
        tracker_role_min_history: int,
        tracker_role_bonus: float,
        tracker_role_penalty: float,
        interaction_tracker_role_penalty_scale: float,
    ) -> None:
        self.role_names = list(role_names)
        self.states = {name: RoleState(name=name) for name in role_names}
        self.max_missed_frames = max_missed_frames
        self.appearance_alpha = appearance_alpha
        self.appearance_weight = appearance_weight
        self.appearance_min_pose_score = appearance_min_pose_score
        self.anchor_appearance_alpha = anchor_appearance_alpha
        self.anchor_appearance_weight = anchor_appearance_weight
        self.anchor_min_pose_score = anchor_min_pose_score
        self.interaction_iou_threshold = interaction_iou_threshold
        self.interaction_center_distance_ratio = interaction_center_distance_ratio
        self.interaction_cooldown_frames = interaction_cooldown_frames
        self.interaction_side_prior_scale = interaction_side_prior_scale
        self.interaction_anchor_motion_weight = interaction_anchor_motion_weight
        self.interaction_anchor_distance_weight = interaction_anchor_distance_weight
        self.interaction_tracker_bonus_scale = interaction_tracker_bonus_scale
        self.interaction_anchor_velocity_scale = interaction_anchor_velocity_scale
        self.tracker_role_memory_frames = tracker_role_memory_frames
        self.tracker_role_min_history = tracker_role_min_history
        self.tracker_role_bonus = tracker_role_bonus
        self.tracker_role_penalty = tracker_role_penalty
        self.interaction_tracker_role_penalty_scale = interaction_tracker_role_penalty_scale
        self.interaction_cooldown = 0
        self.interaction_anchors = {
            name: InteractionAnchor() for name in role_names
        }
        self.tracker_role_history: Dict[int, List[Tuple[int, str]]] = {}

    def _clear_interaction_anchors(self) -> None:
        self.interaction_anchors = {
            name: InteractionAnchor() for name in self.role_names
        }

    def _capture_interaction_anchors(self, frame_index: int) -> None:
        for role_name, state in self.states.items():
            feature = state.anchor_appearance_feature
            if feature is None:
                feature = state.appearance_feature

            self.interaction_anchors[role_name] = InteractionAnchor(
                feature=None if feature is None else feature.copy(),
                bbox=None if state.last_bbox is None else state.last_bbox.copy(),
                center=None if state.last_center is None else state.last_center.copy(),
                velocity=None if state.velocity is None else state.velocity.copy(),
                frame_index=frame_index,
            )

    def _predict_interaction_anchor_bbox(
        self,
        role_name: str,
        frame_shape: Sequence[int],
        frame_index: int,
    ) -> Optional[np.ndarray]:
        anchor = self.interaction_anchors.get(role_name)
        if anchor is None or anchor.bbox is None:
            return None

        predicted_bbox = anchor.bbox.astype(np.float32).copy()
        if anchor.velocity is not None and anchor.frame_index >= 0:
            elapsed = max(0, frame_index - anchor.frame_index)
            if elapsed > 0:
                shift = (
                    anchor.velocity.astype(np.float32)
                    * float(min(elapsed, 6))
                    * float(self.interaction_anchor_velocity_scale)
                )
                predicted_bbox[[0, 2]] += shift[0]
                predicted_bbox[[1, 3]] += shift[1]
        return clip_bbox_to_frame(predicted_bbox, frame_shape)

    def _prune_tracker_role_history(self, frame_index: int) -> None:
        cutoff = frame_index - self.tracker_role_memory_frames
        stale_track_ids = []
        for track_id, observations in self.tracker_role_history.items():
            kept = [(obs_frame, obs_role) for obs_frame, obs_role in observations if obs_frame >= cutoff]
            if kept:
                self.tracker_role_history[track_id] = kept
            else:
                stale_track_ids.append(track_id)
        for track_id in stale_track_ids:
            self.tracker_role_history.pop(track_id, None)

    def _record_tracker_role(
        self,
        tracker_id: int,
        role_name: str,
        frame_index: int,
    ) -> None:
        if tracker_id < 0:
            return
        observations = self.tracker_role_history.setdefault(tracker_id, [])
        observations.append((frame_index, role_name))
        cutoff = frame_index - self.tracker_role_memory_frames
        self.tracker_role_history[tracker_id] = [
            (obs_frame, obs_role)
            for obs_frame, obs_role in observations
            if obs_frame >= cutoff
        ]

    def _is_close_interaction(
        self,
        detection_a: Detection,
        detection_b: Detection,
    ) -> bool:
        iou = bbox_iou(detection_a.bbox, detection_b.bbox)
        if iou >= self.interaction_iou_threshold:
            return True

        width_a = float(detection_a.bbox[2] - detection_a.bbox[0])
        height_a = float(detection_a.bbox[3] - detection_a.bbox[1])
        width_b = float(detection_b.bbox[2] - detection_b.bbox[0])
        height_b = float(detection_b.bbox[3] - detection_b.bbox[1])
        avg_diag = 0.5 * (math.hypot(width_a, height_a) + math.hypot(width_b, height_b))
        if avg_diag <= 1e-6:
            return False

        center_distance = float(np.linalg.norm(detection_a.center - detection_b.center))
        return center_distance / avg_diag <= self.interaction_center_distance_ratio

    def assign(
        self,
        candidates: List[DetectionCandidate],
        frame_index: int,
        frame_shape: Sequence[int],
    ) -> Dict[str, Optional[DetectionCandidate]]:
        assignments = {name: None for name in self.role_names}
        if not candidates:
            self._update_states(assignments, frame_index)
            return assignments

        if not all(self.states[name].initialized for name in self.role_names) and len(candidates) >= 2:
            ordered = sorted(candidates, key=lambda item: float(item.detection.center[0]))
            assignments[self.role_names[0]] = ordered[0]
            assignments[self.role_names[1]] = ordered[-1]
            self._update_states(assignments, frame_index)
            return assignments

        frame_h, frame_w = frame_shape[:2]
        candidate_ids: List[Optional[int]] = [None] + list(range(len(candidates)))
        best_score = -float("inf")
        best_indices: Optional[Tuple[Optional[int], ...]] = None
        interaction_mode = self.interaction_cooldown > 0
        best_pair_is_close = False
        self._prune_tracker_role_history(frame_index)

        for index_combo in product(candidate_ids, repeat=len(self.role_names)):
            used = [idx for idx in index_combo if idx is not None]
            if len(set(used)) != len(used):
                continue

            combo_interaction_mode = interaction_mode
            pair_is_close = False
            score = 0.0
            if len(self.role_names) == 2:
                left = index_combo[0]
                right = index_combo[1]
                if left is not None and right is not None:
                    left_candidate = candidates[left]
                    right_candidate = candidates[right]
                    pair_is_close = self._is_close_interaction(
                        left_candidate.detection,
                        right_candidate.detection,
                    )
                    combo_interaction_mode = combo_interaction_mode or pair_is_close
                    left_x = left_candidate.detection.center[0]
                    right_x = right_candidate.detection.center[0]
                    state_a = self.states[self.role_names[0]]
                    state_b = self.states[self.role_names[1]]
                    if (
                        state_a.last_center is not None
                        and state_b.last_center is not None
                        and state_a.last_center[0] < state_b.last_center[0]
                        and left_x > right_x
                    ):
                        score -= 0.3

            for role_name, det_idx in zip(self.role_names, index_combo):
                candidate = candidates[det_idx] if det_idx is not None else None
                score += self._score_role_assignment(
                    role_name,
                    candidate,
                    frame_w,
                    frame_h,
                    frame_index,
                    interaction_mode=combo_interaction_mode,
                )

            if score > best_score:
                best_score = score
                best_indices = index_combo
                best_pair_is_close = pair_is_close

        if best_indices is not None:
            for role_name, det_idx in zip(self.role_names, best_indices):
                assignments[role_name] = candidates[det_idx] if det_idx is not None else None

        if best_pair_is_close:
            if self.interaction_cooldown == 0:
                self._capture_interaction_anchors(frame_index)
            self.interaction_cooldown = self.interaction_cooldown_frames
        elif self.interaction_cooldown > 0:
            self.interaction_cooldown -= 1
            if self.interaction_cooldown == 0:
                self._clear_interaction_anchors()

        self._update_states(assignments, frame_index)
        return assignments

    def _score_role_assignment(
        self,
        role_name: str,
        candidate: Optional[DetectionCandidate],
        frame_width: int,
        frame_height: int,
        frame_index: int,
        interaction_mode: bool,
    ) -> float:
        state = self.states[role_name]

        if candidate is None:
            return -1.5 - 0.05 * state.missing_frames
        detection = candidate.detection

        score = 0.2 * detection.confidence
        side_prior_scale = self.interaction_side_prior_scale if interaction_mode else 1.0
        score += 0.7 * side_prior_scale * role_side_prior(role_name, detection, frame_width)

        if not state.initialized:
            if state.appearance_feature is not None and candidate.appearance_feature is not None:
                score += self.appearance_weight * appearance_similarity(
                    state.appearance_feature,
                    candidate.appearance_feature,
                )
            return score

        if state.tracker_id is not None and detection.tracker_id == state.tracker_id:
            tracker_bonus = 3.5
            if interaction_mode:
                tracker_bonus *= self.interaction_tracker_bonus_scale
            score += tracker_bonus

        if state.last_bbox is not None:
            last_bbox_iou_weight = 1.15 if interaction_mode else 2.0
            score += last_bbox_iou_weight * bbox_iou(state.last_bbox, detection.bbox)

            last_center = state.last_center if state.last_center is not None else detection.center
            frame_diag = math.hypot(frame_width, frame_height)
            distance = float(np.linalg.norm(detection.center - last_center)) / max(frame_diag, 1.0)
            center_distance_weight = 1.35 if interaction_mode else 2.5
            score -= center_distance_weight * distance

            last_area = max(state.last_bbox[2] - state.last_bbox[0], 1.0) * max(
                state.last_bbox[3] - state.last_bbox[1], 1.0
            )
            area_ratio = max(detection.area, 1.0) / max(last_area, 1.0)
            score -= 0.15 * abs(math.log(area_ratio))

        if state.appearance_feature is not None and candidate.appearance_feature is not None:
            similarity = appearance_similarity(state.appearance_feature, candidate.appearance_feature)
            score += self.appearance_weight * similarity
            if state.appearance_updates >= 3 and similarity < 0.45:
                score -= 1.2

        if state.anchor_appearance_feature is not None and candidate.appearance_feature is not None:
            anchor_similarity = appearance_similarity(
                state.anchor_appearance_feature,
                candidate.appearance_feature,
            )
            anchor_weight = self.anchor_appearance_weight * (1.2 if interaction_mode else 0.75)
            score += anchor_weight * anchor_similarity
            if state.anchor_appearance_updates >= 3 and anchor_similarity < 0.42:
                score -= 1.4 if interaction_mode else 0.7

        if interaction_mode:
            interaction_anchor = self.interaction_anchors.get(role_name)
            if (
                interaction_anchor is not None
                and interaction_anchor.feature is not None
                and candidate.appearance_feature is not None
            ):
                interaction_similarity = appearance_similarity(
                    interaction_anchor.feature,
                    candidate.appearance_feature,
                )
                score += 3.2 * interaction_similarity
                if interaction_similarity < 0.4:
                    score -= 1.6

            predicted_anchor_bbox = self._predict_interaction_anchor_bbox(
                role_name,
                (frame_height, frame_width),
                frame_index,
            )
            if predicted_anchor_bbox is not None:
                score += self.interaction_anchor_motion_weight * bbox_iou(
                    predicted_anchor_bbox,
                    detection.bbox,
                )
                predicted_center = np.array(
                    [
                        0.5 * (predicted_anchor_bbox[0] + predicted_anchor_bbox[2]),
                        0.5 * (predicted_anchor_bbox[1] + predicted_anchor_bbox[3]),
                    ],
                    dtype=np.float32,
                )
                frame_diag = math.hypot(frame_width, frame_height)
                anchor_distance = float(np.linalg.norm(detection.center - predicted_center)) / max(
                    frame_diag,
                    1.0,
                )
                score -= self.interaction_anchor_distance_weight * anchor_distance

        tracker_history = self.tracker_role_history.get(detection.tracker_id)
        if tracker_history and len(tracker_history) >= self.tracker_role_min_history:
            same_role_votes = sum(1 for _, owner_role in tracker_history if owner_role == role_name)
            other_role_votes = len(tracker_history) - same_role_votes
            if same_role_votes > 0:
                score += self.tracker_role_bonus * (
                    float(same_role_votes) / float(len(tracker_history))
                )
            if other_role_votes > 0:
                penalty_scale = (
                    self.interaction_tracker_role_penalty_scale if interaction_mode else 1.0
                )
                score -= (
                    self.tracker_role_penalty
                    * penalty_scale
                    * (float(other_role_votes) / float(len(tracker_history)))
                )
                if same_role_votes == 0:
                    score -= 0.8 * penalty_scale

        if state.missing_frames > self.max_missed_frames:
            score -= 0.6

        return score

    def _update_states(
        self,
        assignments: Dict[str, Optional[DetectionCandidate]],
        frame_index: int,
    ) -> None:
        for role_name, candidate in assignments.items():
            state = self.states[role_name]
            if candidate is None:
                state.missing_frames += 1
                if state.missing_frames > self.max_missed_frames:
                    state.tracker_id = None
                continue

            self._apply_candidate_to_state(state, candidate, frame_index)

    def apply_recovered_candidate(
        self,
        role_name: str,
        candidate: DetectionCandidate,
        frame_index: int,
    ) -> None:
        self._apply_candidate_to_state(self.states[role_name], candidate, frame_index)

    def _apply_candidate_to_state(
        self,
        state: RoleState,
        candidate: DetectionCandidate,
        frame_index: int,
    ) -> None:
        detection = candidate.detection
        previous_center = None if state.last_center is None else state.last_center.copy()
        state.tracker_id = detection.tracker_id
        state.last_bbox = detection.bbox.copy()
        state.last_center = detection.center.copy()
        if previous_center is not None:
            state.velocity = state.last_center - previous_center
        state.last_seen_frame = frame_index
        state.missing_frames = 0
        state.initialized = True
        if (
            candidate.appearance_feature is not None
            and candidate.pose_mean_score >= self.appearance_min_pose_score
        ):
            if state.appearance_feature is None:
                state.appearance_feature = candidate.appearance_feature.copy()
            else:
                blended = (
                    (1.0 - self.appearance_alpha) * state.appearance_feature
                    + self.appearance_alpha * candidate.appearance_feature
                )
                state.appearance_feature = normalize_feature(blended)
            state.appearance_updates += 1

        if (
            self.interaction_cooldown == 0
            and candidate.appearance_feature is not None
            and candidate.pose_mean_score >= self.anchor_min_pose_score
        ):
            if state.anchor_appearance_feature is None:
                state.anchor_appearance_feature = candidate.appearance_feature.copy()
            else:
                anchor_blended = (
                    (1.0 - self.anchor_appearance_alpha) * state.anchor_appearance_feature
                    + self.anchor_appearance_alpha * candidate.appearance_feature
                )
                state.anchor_appearance_feature = normalize_feature(anchor_blended)
            state.anchor_appearance_updates += 1

        if not candidate.is_recovered:
            self._record_tracker_role(detection.tracker_id, state.name, frame_index)


def resolve_rtmlib_pose_spec(
    pose_alias: str,
    pose_mode: str,
    pose_checkpoint: Optional[str],
) -> Tuple[str, Tuple[int, int]]:
    if pose_alias not in RTMLIB_POSE_MODELS:
        raise ValueError(
            f"Unsupported rtmlib pose alias: {pose_alias}. "
            f"Choose from {sorted(RTMLIB_POSE_MODELS)}."
        )
    spec = RTMLIB_POSE_MODELS[pose_alias][pose_mode]
    pose_source = pose_checkpoint or str(spec["pose"])
    pose_input_size = tuple(spec["pose_input_size"])
    return pose_source, pose_input_size


class RTMPoseRunner:
    def __init__(
        self,
        pose_backend: str,
        pose_alias: str,
        pose_mode: str,
        pose_config: Optional[str],
        pose_checkpoint: Optional[str],
        device: str,
        temp_dir: Path,
        keep_temp_pose_inputs: bool,
    ) -> None:
        self.pose_backend = pose_backend
        self.temp_dir = temp_dir
        self.keep_temp_pose_inputs = keep_temp_pose_inputs
        self._allow_ndarray_input = True

        if pose_backend == "mmpose":
            global MMPoseInferencer
            if MMPoseInferencer is None:
                try:
                    from mmpose.apis import MMPoseInferencer as ImportedMMPoseInferencer
                except Exception as exc:
                    raise SystemExit(
                        "MMPose backend requested, but MMPose is not usable in this environment. "
                        "Use --pose-backend rtmlib or install full MMPose + mmcv."
                    ) from exc
                MMPoseInferencer = ImportedMMPoseInferencer
            kwargs = {
                "det_model": "whole_image",
                "device": device,
            }
            if pose_config:
                kwargs["pose2d"] = pose_config
                if pose_checkpoint:
                    kwargs["pose2d_weights"] = pose_checkpoint
            else:
                kwargs["pose2d"] = pose_alias

            self.inferencer = MMPoseInferencer(**kwargs)
            self.pose_source = pose_checkpoint or pose_config or pose_alias
            self.pose_input_size = None
        elif pose_backend == "rtmlib":
            if RTMLibPose is None:
                raise SystemExit(
                    "rtmlib backend requested, but rtmlib is not installed. "
                    "Install `rtmlib` and `onnxruntime` first."
                )
            pose_source, pose_input_size = resolve_rtmlib_pose_spec(
                pose_alias=pose_alias,
                pose_mode=pose_mode,
                pose_checkpoint=pose_checkpoint,
            )
            self.inferencer = RTMLibPose(
                pose_source,
                model_input_size=pose_input_size,
                to_openpose=False,
                backend="onnxruntime",
                device=device,
            )
            self.pose_source = pose_source
            self.pose_input_size = pose_input_size
        else:
            raise ValueError(f"Unsupported pose backend: {pose_backend}")

    def predict(self, crop_bgr: np.ndarray, frame_index: int, role_name: str) -> Optional[PoseResult]:
        if self.pose_backend == "rtmlib":
            keypoints, scores = self.inferencer(crop_bgr)
            keypoints_arr = np.asarray(keypoints, dtype=np.float32)
            scores_arr = np.asarray(scores, dtype=np.float32)

            if keypoints_arr.size == 0:
                return None
            if keypoints_arr.ndim == 3:
                keypoints_arr = keypoints_arr[0]
            if scores_arr.ndim == 2:
                scores_arr = scores_arr[0]
            if len(keypoints_arr) == 0:
                return None

            return PoseResult(keypoints=keypoints_arr, scores=scores_arr.reshape(-1))

        result = None

        if self._allow_ndarray_input:
            try:
                generator = self.inferencer(
                    crop_bgr,
                    show=False,
                    return_vis=False,
                    return_datasamples=False,
                )
                result = next(generator)
            except Exception:
                self._allow_ndarray_input = False

        if result is None:
            temp_path = self.temp_dir / f"{role_name}_{frame_index:06d}.jpg"
            cv2.imwrite(str(temp_path), crop_bgr)
            generator = self.inferencer(
                str(temp_path),
                show=False,
                return_vis=False,
                return_datasamples=False,
            )
            result = next(generator)
            if not self.keep_temp_pose_inputs and temp_path.exists():
                temp_path.unlink()

        return parse_pose_result(result)


def parse_pose_result(result: dict) -> Optional[PoseResult]:
    predictions = result.get("predictions", [])
    if not predictions:
        return None

    instances = predictions[0] if isinstance(predictions[0], list) else predictions
    if not instances:
        return None

    def instance_score(instance: dict) -> float:
        scores = instance.get("keypoint_scores") or instance.get("keypoints_visible")
        if scores is None:
            return 0.0
        scores_arr = np.asarray(scores, dtype=np.float32).reshape(-1)
        return float(scores_arr.mean()) if scores_arr.size else 0.0

    best_instance = max(instances, key=instance_score)
    keypoints = np.asarray(best_instance.get("keypoints"), dtype=np.float32)
    if keypoints.ndim == 3:
        keypoints = keypoints[0]

    scores = best_instance.get("keypoint_scores")
    if scores is None:
        scores = best_instance.get("keypoints_visible")
    scores_arr = np.asarray(scores, dtype=np.float32).reshape(-1) if scores is not None else None

    if keypoints.size == 0:
        return None

    if scores_arr is None or scores_arr.size != len(keypoints):
        scores_arr = np.ones((len(keypoints),), dtype=np.float32)

    return PoseResult(keypoints=keypoints, scores=scores_arr)


def update_detection_track_history(
    histories: Dict[int, DetectionTrackHistory],
    detection: Detection,
    frame_index: int,
    max_length: int,
) -> DetectionTrackHistory:
    history = histories.setdefault(detection.tracker_id, DetectionTrackHistory())
    if history.last_frame_index >= 0 and frame_index - history.last_frame_index > max_length:
        history.centers.clear()
        history.bbox_heights.clear()

    history.centers.append(detection.center.copy())
    history.bbox_heights.append(float(detection.bbox[3] - detection.bbox[1]))
    if len(history.centers) > max_length:
        history.centers = history.centers[-max_length:]
    if len(history.bbox_heights) > max_length:
        history.bbox_heights = history.bbox_heights[-max_length:]
    history.last_frame_index = frame_index
    return history


def is_static_background_detection(
    detection: Detection,
    history: DetectionTrackHistory,
    args: argparse.Namespace,
    is_small_candidate: bool,
) -> bool:
    if not is_small_candidate:
        return False
    if len(history.centers) < args.filter_static_history:
        return False

    centers = np.asarray(history.centers, dtype=np.float32)
    heights = np.asarray(history.bbox_heights, dtype=np.float32)
    if len(centers) < 2:
        return False

    mean_motion = float(np.linalg.norm(np.diff(centers, axis=0), axis=1).mean())
    height_jitter_ratio = float(heights.std() / max(heights.mean(), 1.0)) if len(heights) else 0.0
    return (
        mean_motion <= args.filter_static_motion_threshold
        and height_jitter_ratio <= args.filter_static_height_jitter_ratio
    )


def increment_filter_stats(filter_stats: Dict[str, int], reason: str) -> None:
    filter_stats[reason] = filter_stats.get(reason, 0) + 1


def prepare_detection_candidates(
    detections: List[Detection],
    frame: np.ndarray,
    frame_index: int,
    pose_runner: RTMPoseRunner,
    args: argparse.Namespace,
    track_histories: Dict[int, DetectionTrackHistory],
    filter_stats: Dict[str, int],
) -> List[DetectionCandidate]:
    frame_height, frame_width = frame.shape[:2]
    frame_area = max(1.0, float(frame_height * frame_width))
    frame_max_bbox_height = max(
        (float(det.bbox[3] - det.bbox[1]) for det in detections),
        default=1.0,
    )
    frame_max_bbox_area = max((float(det.area) for det in detections), default=1.0)
    candidates: List[DetectionCandidate] = []

    for detection in detections:
        increment_filter_stats(filter_stats, "input_detections")
        history = update_detection_track_history(
            histories=track_histories,
            detection=detection,
            frame_index=frame_index,
            max_length=max(args.filter_static_history, 2),
        )

        bbox_width = float(detection.bbox[2] - detection.bbox[0])
        bbox_height = float(detection.bbox[3] - detection.bbox[1])
        aspect_ratio = bbox_width / max(bbox_height, 1.0)
        height_ratio = bbox_height / max(1.0, float(frame_height))
        area_ratio = detection.area / frame_area
        relative_height_ratio = bbox_height / max(frame_max_bbox_height, 1.0)
        relative_area_ratio = detection.area / max(frame_max_bbox_area, 1.0)
        is_tiny_absolute = (
            height_ratio < args.filter_min_bbox_height_ratio
            or area_ratio < args.filter_min_bbox_area_ratio
        )
        is_small_relative = (
            relative_height_ratio < args.filter_relative_min_bbox_height_ratio
            or relative_area_ratio < args.filter_relative_min_bbox_area_ratio
        )
        is_small_candidate = is_tiny_absolute or is_small_relative

        rejected = False
        if (
            aspect_ratio < args.filter_min_bbox_aspect_ratio
            or aspect_ratio > args.filter_max_bbox_aspect_ratio
        ):
            increment_filter_stats(filter_stats, "reject_bad_aspect")
            rejected = True
        if is_static_background_detection(detection, history, args, is_small_candidate):
            increment_filter_stats(filter_stats, "reject_static_background")
            rejected = True
        if rejected:
            continue

        crop_x1, crop_y1, crop_x2, crop_y2 = expand_bbox(
            detection.bbox,
            frame.shape,
            args.bbox_padding,
        )
        crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
        if crop.size == 0:
            increment_filter_stats(filter_stats, "reject_empty_crop")
            continue

        pose_result = pose_runner.predict(crop, frame_index, f"candidate_{detection.tracker_id}")
        if pose_result is None:
            increment_filter_stats(filter_stats, "reject_no_pose")
            continue

        pose_mean_score = float(np.mean(pose_result.scores)) if pose_result.scores.size else 0.0
        confident_keypoint_count = int(
            np.count_nonzero(pose_result.scores >= args.filter_pose_keypoint_score)
        )
        appearance_feature = compute_appearance_feature(
            crop,
            pose_result,
            pose_score_threshold=args.filter_pose_keypoint_score,
        )

        if pose_mean_score < args.filter_min_pose_mean_score and is_small_candidate:
            increment_filter_stats(filter_stats, "reject_low_pose_score")
            continue
        if confident_keypoint_count < args.filter_min_pose_keypoints and is_small_candidate:
            increment_filter_stats(filter_stats, "reject_sparse_pose")
            continue

        pose_result.keypoints[:, 0] += crop_x1
        pose_result.keypoints[:, 1] += crop_y1
        candidates.append(
            DetectionCandidate(
                detection=detection,
                pose_crop_bbox=(crop_x1, crop_y1, crop_x2, crop_y2),
                pose_result=pose_result,
                pose_mean_score=pose_mean_score,
                confident_keypoint_count=confident_keypoint_count,
                appearance_feature=appearance_feature,
            )
        )
        increment_filter_stats(filter_stats, "kept_candidates")

    return candidates


def extract_detections(result) -> List[Detection]:
    if result.boxes is None:
        return []

    boxes = result.boxes.xyxy.cpu().numpy()
    scores = result.boxes.conf.cpu().numpy() if result.boxes.conf is not None else np.ones((len(boxes),))
    ids = result.boxes.id.cpu().numpy() if result.boxes.id is not None else None
    if ids is None:
        return []

    detections = []
    for box, score, track_id in zip(boxes, scores, ids):
        x1, y1, x2, y2 = box.astype(np.float32)
        if x2 <= x1 or y2 <= y1:
            continue
        detections.append(
            Detection(
                tracker_id=int(track_id),
                bbox=np.array([x1, y1, x2, y2], dtype=np.float32),
                confidence=float(score),
            )
        )
    return detections


def update_track_probation_states(
    candidates: Sequence[DetectionCandidate],
    probation_states: Dict[int, TrackProbationState],
    frame_index: int,
) -> None:
    for candidate in candidates:
        track_id = candidate.detection.tracker_id
        state = probation_states.setdefault(track_id, TrackProbationState())
        if state.last_seen_frame == frame_index - 1:
            state.consecutive_frames += 1
        else:
            state.consecutive_frames = 1
        state.last_seen_frame = frame_index


def filter_candidates_by_probation(
    candidates: Sequence[DetectionCandidate],
    probation_states: Dict[int, TrackProbationState],
    role_states: Dict[str, RoleState],
    frame_index: int,
    args: argparse.Namespace,
    probation_stats: Dict[str, int],
) -> List[DetectionCandidate]:
    probation_frames = args.new_track_probation_frames
    if probation_frames <= 1:
        return list(candidates)

    if not all(state.initialized for state in role_states.values()):
        increment_filter_stats(probation_stats, "initialization_bypass")
        return list(candidates)

    active_track_ids = {
        state.tracker_id
        for state in role_states.values()
        if state.tracker_id is not None and state.missing_frames <= 1
    }
    filtered: List[DetectionCandidate] = []

    for candidate in candidates:
        track_id = candidate.detection.tracker_id
        if candidate.is_recovered or track_id in active_track_ids:
            filtered.append(candidate)
            continue

        state = probation_states.setdefault(track_id, TrackProbationState())
        if state.approved:
            filtered.append(candidate)
            continue

        required_frames = probation_frames
        weak_pose = (
            candidate.pose_mean_score < args.probation_low_pose_threshold
            or candidate.confident_keypoint_count < args.filter_min_pose_keypoints + 2
        )
        if weak_pose:
            required_frames += args.probation_extra_frames
            increment_filter_stats(probation_stats, "extended_low_pose")

        role_similarities = [
            appearance_similarity(role_state.appearance_feature, candidate.appearance_feature)
            for role_state in role_states.values()
            if role_state.appearance_feature is not None
        ]
        if role_similarities:
            best_similarity = max(role_similarities)
            if best_similarity < args.probation_min_appearance_similarity:
                required_frames += args.probation_extra_frames
                increment_filter_stats(probation_stats, "extended_low_similarity")

        if state.last_seen_frame == frame_index and state.consecutive_frames >= required_frames:
            state.approved = True
            increment_filter_stats(probation_stats, "approved")
            filtered.append(candidate)
            continue

        increment_filter_stats(probation_stats, "blocked")

    return filtered


def approve_assigned_track_candidates(
    assignments: Dict[str, Optional[DetectionCandidate]],
    probation_states: Dict[int, TrackProbationState],
) -> None:
    for candidate in assignments.values():
        if candidate is None:
            continue
        track_id = candidate.detection.tracker_id
        state = probation_states.setdefault(track_id, TrackProbationState())
        state.approved = True


def attempt_recovery_candidate(
    role_name: str,
    state_before: Optional[RoleState],
    current_states: Dict[str, RoleState],
    frame: np.ndarray,
    frame_index: int,
    pose_runner: RTMPoseRunner,
    args: argparse.Namespace,
    recovery_stats: Dict[str, int],
) -> Optional[DetectionCandidate]:
    increment_filter_stats(recovery_stats, "attempted")
    if state_before is None or not state_before.initialized:
        increment_filter_stats(recovery_stats, "skip_uninitialized")
        return None
    if state_before.last_bbox is None or state_before.last_center is None:
        increment_filter_stats(recovery_stats, "skip_no_history")
        return None

    gap_frames = int(state_before.missing_frames) + 1
    if gap_frames > args.recovery_max_gap:
        increment_filter_stats(recovery_stats, "skip_gap_too_large")
        return None

    predicted_bbox = predict_recovery_bbox(
        state=state_before,
        frame_shape=frame.shape,
        gap_frames=gap_frames,
        velocity_scale=args.recovery_velocity_scale,
    )
    if predicted_bbox is None:
        increment_filter_stats(recovery_stats, "skip_no_prediction")
        return None

    crop_x1, crop_y1, crop_x2, crop_y2 = expand_bbox(
        predicted_bbox,
        frame.shape,
        args.bbox_padding + args.recovery_extra_padding,
    )
    crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
    if crop.size == 0:
        increment_filter_stats(recovery_stats, "reject_empty_crop")
        return None

    pose_result = pose_runner.predict(crop, frame_index, f"{role_name}_recovery")
    if pose_result is None:
        increment_filter_stats(recovery_stats, "reject_no_pose")
        return None

    pose_mean_score = float(np.mean(pose_result.scores)) if pose_result.scores.size else 0.0
    confident_keypoint_count = int(
        np.count_nonzero(pose_result.scores >= args.filter_pose_keypoint_score)
    )
    if pose_mean_score < args.recovery_min_pose_mean_score:
        increment_filter_stats(recovery_stats, "reject_low_pose_score")
        return None
    if confident_keypoint_count < args.recovery_min_keypoints:
        increment_filter_stats(recovery_stats, "reject_sparse_pose")
        return None

    pose_result.keypoints[:, 0] += crop_x1
    pose_result.keypoints[:, 1] += crop_y1

    valid_mask = pose_result.scores >= args.filter_pose_keypoint_score
    valid_points = pose_result.keypoints[valid_mask]
    if len(valid_points) == 0:
        valid_points = pose_result.keypoints[np.isfinite(pose_result.keypoints).all(axis=1)]
    if len(valid_points) == 0:
        increment_filter_stats(recovery_stats, "reject_invalid_pose")
        return None

    pose_center = valid_points.mean(axis=0)
    bbox_center = np.array(
        [
            0.5 * (predicted_bbox[0] + predicted_bbox[2]),
            0.5 * (predicted_bbox[1] + predicted_bbox[3]),
        ],
        dtype=np.float32,
    )
    center_distance_ratio = float(np.linalg.norm(pose_center - bbox_center)) / max(
        math.hypot(
            float(predicted_bbox[2] - predicted_bbox[0]),
            float(predicted_bbox[3] - predicted_bbox[1]),
        ),
        1.0,
    )
    if center_distance_ratio > args.recovery_max_center_distance_ratio:
        increment_filter_stats(recovery_stats, "reject_off_target_pose")
        return None

    local_pose = PoseResult(
        keypoints=pose_result.keypoints.copy(),
        scores=pose_result.scores.copy(),
    )
    local_pose.keypoints[:, 0] -= crop_x1
    local_pose.keypoints[:, 1] -= crop_y1
    appearance_feature = compute_appearance_feature(
        crop,
        local_pose,
        pose_score_threshold=args.filter_pose_keypoint_score,
    )

    target_state = current_states.get(role_name)
    target_similarity = appearance_similarity(
        target_state.appearance_feature if target_state is not None else None,
        appearance_feature,
    )
    other_best_similarity = -1.0
    for other_role_name, other_state in current_states.items():
        if other_role_name == role_name:
            continue
        other_best_similarity = max(
            other_best_similarity,
            appearance_similarity(other_state.appearance_feature, appearance_feature),
        )

    if target_state is not None and target_state.appearance_feature is not None:
        if target_similarity < args.recovery_min_appearance_similarity:
            increment_filter_stats(recovery_stats, "reject_appearance")
            return None
        if other_best_similarity > target_similarity + args.recovery_appearance_margin:
            increment_filter_stats(recovery_stats, "reject_other_role_match")
            return None

    tracker_id = state_before.tracker_id if state_before.tracker_id is not None else -1
    if tracker_id >= 0:
        for other_role_name, other_state in current_states.items():
            if other_role_name == role_name:
                continue
            if (
                other_state.tracker_id == tracker_id
                and other_state.last_seen_frame == frame_index
                and other_state.missing_frames == 0
            ):
                increment_filter_stats(recovery_stats, "reject_duplicate_tracker")
                return None
    detection = Detection(
        tracker_id=int(tracker_id),
        bbox=predicted_bbox,
        confidence=max(0.15, 0.35 - 0.02 * float(gap_frames - 1)),
    )
    increment_filter_stats(recovery_stats, "recovered")
    return DetectionCandidate(
        detection=detection,
        pose_crop_bbox=(crop_x1, crop_y1, crop_x2, crop_y2),
        pose_result=pose_result,
        pose_mean_score=pose_mean_score,
        confident_keypoint_count=confident_keypoint_count,
        appearance_feature=appearance_feature,
        is_recovered=True,
    )


def make_record(
    frame_index: int,
    role: str,
    state_before: Optional[RoleState],
    detection: Optional[Detection],
    pose_crop_bbox: Optional[Tuple[int, int, int, int]],
    pose_result: Optional[PoseResult],
    is_recovered: bool = False,
) -> FrameRoleRecord:
    track_id_changed = False
    if state_before is not None and detection is not None:
        track_id_changed = state_before.tracker_id is not None and state_before.tracker_id != detection.tracker_id

    record = FrameRoleRecord(
        frame_index=frame_index,
        role=role,
        visible=detection is not None and pose_result is not None,
        tracker_id=detection.tracker_id if detection is not None else None,
        tracker_bbox=detection.bbox.round(2).tolist() if detection is not None else None,
        pose_crop_bbox=list(pose_crop_bbox) if pose_crop_bbox is not None else None,
        source_track_id_changed=track_id_changed,
        is_recovered=is_recovered,
    )
    if pose_result is not None:
        record.raw_keypoints = pose_result.keypoints.round(3).tolist()
        record.raw_scores = pose_result.scores.round(5).tolist()
    return record


def interpolate_short_gaps(
    keypoints: np.ndarray,
    valid_mask: np.ndarray,
    max_gap: int,
) -> Tuple[np.ndarray, np.ndarray]:
    smoothed = keypoints.copy()
    smoothed_valid = valid_mask.copy()
    num_frames, num_keypoints, _ = smoothed.shape

    for keypoint_idx in range(num_keypoints):
        frame_idx = 0
        while frame_idx < num_frames:
            if smoothed_valid[frame_idx, keypoint_idx]:
                frame_idx += 1
                continue

            gap_start = frame_idx
            while frame_idx < num_frames and not smoothed_valid[frame_idx, keypoint_idx]:
                frame_idx += 1
            gap_end = frame_idx
            gap_length = gap_end - gap_start

            left_idx = gap_start - 1
            right_idx = gap_end

            if (
                gap_length <= max_gap
                and left_idx >= 0
                and right_idx < num_frames
                and smoothed_valid[left_idx, keypoint_idx]
                and smoothed_valid[right_idx, keypoint_idx]
            ):
                left_point = smoothed[left_idx, keypoint_idx]
                right_point = smoothed[right_idx, keypoint_idx]
                for offset, target_idx in enumerate(range(gap_start, gap_end), start=1):
                    ratio = offset / (gap_length + 1)
                    smoothed[target_idx, keypoint_idx] = left_point * (1.0 - ratio) + right_point * ratio
                    smoothed_valid[target_idx, keypoint_idx] = True

    return smoothed, smoothed_valid


def apply_ema_smoothing(
    keypoints: np.ndarray,
    valid_mask: np.ndarray,
    alpha: float,
) -> np.ndarray:
    smoothed = keypoints.copy()
    num_frames, num_keypoints, _ = smoothed.shape

    for keypoint_idx in range(num_keypoints):
        previous: Optional[np.ndarray] = None
        for frame_idx in range(num_frames):
            if not valid_mask[frame_idx, keypoint_idx]:
                continue
            current = smoothed[frame_idx, keypoint_idx]
            if previous is None:
                previous = current.copy()
            else:
                previous = alpha * current + (1.0 - alpha) * previous
                smoothed[frame_idx, keypoint_idx] = previous

    return smoothed


def postprocess_records(
    records_by_role: Dict[str, List[FrameRoleRecord]],
    min_keypoint_score: float,
    interpolate_gap: int,
    ema_alpha: float,
) -> int:
    keypoint_count = 0

    for role, records in records_by_role.items():
        valid_records = [record for record in records if record.raw_keypoints is not None]
        if not valid_records:
            continue

        keypoint_count = len(valid_records[0].raw_keypoints)
        frame_count = len(records)
        raw = np.full((frame_count, keypoint_count, 2), np.nan, dtype=np.float32)
        valid = np.zeros((frame_count, keypoint_count), dtype=bool)

        for record in records:
            if record.raw_keypoints is None or record.raw_scores is None:
                continue
            keypoints = np.asarray(record.raw_keypoints, dtype=np.float32)
            scores = np.asarray(record.raw_scores, dtype=np.float32)
            if len(keypoints) != keypoint_count:
                continue
            raw[record.frame_index] = keypoints
            valid[record.frame_index] = scores >= min_keypoint_score

        interpolated, interpolated_valid = interpolate_short_gaps(raw, valid, interpolate_gap)
        smoothed = apply_ema_smoothing(interpolated, interpolated_valid, ema_alpha)

        for record in records:
            if not interpolated_valid[record.frame_index].any():
                record.smoothed_keypoints = None
                continue

            role_points = []
            for kp_idx in range(keypoint_count):
                if interpolated_valid[record.frame_index, kp_idx]:
                    role_points.append(smoothed[record.frame_index, kp_idx].round(3).tolist())
                else:
                    role_points.append([None, None])
            record.smoothed_keypoints = role_points

    return keypoint_count


def draw_overlay(
    frame: np.ndarray,
    records_for_frame: Iterable[FrameRoleRecord],
) -> np.ndarray:
    canvas = frame.copy()
    for record in records_for_frame:
        color = ROLE_COLORS.get(record.role, (255, 255, 255))

        if record.tracker_bbox is not None:
            x1, y1, x2, y2 = map(int, record.tracker_bbox)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
            label = record.role
            if record.tracker_id is not None:
                label += f" | track {record.tracker_id}"
            if record.is_recovered:
                label += " | recovered"
            cv2.putText(
                canvas,
                label,
                (x1, max(20, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )

        points = record.smoothed_keypoints or record.raw_keypoints
        if points is None:
            continue

        for point in points:
            if point[0] is None or point[1] is None:
                continue
            px, py = int(round(point[0])), int(round(point[1]))
            cv2.circle(canvas, (px, py), 3, color, -1)

    return canvas


def save_records_json(
    output_root: Path,
    metadata: dict,
    records_by_role: Dict[str, List[FrameRoleRecord]],
) -> None:
    serializable_records = {
        role: [record.__dict__ for record in records]
        for role, records in records_by_role.items()
    }
    payload = {
        "metadata": metadata,
        "roles": serializable_records,
    }
    with (output_root / "pose_sequences.json").open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2)


def save_records_csv(
    output_root: Path,
    records_by_role: Dict[str, List[FrameRoleRecord]],
) -> None:
    csv_path = output_root / "pose_sequences.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        writer.writerow(
            [
                "frame",
                "role",
                "visible",
                "recovered",
                "tracker_id",
                "track_id_changed",
                "bbox_x1",
                "bbox_y1",
                "bbox_x2",
                "bbox_y2",
                "keypoint_index",
                "raw_x",
                "raw_y",
                "raw_score",
                "smoothed_x",
                "smoothed_y",
            ]
        )

        for role, records in records_by_role.items():
            for record in records:
                raw_points = record.raw_keypoints or []
                raw_scores = record.raw_scores or []
                smoothed_points = record.smoothed_keypoints or []
                count = max(len(raw_points), len(smoothed_points), len(raw_scores), 1)

                bbox = record.tracker_bbox or [None, None, None, None]
                for keypoint_index in range(count):
                    raw_point = raw_points[keypoint_index] if keypoint_index < len(raw_points) else [None, None]
                    raw_score = raw_scores[keypoint_index] if keypoint_index < len(raw_scores) else None
                    smoothed_point = (
                        smoothed_points[keypoint_index]
                        if keypoint_index < len(smoothed_points)
                        else [None, None]
                    )
                    writer.writerow(
                        [
                            record.frame_index,
                            role,
                            record.visible,
                            record.is_recovered,
                            record.tracker_id,
                            record.source_track_id_changed,
                            bbox[0],
                            bbox[1],
                            bbox[2],
                            bbox[3],
                            keypoint_index,
                            raw_point[0],
                            raw_point[1],
                            raw_score,
                            smoothed_point[0],
                            smoothed_point[1],
                        ]
                    )


def resolve_keypoint_names(count: int) -> List[str]:
    names = KEYPOINT_NAMES.get(count)
    if names is not None:
        return names
    return [f"kp_{idx}" for idx in range(count)]


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parent
    output_root = ensure_dir(Path(args.output_root).resolve())
    temp_pose_dir = ensure_dir(output_root / "_temp_pose_inputs")
    crops_dir = ensure_dir(output_root / "crops") if args.save_crops else None

    video_path = resolve_input_video_path(args.video_path, project_root)
    yolo_weights = resolve_yolo_weights_source(args.yolo_weights, project_root)
    device = select_device(args.device, args.pose_backend)
    yolo_device = select_yolo_device(device)
    tracker_yaml = build_tracker_yaml(args, output_root)

    if not video_path.exists():
        raise SystemExit(f"Video not found: {video_path}")

    model = YOLO(yolo_weights)
    pose_runner = RTMPoseRunner(
        pose_backend=args.pose_backend,
        pose_alias=args.pose_alias,
        pose_mode=args.pose_mode,
        pose_config=args.pose_config,
        pose_checkpoint=args.pose_checkpoint,
        device=device,
        temp_dir=temp_pose_dir,
        keep_temp_pose_inputs=args.keep_temp_pose_inputs,
    )
    role_manager = RoleLockManager(
        DEFAULT_ROLE_NAMES,
        args.max_missed_frames,
        appearance_alpha=args.appearance_alpha,
        appearance_weight=args.appearance_weight,
        appearance_min_pose_score=args.appearance_min_pose_score,
        anchor_appearance_alpha=args.anchor_appearance_alpha,
        anchor_appearance_weight=args.anchor_appearance_weight,
        anchor_min_pose_score=args.anchor_min_pose_score,
        interaction_iou_threshold=args.interaction_iou_threshold,
        interaction_center_distance_ratio=args.interaction_center_distance_ratio,
        interaction_cooldown_frames=args.interaction_cooldown_frames,
        interaction_side_prior_scale=args.interaction_side_prior_scale,
        interaction_anchor_motion_weight=args.interaction_anchor_motion_weight,
        interaction_anchor_distance_weight=args.interaction_anchor_distance_weight,
        interaction_tracker_bonus_scale=args.interaction_tracker_bonus_scale,
        interaction_anchor_velocity_scale=args.interaction_anchor_velocity_scale,
        tracker_role_memory_frames=args.tracker_role_memory_frames,
        tracker_role_min_history=args.tracker_role_min_history,
        tracker_role_bonus=args.tracker_role_bonus,
        tracker_role_penalty=args.tracker_role_penalty,
        interaction_tracker_role_penalty_scale=args.interaction_tracker_role_penalty_scale,
    )
    detection_filter_enabled = not args.disable_detection_filter

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"Failed to open video: {video_path}")

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 0
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 0
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if fps and fps > 0 else 24.0

    writer = None
    if args.save_video:
        output_video_path = output_root / "debug_overlay.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            str(output_video_path),
            fourcc,
            fps,
            (frame_width, frame_height),
        )

    records_by_role: Dict[str, List[FrameRoleRecord]] = {
        role: [] for role in DEFAULT_ROLE_NAMES
    }
    detection_track_histories: Dict[int, DetectionTrackHistory] = {}
    detection_filter_stats: Dict[str, int] = {}
    track_probation_states: Dict[int, TrackProbationState] = {}

    frame_index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if args.max_frames is not None and frame_index >= args.max_frames:
            break

        state_snapshot = {
            role: RoleState(
                name=state.name,
                tracker_id=state.tracker_id,
                last_bbox=None if state.last_bbox is None else state.last_bbox.copy(),
                last_center=None if state.last_center is None else state.last_center.copy(),
                velocity=None if state.velocity is None else state.velocity.copy(),
                last_seen_frame=state.last_seen_frame,
                missing_frames=state.missing_frames,
                initialized=state.initialized,
                appearance_feature=None if state.appearance_feature is None else state.appearance_feature.copy(),
                appearance_updates=state.appearance_updates,
                anchor_appearance_feature=None
                if state.anchor_appearance_feature is None
                else state.anchor_appearance_feature.copy(),
                anchor_appearance_updates=state.anchor_appearance_updates,
            )
            for role, state in role_manager.states.items()
        }

        track_results = model.track(
            frame,
            persist=True,
            tracker=str(tracker_yaml),
            classes=[0],
            conf=args.yolo_conf,
            iou=args.yolo_iou,
            device=yolo_device,
            verbose=False,
        )
        result = track_results[0]
        detections = extract_detections(result)
        if detection_filter_enabled:
            candidates = prepare_detection_candidates(
                detections=detections,
                frame=frame,
                frame_index=frame_index,
                pose_runner=pose_runner,
                args=args,
                track_histories=detection_track_histories,
                filter_stats=detection_filter_stats,
            )
        else:
            candidates = [
                DetectionCandidate(
                    detection=detection,
                    pose_crop_bbox=(0, 0, 0, 0),
                    pose_result=None,
                    pose_mean_score=0.0,
                    confident_keypoint_count=0,
                    appearance_feature=None,
                )
                for detection in detections
            ]

        update_track_probation_states(candidates, track_probation_states, frame_index)
        probation_stats: Dict[str, int] = {}
        probation_candidates = filter_candidates_by_probation(
            candidates=candidates,
            probation_states=track_probation_states,
            role_states=role_manager.states,
            frame_index=frame_index,
            args=args,
            probation_stats=probation_stats,
        )

        assignments = role_manager.assign(probation_candidates, frame_index, frame.shape)
        recovery_stats: Dict[str, int] = {}
        for role_name in DEFAULT_ROLE_NAMES:
            if assignments[role_name] is not None:
                continue
            recovered_candidate = attempt_recovery_candidate(
                role_name=role_name,
                state_before=state_snapshot.get(role_name),
                current_states=role_manager.states,
                frame=frame,
                frame_index=frame_index,
                pose_runner=pose_runner,
                args=args,
                recovery_stats=recovery_stats,
            )
            if recovered_candidate is None:
                continue
            assignments[role_name] = recovered_candidate
            role_manager.apply_recovered_candidate(role_name, recovered_candidate, frame_index)

        approve_assigned_track_candidates(assignments, track_probation_states)
        frame_records: List[FrameRoleRecord] = []

        for role_name in DEFAULT_ROLE_NAMES:
            assigned_candidate = assignments[role_name]
            detection = assigned_candidate.detection if assigned_candidate is not None else None
            pose_crop_bbox = None
            pose_result = None

            if detection is not None:
                if detection_filter_enabled:
                    if assigned_candidate is not None:
                        pose_crop_bbox = assigned_candidate.pose_crop_bbox
                        pose_result = assigned_candidate.pose_result
                else:
                    crop_x1, crop_y1, crop_x2, crop_y2 = expand_bbox(
                        detection.bbox,
                        frame.shape,
                        args.bbox_padding,
                    )
                    pose_crop_bbox = (crop_x1, crop_y1, crop_x2, crop_y2)
                    crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]

                    if crop.size > 0:
                        pose_result = pose_runner.predict(crop, frame_index, role_name)
                        if pose_result is not None:
                            pose_result.keypoints[:, 0] += crop_x1
                            pose_result.keypoints[:, 1] += crop_y1

                if crops_dir is not None and pose_crop_bbox is not None:
                    crop_x1, crop_y1, crop_x2, crop_y2 = pose_crop_bbox
                    crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
                    if crop.size > 0:
                        role_dir = ensure_dir(crops_dir / role_name)
                        cv2.imwrite(str(role_dir / f"{frame_index:06d}.jpg"), crop)

            record = make_record(
                frame_index=frame_index,
                role=role_name,
                state_before=state_snapshot.get(role_name),
                detection=detection,
                pose_crop_bbox=pose_crop_bbox,
                pose_result=pose_result,
                is_recovered=assigned_candidate.is_recovered if assigned_candidate is not None else False,
            )
            records_by_role[role_name].append(record)
            frame_records.append(record)

        stop_requested = False
        if writer is not None or args.display:
            vis_frame = draw_overlay(frame, frame_records)
            if writer is not None:
                writer.write(vis_frame)
            if args.display:
                cv2.imshow("two_character_rtmpose_pipeline", vis_frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    stop_requested = True

        frame_index += 1
        for key, value in probation_stats.items():
            detection_filter_stats[f"probation_{key}"] = detection_filter_stats.get(f"probation_{key}", 0) + value
        for key, value in recovery_stats.items():
            detection_filter_stats[f"recovery_{key}"] = detection_filter_stats.get(f"recovery_{key}", 0) + value
        if stop_requested:
            break

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()

    keypoint_count = postprocess_records(
        records_by_role=records_by_role,
        min_keypoint_score=args.min_keypoint_score,
        interpolate_gap=args.interpolate_gap,
        ema_alpha=args.ema_alpha,
    )

    metadata = {
        "video_path": str(video_path),
        "project_root": str(project_root),
        "yolo_weights": str(yolo_weights),
        "yolo_device": yolo_device,
        "pose_backend": args.pose_backend,
        "pose_alias": args.pose_alias,
        "pose_mode": args.pose_mode,
        "pose_config": args.pose_config,
        "pose_checkpoint": args.pose_checkpoint,
        "pose_source": pose_runner.pose_source,
        "pose_input_size": pose_runner.pose_input_size,
        "device": device,
        "frame_width": frame_width,
        "frame_height": frame_height,
        "fps": fps,
        "processed_frames": len(records_by_role[DEFAULT_ROLE_NAMES[0]]),
        "role_names": list(DEFAULT_ROLE_NAMES),
        "keypoint_count": keypoint_count,
        "keypoint_names": resolve_keypoint_names(keypoint_count),
        "tracker_yaml": str(tracker_yaml),
        "settings": {
            "tracking_preset": args.tracking_preset,
            "bbox_padding": args.bbox_padding,
            "yolo_conf": args.yolo_conf,
            "yolo_iou": args.yolo_iou,
            "min_keypoint_score": args.min_keypoint_score,
            "ema_alpha": args.ema_alpha,
            "interpolate_gap": args.interpolate_gap,
            "max_missed_frames": args.max_missed_frames,
            "detection_filter_enabled": detection_filter_enabled,
            "filter_min_bbox_height_ratio": args.filter_min_bbox_height_ratio,
            "filter_min_bbox_area_ratio": args.filter_min_bbox_area_ratio,
            "filter_relative_min_bbox_height_ratio": args.filter_relative_min_bbox_height_ratio,
            "filter_relative_min_bbox_area_ratio": args.filter_relative_min_bbox_area_ratio,
            "filter_min_bbox_aspect_ratio": args.filter_min_bbox_aspect_ratio,
            "filter_max_bbox_aspect_ratio": args.filter_max_bbox_aspect_ratio,
            "filter_min_pose_mean_score": args.filter_min_pose_mean_score,
            "filter_pose_keypoint_score": args.filter_pose_keypoint_score,
            "filter_min_pose_keypoints": args.filter_min_pose_keypoints,
            "filter_static_history": args.filter_static_history,
            "filter_static_motion_threshold": args.filter_static_motion_threshold,
            "filter_static_height_jitter_ratio": args.filter_static_height_jitter_ratio,
            "appearance_alpha": args.appearance_alpha,
            "appearance_weight": args.appearance_weight,
            "appearance_min_pose_score": args.appearance_min_pose_score,
            "anchor_appearance_alpha": args.anchor_appearance_alpha,
            "anchor_appearance_weight": args.anchor_appearance_weight,
            "anchor_min_pose_score": args.anchor_min_pose_score,
            "interaction_iou_threshold": args.interaction_iou_threshold,
            "interaction_center_distance_ratio": args.interaction_center_distance_ratio,
            "interaction_cooldown_frames": args.interaction_cooldown_frames,
            "interaction_side_prior_scale": args.interaction_side_prior_scale,
            "interaction_anchor_motion_weight": args.interaction_anchor_motion_weight,
            "interaction_anchor_distance_weight": args.interaction_anchor_distance_weight,
            "interaction_tracker_bonus_scale": args.interaction_tracker_bonus_scale,
            "interaction_anchor_velocity_scale": args.interaction_anchor_velocity_scale,
            "tracker_role_memory_frames": args.tracker_role_memory_frames,
            "tracker_role_min_history": args.tracker_role_min_history,
            "tracker_role_bonus": args.tracker_role_bonus,
            "tracker_role_penalty": args.tracker_role_penalty,
            "interaction_tracker_role_penalty_scale": args.interaction_tracker_role_penalty_scale,
            "recovery_max_gap": args.recovery_max_gap,
            "recovery_extra_padding": args.recovery_extra_padding,
            "recovery_velocity_scale": args.recovery_velocity_scale,
            "recovery_min_pose_mean_score": args.recovery_min_pose_mean_score,
            "recovery_min_keypoints": args.recovery_min_keypoints,
            "recovery_min_appearance_similarity": args.recovery_min_appearance_similarity,
            "recovery_appearance_margin": args.recovery_appearance_margin,
            "recovery_max_center_distance_ratio": args.recovery_max_center_distance_ratio,
            "new_track_probation_frames": args.new_track_probation_frames,
            "probation_extra_frames": args.probation_extra_frames,
            "probation_low_pose_threshold": args.probation_low_pose_threshold,
            "probation_min_appearance_similarity": args.probation_min_appearance_similarity,
        },
        "detection_filter_stats": dict(sorted(detection_filter_stats.items())),
    }

    with (output_root / "metadata.json").open("w", encoding="utf-8") as fp:
        json.dump(metadata, fp, indent=2)

    save_records_json(output_root, metadata, records_by_role)
    save_records_csv(output_root, records_by_role)

    if not args.keep_temp_pose_inputs and temp_pose_dir.exists():
        try:
            temp_pose_dir.rmdir()
        except OSError:
            pass

    print(f"Finished. Output saved to: {output_root}")


if __name__ == "__main__":
    main()
