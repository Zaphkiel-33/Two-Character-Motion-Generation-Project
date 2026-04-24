#!/usr/bin/env python3
"""3D reconstruction stage for the two-character motion pipeline.

This module follows the WHAM-oriented staging described in the project report:

1. keep the existing detection/tracking/2D-pose frontend as the identity source
2. convert frontend output into a stable sequence schema
3. reconstruct per-role 3D motion on the shared original-video timeline
4. run lightweight dual-interaction refinement and quality metrics
5. export WHAM-compatible observations for a future official-WHAM backend

The default backend is a local weak-perspective 2.5D lifter. It is intentionally
dependency-light so the 3D stage is runnable before SMPL/WHAM licensed assets are
available. The output schema mirrors the WHAM fields the next stage should use.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    import joblib
except Exception:  # pragma: no cover - optional WHAM export dependency
    joblib = None


ROLE_NAMES = ("character_A", "character_B")

COCO17_NAMES = [
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
]

HALPE26_EXTRA_NAMES = [
    "head",
    "neck",
    "hip",
    "left_big_toe",
    "right_big_toe",
    "left_small_toe",
    "right_small_toe",
    "left_heel",
    "right_heel",
]


SKELETON_17 = [
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
]

SKELETON_26 = [
    (17, 0),
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (17, 18),
    (18, 5),
    (18, 6),
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (18, 19),
    (19, 11),
    (19, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
    (15, 24),
    (15, 20),
    (15, 22),
    (16, 25),
    (16, 21),
    (16, 23),
]

# Approximate adult body segment lengths in meters. These are not SMPL
# parameters; they are only used to add stable depth hints to the local fallback.
BONE_LENGTHS_M = {
    (17, 18): 0.13,
    (18, 19): 0.50,
    (18, 5): 0.18,
    (18, 6): 0.18,
    (5, 7): 0.29,
    (7, 9): 0.27,
    (6, 8): 0.29,
    (8, 10): 0.27,
    (19, 11): 0.13,
    (19, 12): 0.13,
    (11, 13): 0.43,
    (13, 15): 0.43,
    (12, 14): 0.43,
    (14, 16): 0.43,
    (17, 0): 0.14,
    (0, 1): 0.05,
    (0, 2): 0.05,
    (1, 3): 0.08,
    (2, 4): 0.08,
    (15, 20): 0.16,
    (15, 22): 0.14,
    (15, 24): 0.08,
    (16, 21): 0.16,
    (16, 23): 0.14,
    (16, 25): 0.08,
    (5, 11): 0.48,
    (6, 12): 0.48,
}

LEFT_FOOT = (15, 20, 22, 24)
RIGHT_FOOT = (16, 21, 23, 25)
BODY_CORE = (5, 6, 11, 12, 18, 19)
HAND_INDICES = (9, 10)


@dataclass
class Role2DSequence:
    name: str
    keypoints2d: np.ndarray
    scores: np.ndarray
    bboxes: np.ndarray
    track_ids: np.ndarray
    visible: np.ndarray


@dataclass
class Role3DSequence:
    name: str
    joints_cam: np.ndarray
    joints_world: np.ndarray
    joints_world_smooth: np.ndarray
    joints_world_refined: np.ndarray
    trans_world: np.ndarray
    trans_world_smooth: np.ndarray
    trans_world_refined: np.ndarray
    poses_root_world: np.ndarray
    foot_contacts: np.ndarray
    valid: np.ndarray
    bone_reference_lengths: Dict[Tuple[int, int], float]


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parent
    default_output_root = project_root / "outputs" / "two_character_3d"

    parser = argparse.ArgumentParser(
        description="Reconstruct and refine two-character 3D motion from saved 2D pose sequences."
    )
    parser.add_argument(
        "--sequence-json",
        default=str(project_root / "outputs" / "two_character_rtmpose" / "pose_sequences.json"),
        help="2D sequence JSON produced by two_character_rtmpose_pipeline.py.",
    )
    parser.add_argument(
        "--metadata-json",
        default=str(project_root / "outputs" / "two_character_rtmpose" / "metadata.json"),
        help="Metadata JSON produced by two_character_rtmpose_pipeline.py.",
    )
    parser.add_argument("--output-root", default=str(default_output_root))
    parser.add_argument("--video-path", default=None, help="Optional source video path recorded in output metadata.")
    parser.add_argument("--use-smoothed", action="store_true", help="Use smoothed 2D keypoints when available.")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--min-score", type=float, default=0.2)
    parser.add_argument("--reconstruction-mode", choices=("legacy", "improved"), default="improved")
    parser.add_argument("--assumed-height-m", type=float, default=1.70)
    parser.add_argument("--focal-length", type=float, default=None)
    parser.add_argument("--depth-min-m", type=float, default=1.0)
    parser.add_argument("--depth-max-m", type=float, default=20.0)
    parser.add_argument("--depth-alpha", type=float, default=0.45)
    parser.add_argument("--smooth-alpha", type=float, default=0.35)
    parser.add_argument("--foot-ground-threshold", type=float, default=0.08)
    parser.add_argument("--foot-contact-speed", type=float, default=0.25)
    parser.add_argument("--foot-lock-min-frames", type=int, default=3)
    parser.add_argument("--foot-lock-blend", type=float, default=0.85)
    parser.add_argument("--hand-contact-threshold", type=float, default=0.35)
    parser.add_argument("--hand-body-threshold", type=float, default=0.28)
    parser.add_argument("--collision-radius", type=float, default=0.38)
    parser.add_argument("--skip-wham-export", action="store_true")
    parser.add_argument(
        "--visibility-mode",
        choices=("legacy", "occlusion_aware"),
        default="occlusion_aware",
        help=(
            "How to mark frames as valid for 3D. "
            "'legacy' only trusts the original 2D visible flag; "
            "'occlusion_aware' also accepts smoothed-only recovered frames."
        ),
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_json(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"Required file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def keypoint_names(count: int) -> List[str]:
    if count == 17:
        return COCO17_NAMES
    if count == 26:
        return COCO17_NAMES + HALPE26_EXTRA_NAMES
    return [f"kp_{idx}" for idx in range(count)]


def finite_point(point: Sequence[object]) -> bool:
    return len(point) >= 2 and point[0] is not None and point[1] is not None


def array_to_json(value: np.ndarray | Sequence[float] | float | int | bool | None):
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return array_to_json(value.item())
        return [array_to_json(item) for item in value]
    if isinstance(value, (np.floating, float)):
        if not math.isfinite(float(value)):
            return None
        return float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (list, tuple)):
        return [array_to_json(item) for item in value]
    return value


def detect_keypoint_count(payload: dict, metadata: dict) -> int:
    if metadata.get("keypoint_count"):
        return int(metadata["keypoint_count"])
    for records in payload.get("roles", {}).values():
        for record in records:
            points = record.get("raw_keypoints") or record.get("smoothed_keypoints")
            if points:
                return len(points)
    raise SystemExit("Could not infer keypoint count from sequence JSON.")


def infer_frame_count(payload: dict, max_frames: Optional[int]) -> int:
    max_index = -1
    for records in payload.get("roles", {}).values():
        for record in records:
            max_index = max(max_index, int(record.get("frame_index", -1)))
    if max_index < 0:
        raise SystemExit("No frame records found in sequence JSON.")
    frame_count = max_index + 1
    return min(frame_count, max_frames) if max_frames is not None else frame_count


def resolve_record_points(record: dict, use_smoothed: bool) -> Optional[List[List[object]]]:
    if use_smoothed and record.get("smoothed_keypoints") is not None:
        return record.get("smoothed_keypoints")
    return record.get("raw_keypoints")


def build_role_2d_sequences(
    payload: dict,
    metadata: dict,
    use_smoothed: bool,
    min_score: float,
    max_frames: Optional[int],
    visibility_mode: str,
) -> Tuple[Dict[str, Role2DSequence], int, int]:
    keypoint_count = detect_keypoint_count(payload, metadata)
    frame_count = infer_frame_count(payload, max_frames)
    output: Dict[str, Role2DSequence] = {}

    for role_name, records in payload.get("roles", {}).items():
        keypoints = np.full((frame_count, keypoint_count, 2), np.nan, dtype=np.float32)
        scores = np.zeros((frame_count, keypoint_count), dtype=np.float32)
        bboxes = np.full((frame_count, 4), np.nan, dtype=np.float32)
        track_ids = np.full((frame_count,), -1, dtype=np.int32)
        visible = np.zeros((frame_count,), dtype=bool)

        for record in records:
            frame_index = int(record.get("frame_index", -1))
            if frame_index < 0 or frame_index >= frame_count:
                continue

            bbox = record.get("tracker_bbox")
            if bbox is not None and len(bbox) == 4:
                bboxes[frame_index] = np.asarray(bbox, dtype=np.float32)

            if record.get("tracker_id") is not None:
                track_ids[frame_index] = int(record["tracker_id"])

            points = resolve_record_points(record, use_smoothed)
            raw_scores = record.get("raw_scores") or []
            has_smoothed_points = use_smoothed and record.get("smoothed_keypoints") is not None
            if not points:
                continue

            valid_points = 0
            for idx, point in enumerate(points[:keypoint_count]):
                if not finite_point(point):
                    continue
                if idx < len(raw_scores) and raw_scores[idx] is not None:
                    score = float(raw_scores[idx])
                elif has_smoothed_points:
                    # Interpolated smoothed points do not carry original scores,
                    # so keep them just above the downstream minimum threshold.
                    score = float(min_score)
                else:
                    score = 1.0
                if score < min_score:
                    continue
                keypoints[frame_index, idx] = (float(point[0]), float(point[1]))
                scores[frame_index, idx] = score
                valid_points += 1

            raw_visible = bool(record.get("visible")) and valid_points >= 4
            smoothed_visible = has_smoothed_points and valid_points >= 4
            if visibility_mode == "legacy":
                visible[frame_index] = raw_visible
            else:
                visible[frame_index] = raw_visible or smoothed_visible

        output[role_name] = Role2DSequence(
            name=role_name,
            keypoints2d=keypoints,
            scores=scores,
            bboxes=bboxes,
            track_ids=track_ids,
            visible=visible,
        )

    for role_name in ROLE_NAMES:
        if role_name not in output:
            output[role_name] = Role2DSequence(
                name=role_name,
                keypoints2d=np.full((frame_count, keypoint_count, 2), np.nan, dtype=np.float32),
                scores=np.zeros((frame_count, keypoint_count), dtype=np.float32),
                bboxes=np.full((frame_count, 4), np.nan, dtype=np.float32),
                track_ids=np.full((frame_count,), -1, dtype=np.int32),
                visible=np.zeros((frame_count,), dtype=bool),
            )

    return output, frame_count, keypoint_count


def pelvis_2d(points: np.ndarray) -> Optional[np.ndarray]:
    if len(points) > 19 and np.isfinite(points[19]).all():
        return points[19].astype(np.float32)
    hip_points = [points[idx] for idx in (11, 12) if idx < len(points) and np.isfinite(points[idx]).all()]
    if hip_points:
        return np.mean(np.asarray(hip_points, dtype=np.float32), axis=0)
    finite = points[np.isfinite(points).all(axis=1)]
    if len(finite):
        return np.mean(finite.astype(np.float32), axis=0)
    return None


def pelvis_3d(points: np.ndarray) -> Optional[np.ndarray]:
    if len(points) > 19 and np.isfinite(points[19]).all():
        return points[19].astype(np.float32)
    hip_points = [points[idx] for idx in (11, 12) if idx < len(points) and np.isfinite(points[idx]).all()]
    if hip_points:
        return np.mean(np.asarray(hip_points, dtype=np.float32), axis=0)
    finite = points[np.isfinite(points).all(axis=1)]
    if len(finite):
        return np.mean(finite.astype(np.float32), axis=0)
    return None


def bbox_height_or_keypoint_span(bbox: np.ndarray, points: np.ndarray) -> float:
    if np.isfinite(bbox).all() and bbox[3] > bbox[1]:
        return float(bbox[3] - bbox[1])
    finite = points[np.isfinite(points).all(axis=1)]
    if len(finite) >= 2:
        return float(np.nanmax(finite[:, 1]) - np.nanmin(finite[:, 1]))
    return 0.0


def default_focal_length(metadata: dict) -> float:
    width = float(metadata.get("frame_width") or 1280.0)
    height = float(metadata.get("frame_height") or 720.0)
    return math.sqrt(width * width + height * height)


def project_points_to_camera(
    points2d: np.ndarray,
    depth_m: float,
    focal_length: float,
    cx: float,
    cy: float,
) -> np.ndarray:
    points3d = np.full((len(points2d), 3), np.nan, dtype=np.float32)
    finite = np.isfinite(points2d).all(axis=1)
    points3d[finite, 0] = (points2d[finite, 0] - cx) / focal_length * depth_m
    points3d[finite, 1] = -(points2d[finite, 1] - cy) / focal_length * depth_m
    points3d[finite, 2] = depth_m
    return points3d


def ordered_bones(keypoint_count: int) -> List[Tuple[int, int, float]]:
    if keypoint_count >= 26:
        bones = SKELETON_26
    else:
        bones = SKELETON_17
    return [
        (parent, child, BONE_LENGTHS_M.get((parent, child), 0.25))
        for parent, child in bones
        if parent < keypoint_count and child < keypoint_count
    ]


def bone_traversal(keypoint_count: int) -> List[Tuple[int, int]]:
    if keypoint_count >= 26:
        return [
            (19, 18),
            (18, 17),
            (17, 0),
            (0, 1),
            (1, 3),
            (0, 2),
            (2, 4),
            (18, 5),
            (5, 7),
            (7, 9),
            (18, 6),
            (6, 8),
            (8, 10),
            (19, 11),
            (11, 13),
            (13, 15),
            (15, 20),
            (15, 22),
            (15, 24),
            (19, 12),
            (12, 14),
            (14, 16),
            (16, 21),
            (16, 23),
            (16, 25),
        ]
    return [
        (11, 12),
        (11, 5),
        (5, 7),
        (7, 9),
        (5, 0),
        (0, 1),
        (1, 3),
        (0, 2),
        (2, 4),
        (12, 6),
        (6, 8),
        (8, 10),
        (11, 13),
        (13, 15),
        (12, 14),
        (14, 16),
    ]


def fallback_bone_length(parent: int, child: int) -> float:
    if (parent, child) in BONE_LENGTHS_M:
        return float(BONE_LENGTHS_M[parent, child])
    if (child, parent) in BONE_LENGTHS_M:
        return float(BONE_LENGTHS_M[child, parent])
    return 0.25


def default_depth_sign(child_idx: int) -> float:
    if child_idx in (7, 8, 9, 10, 0, 1, 2, 3, 4, 17):
        return -1.0
    if child_idx in (13, 14, 15, 16, 20, 21, 22, 23, 24, 25):
        return 1.0
    return 1.0


def add_limb_depth_offsets(
    points3d: np.ndarray,
    keypoint_count: int,
    previous_relative_z: Optional[np.ndarray],
) -> np.ndarray:
    adjusted = points3d.copy()
    root = pelvis_3d(adjusted)
    if root is None:
        return adjusted

    for parent, child, length_m in ordered_bones(keypoint_count):
        if not (np.isfinite(adjusted[parent]).all() and np.isfinite(adjusted[child]).all()):
            continue

        xy_dist = float(np.linalg.norm(adjusted[child, :2] - adjusted[parent, :2]))
        dz_abs = math.sqrt(max(length_m * length_m - xy_dist * xy_dist, 0.0))
        sign = default_depth_sign(child)

        if previous_relative_z is not None:
            previous_delta = float(previous_relative_z[child] - previous_relative_z[parent])
            if abs(previous_delta) > 1e-3:
                sign = 1.0 if previous_delta > 0 else -1.0

        adjusted[child, 2] = adjusted[parent, 2] + sign * dz_abs

    return adjusted


def weighted_median(values: Sequence[float], weights: Sequence[float]) -> float:
    pairs = sorted(zip(values, weights), key=lambda item: item[0])
    if not pairs:
        raise ValueError("weighted_median requires at least one value")
    total = sum(weight for _, weight in pairs)
    threshold = total * 0.5
    running = 0.0
    for value, weight in pairs:
        running += weight
        if running >= threshold:
            return float(value)
    return float(pairs[-1][0])


def smooth_scalar_series(values: np.ndarray, valid: np.ndarray, alpha: float) -> np.ndarray:
    output = np.full_like(values, np.nan, dtype=np.float32)
    last: Optional[float] = None
    for frame_index, value in enumerate(values):
        if valid[frame_index] and math.isfinite(float(value)):
            if last is None:
                smoothed = float(value)
            else:
                smoothed = float(alpha) * float(value) + (1.0 - float(alpha)) * last
            output[frame_index] = smoothed
            last = smoothed
        elif last is not None:
            output[frame_index] = last
    return output


def point_distance(points: np.ndarray, a: int, b: int) -> Optional[float]:
    if a >= len(points) or b >= len(points):
        return None
    if not (np.isfinite(points[a]).all() and np.isfinite(points[b]).all()):
        return None
    distance = float(np.linalg.norm(points[a] - points[b]))
    return distance if distance > 1e-6 else None


def estimate_depth_from_observations(
    points2d: np.ndarray,
    bbox: np.ndarray,
    focal_length: float,
    assumed_height_m: float,
    depth_min_m: float,
    depth_max_m: float,
) -> float:
    candidates: List[float] = []
    weights: List[float] = []

    body_span = bbox_height_or_keypoint_span(bbox, points2d)
    if body_span > 1.0:
        candidates.append(focal_length * assumed_height_m / body_span)
        weights.append(1.35)

    priors = [
        ((18, 19), 0.50, 1.15),
        ((5, 6), 0.36, 0.75),
        ((11, 12), 0.28, 0.65),
        ((5, 11), 0.48, 0.9),
        ((6, 12), 0.48, 0.9),
        ((11, 15), 0.86, 1.0),
        ((12, 16), 0.86, 1.0),
    ]
    for (a, b), body_length_m, weight in priors:
        pixel_length = point_distance(points2d, a, b)
        if pixel_length is None:
            continue
        candidates.append(focal_length * body_length_m / pixel_length)
        weights.append(weight)

    if not candidates:
        return float(np.clip(4.0, depth_min_m, depth_max_m))
    depth = weighted_median(candidates, weights)
    return float(np.clip(depth, depth_min_m, depth_max_m))


def estimate_reference_bone_lengths(
    joints: np.ndarray,
    valid: np.ndarray,
    keypoint_count: int,
) -> Dict[Tuple[int, int], float]:
    references: Dict[Tuple[int, int], float] = {}
    for parent, child in bone_traversal(keypoint_count):
        samples = []
        for frame_index in range(len(valid)):
            if not valid[frame_index]:
                continue
            frame = joints[frame_index]
            if parent >= len(frame) or child >= len(frame):
                continue
            if not (np.isfinite(frame[parent]).all() and np.isfinite(frame[child]).all()):
                continue
            length = float(np.linalg.norm(frame[child] - frame[parent]))
            canonical = fallback_bone_length(parent, child)
            if 0.35 * canonical <= length <= 2.75 * canonical:
                samples.append(length)
        if samples:
            references[parent, child] = float(np.median(np.asarray(samples, dtype=np.float32)))
        else:
            references[parent, child] = fallback_bone_length(parent, child)
    return references


def normalize_pose_bone_lengths(
    points3d: np.ndarray,
    keypoint_count: int,
    reference_lengths: Dict[Tuple[int, int], float],
) -> np.ndarray:
    normalized = points3d.copy()
    for parent, child in bone_traversal(keypoint_count):
        if parent >= len(normalized) or child >= len(normalized):
            continue
        if not (np.isfinite(normalized[parent]).all() and np.isfinite(normalized[child]).all()):
            continue
        direction = normalized[child] - normalized[parent]
        norm = float(np.linalg.norm(direction))
        if norm <= 1e-6:
            continue
        target_length = reference_lengths.get((parent, child), fallback_bone_length(parent, child))
        normalized[child] = normalized[parent] + direction / norm * target_length
    return normalized


def smooth_joints(values: np.ndarray, alpha: float) -> np.ndarray:
    output = np.full_like(values, np.nan)
    last = np.full(values.shape[1:], np.nan, dtype=np.float32)

    for frame_index in range(values.shape[0]):
        for joint_index in range(values.shape[1]):
            value = values[frame_index, joint_index]
            if np.isfinite(value).all():
                if np.isfinite(last[joint_index]).all():
                    smoothed = alpha * value + (1.0 - alpha) * last[joint_index]
                else:
                    smoothed = value
                output[frame_index, joint_index] = smoothed
                last[joint_index] = smoothed
            elif np.isfinite(last[joint_index]).all():
                output[frame_index, joint_index] = last[joint_index]

    return output


def smooth_vectors(values: np.ndarray, valid: np.ndarray, alpha: float) -> np.ndarray:
    output = np.full_like(values, np.nan)
    last: Optional[np.ndarray] = None
    for frame_index, value in enumerate(values):
        if valid[frame_index] and np.isfinite(value).all():
            if last is None:
                smoothed = value
            else:
                smoothed = alpha * value + (1.0 - alpha) * last
            output[frame_index] = smoothed
            last = smoothed
        elif last is not None:
            output[frame_index] = last
    return output


def compute_root_yaw(joints: np.ndarray, valid: np.ndarray) -> np.ndarray:
    yaw = np.zeros((len(joints),), dtype=np.float32)
    last_yaw = 0.0
    for frame_index, frame in enumerate(joints):
        if not valid[frame_index]:
            yaw[frame_index] = last_yaw
            continue

        vectors = []
        for left, right in ((5, 6), (11, 12)):
            if right < len(frame) and np.isfinite(frame[left]).all() and np.isfinite(frame[right]).all():
                vectors.append(frame[right] - frame[left])

        if vectors:
            right_vec = np.mean(np.asarray(vectors), axis=0)
            horizontal_norm = float(np.linalg.norm(right_vec[[0, 2]]))
            if horizontal_norm > 1e-6:
                last_yaw = float(math.atan2(right_vec[2], right_vec[0]))

        yaw[frame_index] = last_yaw

    return np.unwrap(yaw).astype(np.float32)


def reconstruct_role_3d(
    seq: Role2DSequence,
    metadata: dict,
    keypoint_count: int,
    args: argparse.Namespace,
) -> Role3DSequence:
    frame_width = float(metadata.get("frame_width") or 1280.0)
    frame_height = float(metadata.get("frame_height") or 720.0)
    cx = frame_width * 0.5
    cy = frame_height * 0.5
    focal_length = float(args.focal_length or default_focal_length(metadata))

    frame_count = len(seq.visible)
    joints_cam = np.full((frame_count, keypoint_count, 3), np.nan, dtype=np.float32)
    trans_world = np.full((frame_count, 3), np.nan, dtype=np.float32)
    valid = np.zeros((frame_count,), dtype=bool)
    depth_estimates = np.full((frame_count,), np.nan, dtype=np.float32)
    depth_valid = np.zeros((frame_count,), dtype=bool)

    for frame_index in range(frame_count):
        points2d = seq.keypoints2d[frame_index]
        root2d = pelvis_2d(points2d)
        if not seq.visible[frame_index] or root2d is None:
            continue

        if args.reconstruction_mode == "legacy":
            height_px = bbox_height_or_keypoint_span(seq.bboxes[frame_index], points2d)
            if height_px > 1.0:
                depth_estimates[frame_index] = float(
                    np.clip(
                        focal_length * float(args.assumed_height_m) / height_px,
                        args.depth_min_m,
                        args.depth_max_m,
                    )
                )
                depth_valid[frame_index] = True
        else:
            depth_estimates[frame_index] = estimate_depth_from_observations(
                points2d=points2d,
                bbox=seq.bboxes[frame_index],
                focal_length=focal_length,
                assumed_height_m=float(args.assumed_height_m),
                depth_min_m=float(args.depth_min_m),
                depth_max_m=float(args.depth_max_m),
            )
            depth_valid[frame_index] = True

    if args.reconstruction_mode == "legacy":
        smoothed_depths = depth_estimates.copy()
    else:
        smoothed_depths = smooth_scalar_series(depth_estimates, depth_valid, args.depth_alpha)

    origin: Optional[np.ndarray] = None
    previous_depth = 4.0
    previous_relative_z: Optional[np.ndarray] = None

    for frame_index in range(frame_count):
        points2d = seq.keypoints2d[frame_index]
        root2d = pelvis_2d(points2d)
        if not seq.visible[frame_index] or root2d is None:
            continue

        if depth_valid[frame_index] and math.isfinite(float(smoothed_depths[frame_index])):
            depth = float(smoothed_depths[frame_index])
            previous_depth = depth
        else:
            depth = previous_depth

        points3d = project_points_to_camera(points2d, depth, focal_length, cx, cy)
        points3d = add_limb_depth_offsets(points3d, keypoint_count, previous_relative_z)
        root3d = pelvis_3d(points3d)
        if root3d is None:
            continue

        if origin is None:
            origin = root3d.copy()

        joints_cam[frame_index] = points3d
        trans_world[frame_index] = root3d - origin
        valid[frame_index] = True

        previous_relative_z = points3d[:, 2] - root3d[2]
        previous_relative_z[~np.isfinite(previous_relative_z)] = 0.0

    bone_reference_lengths = estimate_reference_bone_lengths(joints_cam, valid, keypoint_count)
    if args.reconstruction_mode == "improved":
        for frame_index in range(frame_count):
            if not valid[frame_index]:
                continue
            root_before = pelvis_3d(joints_cam[frame_index])
            normalized = normalize_pose_bone_lengths(
                joints_cam[frame_index],
                keypoint_count=keypoint_count,
                reference_lengths=bone_reference_lengths,
            )
            root_after = pelvis_3d(normalized)
            if root_before is not None and root_after is not None:
                normalized += (root_before - root_after)
            joints_cam[frame_index] = normalized

    joints_world = joints_cam.copy()
    if origin is not None:
        joints_world = joints_cam - origin.reshape(1, 1, 3)

    trans_world_smooth = smooth_vectors(trans_world, valid, args.smooth_alpha)
    joints_world_smooth = smooth_joints(joints_world, args.smooth_alpha)
    yaw = compute_root_yaw(joints_world_smooth, valid)
    poses_root_world = np.zeros((frame_count, 3), dtype=np.float32)
    poses_root_world[:, 1] = yaw

    foot_contacts = np.zeros((frame_count, 2), dtype=bool)

    return Role3DSequence(
        name=seq.name,
        joints_cam=joints_cam,
        joints_world=joints_world,
        joints_world_smooth=joints_world_smooth,
        joints_world_refined=joints_world_smooth.copy(),
        trans_world=trans_world,
        trans_world_smooth=trans_world_smooth,
        trans_world_refined=trans_world_smooth.copy(),
        poses_root_world=poses_root_world,
        foot_contacts=foot_contacts,
        valid=valid,
        bone_reference_lengths=bone_reference_lengths,
    )


def available_indices(indices: Iterable[int], keypoint_count: int) -> List[int]:
    return [idx for idx in indices if idx < keypoint_count]


def mean_joint(frame: np.ndarray, indices: Sequence[int]) -> Optional[np.ndarray]:
    points = [frame[idx] for idx in indices if idx < len(frame) and np.isfinite(frame[idx]).all()]
    if not points:
        return None
    return np.mean(np.asarray(points, dtype=np.float32), axis=0)


def foot_points(frame: np.ndarray, side: str) -> Optional[np.ndarray]:
    if side == "left":
        return mean_joint(frame, LEFT_FOOT)
    return mean_joint(frame, RIGHT_FOOT)


def horizontal_speed(points: np.ndarray, fps: float) -> np.ndarray:
    speeds = np.zeros((len(points),), dtype=np.float32)
    for frame_index in range(1, len(points)):
        if np.isfinite(points[frame_index]).all() and np.isfinite(points[frame_index - 1]).all():
            speeds[frame_index] = float(np.linalg.norm(points[frame_index, [0, 2]] - points[frame_index - 1, [0, 2]]) * fps)
    return speeds


def contiguous_true_segments(mask: np.ndarray) -> List[Tuple[int, int]]:
    segments: List[Tuple[int, int]] = []
    start: Optional[int] = None
    for index, value in enumerate(mask.astype(bool)):
        if value and start is None:
            start = index
        elif not value and start is not None:
            segments.append((start, index))
            start = None
    if start is not None:
        segments.append((start, len(mask)))
    return segments


def compute_foot_arrays_and_contacts(
    role: Role3DSequence,
    source_points: np.ndarray,
    ground_y: float,
    fps: float,
    args: argparse.Namespace,
) -> Dict[str, np.ndarray]:
    left = np.full((len(role.valid), 3), np.nan, dtype=np.float32)
    right = np.full_like(left, np.nan)
    for frame_index, frame in enumerate(source_points):
        if not role.valid[frame_index]:
            continue
        left_point = foot_points(frame, "left")
        right_point = foot_points(frame, "right")
        if left_point is not None:
            left[frame_index] = left_point
        if right_point is not None:
            right[frame_index] = right_point

    role.foot_contacts[:] = False
    speeds = {
        "left": horizontal_speed(left, fps),
        "right": horizontal_speed(right, fps),
    }
    for side_index, side in enumerate(("left", "right")):
        points = left if side == "left" else right
        near_ground = np.abs(points[:, 1] - ground_y) <= args.foot_ground_threshold
        slow = speeds[side] <= args.foot_contact_speed
        role.foot_contacts[:, side_index] = role.valid & np.isfinite(points).all(axis=1) & near_ground & slow

    return {"left": left, "right": right}


def apply_foot_lock_refinement(
    role: Role3DSequence,
    foot_arrays: Dict[str, np.ndarray],
    ground_y: float,
    args: argparse.Namespace,
) -> None:
    if args.reconstruction_mode != "improved":
        return

    delta_sum = np.zeros((len(role.valid), 3), dtype=np.float32)
    delta_count = np.zeros((len(role.valid),), dtype=np.float32)

    for side_index, side in enumerate(("left", "right")):
        contact_mask = role.foot_contacts[:, side_index]
        for start, end in contiguous_true_segments(contact_mask):
            if end - start < args.foot_lock_min_frames:
                continue
            segment_points = foot_arrays[side][start:end]
            finite_mask = np.isfinite(segment_points).all(axis=1)
            if int(finite_mask.sum()) < args.foot_lock_min_frames:
                continue
            anchor_xz = np.median(segment_points[finite_mask][:, [0, 2]], axis=0)
            for offset, frame_index in enumerate(range(start, end)):
                current = segment_points[offset]
                if not np.isfinite(current).all():
                    continue
                delta = np.array(
                    [
                        anchor_xz[0] - current[0],
                        ground_y - current[1],
                        anchor_xz[1] - current[2],
                    ],
                    dtype=np.float32,
                ) * float(args.foot_lock_blend)
                delta_sum[frame_index] += delta
                delta_count[frame_index] += 1.0

    locked_mask = delta_count > 0
    for frame_index in np.where(locked_mask)[0]:
        delta = delta_sum[frame_index] / delta_count[frame_index]
        role.joints_world_refined[frame_index] += delta
        role.trans_world_refined[frame_index] += delta


def detect_and_refine_interactions(
    roles_3d: Dict[str, Role3DSequence],
    keypoint_count: int,
    fps: float,
    args: argparse.Namespace,
) -> Tuple[Dict[str, object], Dict[str, object]]:
    foot_samples = []
    for role in roles_3d.values():
        for frame_index, frame in enumerate(role.joints_world_smooth):
            if not role.valid[frame_index]:
                continue
            for side in ("left", "right"):
                point = foot_points(frame, side)
                if point is not None:
                    foot_samples.append(point[1])

    ground_y = float(np.percentile(foot_samples, 5)) if foot_samples else 0.0
    contacts: Dict[str, object] = {
        "ground_y": ground_y,
        "hand_hand": [],
        "hand_body": [],
        "collision_corrections": [],
        "foot_lock_corrections": [],
    }

    for role in roles_3d.values():
        foot_arrays = compute_foot_arrays_and_contacts(
            role=role,
            source_points=role.joints_world_smooth,
            ground_y=ground_y,
            fps=fps,
            args=args,
        )

        for frame_index in range(len(role.valid)):
            contact_points = []
            for side_index, side in enumerate(("left", "right")):
                if role.foot_contacts[frame_index, side_index]:
                    point = foot_arrays[side][frame_index]
                    if np.isfinite(point).all():
                        contact_points.append(point)
            if not contact_points:
                continue
            correction_y = ground_y - float(np.mean(np.asarray(contact_points)[:, 1]))
            role.joints_world_refined[frame_index, :, 1] += correction_y
            role.trans_world_refined[frame_index, 1] += correction_y

    role_a = roles_3d.get(ROLE_NAMES[0])
    role_b = roles_3d.get(ROLE_NAMES[1])
    if role_a is not None and role_b is not None:
        for frame_index in range(len(role_a.valid)):
            if not (role_a.valid[frame_index] and role_b.valid[frame_index]):
                continue

            frame_a = role_a.joints_world_refined[frame_index]
            frame_b = role_b.joints_world_refined[frame_index]

            # Simple collision proxy: keep pelvis/root centers separated in the XZ plane.
            root_a = role_a.trans_world_refined[frame_index]
            root_b = role_b.trans_world_refined[frame_index]
            if np.isfinite(root_a).all() and np.isfinite(root_b).all():
                diff = root_b[[0, 2]] - root_a[[0, 2]]
                dist = float(np.linalg.norm(diff))
                if 1e-6 < dist < args.collision_radius:
                    push = (args.collision_radius - dist) * 0.5
                    direction = diff / dist
                    delta = np.array([direction[0] * push, 0.0, direction[1] * push], dtype=np.float32)
                    role_a.joints_world_refined[frame_index] -= delta
                    role_b.joints_world_refined[frame_index] += delta
                    role_a.trans_world_refined[frame_index] -= delta
                    role_b.trans_world_refined[frame_index] += delta
                    contacts["collision_corrections"].append(
                        {
                            "frame_index": frame_index,
                            "separation_before_m": dist,
                            "push_each_m": push,
                        }
                    )

    for role_name, role in roles_3d.items():
        refined_feet = compute_foot_arrays_and_contacts(
            role=role,
            source_points=role.joints_world_refined,
            ground_y=ground_y,
            fps=fps,
            args=args,
        )
        before = role.trans_world_refined.copy()
        apply_foot_lock_refinement(role, refined_feet, ground_y, args)
        locked_frames = np.where(np.linalg.norm(role.trans_world_refined - before, axis=1) > 1e-5)[0]
        for frame_index in locked_frames.tolist():
            contacts["foot_lock_corrections"].append(
                {
                    "frame_index": int(frame_index),
                    "role": role_name,
                    "delta": array_to_json(role.trans_world_refined[frame_index] - before[frame_index]),
                }
            )
        compute_foot_arrays_and_contacts(
            role=role,
            source_points=role.joints_world_refined,
            ground_y=ground_y,
            fps=fps,
            args=args,
        )

    if role_a is not None and role_b is not None:
        for frame_index in range(len(role_a.valid)):
            if not (role_a.valid[frame_index] and role_b.valid[frame_index]):
                continue

            frame_a = role_a.joints_world_refined[frame_index]
            frame_b = role_b.joints_world_refined[frame_index]

            for hand_a in available_indices(HAND_INDICES, keypoint_count):
                if not np.isfinite(frame_a[hand_a]).all():
                    continue
                for hand_b in available_indices(HAND_INDICES, keypoint_count):
                    if not np.isfinite(frame_b[hand_b]).all():
                        continue
                    distance = float(np.linalg.norm(frame_a[hand_a] - frame_b[hand_b]))
                    if distance <= args.hand_contact_threshold:
                        contacts["hand_hand"].append(
                            {
                                "frame_index": frame_index,
                                "role_a_joint": keypoint_names(keypoint_count)[hand_a],
                                "role_b_joint": keypoint_names(keypoint_count)[hand_b],
                                "distance_m": distance,
                            }
                        )

            core_a = mean_joint(frame_a, BODY_CORE)
            core_b = mean_joint(frame_b, BODY_CORE)
            for role_name, frame_src, other_core in (
                (ROLE_NAMES[0], frame_a, core_b),
                (ROLE_NAMES[1], frame_b, core_a),
            ):
                if other_core is None:
                    continue
                for hand_idx in available_indices(HAND_INDICES, keypoint_count):
                    if not np.isfinite(frame_src[hand_idx]).all():
                        continue
                    distance = float(np.linalg.norm(frame_src[hand_idx] - other_core))
                    if distance <= args.hand_body_threshold:
                        contacts["hand_body"].append(
                            {
                                "frame_index": frame_index,
                                "source_role": role_name,
                                "source_joint": keypoint_names(keypoint_count)[hand_idx],
                                "target": "other_body_core",
                                "distance_m": distance,
                            }
                        )

    metrics = compute_metrics(roles_3d, contacts, fps)
    return contacts, metrics


def compute_metrics(
    roles_3d: Dict[str, Role3DSequence],
    contacts: Dict[str, object],
    fps: float,
) -> Dict[str, object]:
    metrics: Dict[str, object] = {
        "fps": fps,
        "roles": {},
        "contacts": {
            "hand_hand_count": len(contacts.get("hand_hand", [])),
            "hand_body_count": len(contacts.get("hand_body", [])),
            "collision_correction_count": len(contacts.get("collision_corrections", [])),
            "foot_lock_correction_count": len(contacts.get("foot_lock_corrections", [])),
        },
    }

    for role_name, role in roles_3d.items():
        role_metrics: Dict[str, object] = {
            "valid_frames": int(role.valid.sum()),
            "valid_ratio": float(role.valid.mean()) if len(role.valid) else 0.0,
            "left_foot_contact_frames": int(role.foot_contacts[:, 0].sum()),
            "right_foot_contact_frames": int(role.foot_contacts[:, 1].sum()),
        }

        bone_errors = []
        for parent, child in role.bone_reference_lengths:
            reference = role.bone_reference_lengths[parent, child]
            for frame_index in range(len(role.valid)):
                if not role.valid[frame_index]:
                    continue
                frame = role.joints_world_refined[frame_index]
                if parent >= len(frame) or child >= len(frame):
                    continue
                if not (np.isfinite(frame[parent]).all() and np.isfinite(frame[child]).all()):
                    continue
                length = float(np.linalg.norm(frame[child] - frame[parent]))
                bone_errors.append(abs(length - reference))
        role_metrics["mean_bone_length_error_m"] = (
            float(np.mean(np.asarray(bone_errors, dtype=np.float32))) if bone_errors else 0.0
        )

        root_speeds = horizontal_speed(role.trans_world_refined, fps)
        valid_speeds = root_speeds[np.isfinite(root_speeds)]
        role_metrics["mean_root_speed_mps"] = float(valid_speeds.mean()) if len(valid_speeds) else 0.0

        for side_index, side in enumerate(("left", "right")):
            foot = np.full((len(role.valid), 3), np.nan, dtype=np.float32)
            for frame_index, frame in enumerate(role.joints_world_refined):
                point = foot_points(frame, side)
                if point is not None:
                    foot[frame_index] = point
            speeds = horizontal_speed(foot, fps)
            contact_speeds = speeds[role.foot_contacts[:, side_index]]
            role_metrics[f"{side}_foot_sliding_mean_mps"] = (
                float(contact_speeds.mean()) if len(contact_speeds) else 0.0
            )
            role_metrics[f"{side}_foot_sliding_max_mps"] = (
                float(contact_speeds.max()) if len(contact_speeds) else 0.0
            )

        metrics["roles"][role_name] = role_metrics

    return metrics


def wham_bbox_from_xyxy_or_keypoints(bbox: np.ndarray, points: np.ndarray) -> np.ndarray:
    if np.isfinite(bbox).all() and bbox[2] > bbox[0] and bbox[3] > bbox[1]:
        cx = float((bbox[0] + bbox[2]) * 0.5)
        cy = float((bbox[1] + bbox[3]) * 0.5)
        side = float(max(bbox[2] - bbox[0], bbox[3] - bbox[1]))
    else:
        finite = points[np.isfinite(points).all(axis=1)]
        if len(finite) == 0:
            return np.array([0.0, 0.0, 1.0], dtype=np.float32)
        x1, y1 = finite.min(axis=0)
        x2, y2 = finite.max(axis=0)
        cx = float((x1 + x2) * 0.5)
        cy = float((y1 + y2) * 0.5)
        side = float(max(x2 - x1, y2 - y1, 1.0))
    return np.array([cx, cy, side * 1.2 / 200.0], dtype=np.float32)


def export_wham_observations(
    roles_2d: Dict[str, Role2DSequence],
    output_root: Path,
    metadata: dict,
    min_score: float,
) -> Optional[Path]:
    if joblib is None:
        return None

    output = {}
    for role_idx, role_name in enumerate(ROLE_NAMES):
        seq = roles_2d[role_name]
        frame_ids = []
        bboxes = []
        keypoints = []
        for frame_index in range(len(seq.visible)):
            if not seq.visible[frame_index]:
                continue
            kp17 = np.full((17, 3), 0.0, dtype=np.float32)
            available = min(17, seq.keypoints2d.shape[1])
            kp17[:available, :2] = np.nan_to_num(seq.keypoints2d[frame_index, :available], nan=0.0)
            kp17[:available, 2] = seq.scores[frame_index, :available]
            if int((kp17[:, 2] >= min_score).sum()) < 4:
                continue
            frame_ids.append(frame_index)
            bboxes.append(wham_bbox_from_xyxy_or_keypoints(seq.bboxes[frame_index], seq.keypoints2d[frame_index]))
            keypoints.append(kp17)

        if not frame_ids:
            continue

        output[role_idx] = {
            "role": role_name,
            "frame_id": np.asarray(frame_ids, dtype=np.int32),
            "bbox": np.asarray(bboxes, dtype=np.float32),
            "keypoints": np.asarray(keypoints, dtype=np.float32),
        }

    path = output_root / "wham_tracking_observations.pth"
    joblib.dump(output, path)
    summary = {
        "path": str(path),
        "note": (
            "This contains bbox/keypoints/frame_id from this project's frontend. "
            "Official WHAM still needs its FeatureExtractor to add image features and initial SMPL estimates."
        ),
        "roles": {
            value["role"]: {
                "subject_id": int(subject_id),
                "frames": int(len(value["frame_id"])),
            }
            for subject_id, value in output.items()
        },
        "source_video": metadata.get("video_path"),
    }
    (output_root / "wham_adapter_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return path


def save_motion_json(
    output_root: Path,
    roles_2d: Dict[str, Role2DSequence],
    roles_3d: Dict[str, Role3DSequence],
    metadata: dict,
    contacts: Dict[str, object],
    metrics: Dict[str, object],
    keypoint_count: int,
    args: argparse.Namespace,
) -> None:
    fps = float(metadata.get("fps") or metrics.get("fps") or 24.0)
    payload = {
        "metadata": {
            "source_sequence_json": str(Path(args.sequence_json).resolve()),
            "source_metadata_json": str(Path(args.metadata_json).resolve()),
            "source_video": args.video_path or metadata.get("video_path"),
            "backend": "weak_perspective_2p5d_with_wham_adapter",
            "coordinate_system": "X right, Y up, Z camera-forward; world origin is first valid pelvis/root.",
            "frame_width": metadata.get("frame_width"),
            "frame_height": metadata.get("frame_height"),
            "fps": fps,
            "keypoint_count": keypoint_count,
            "keypoint_names": keypoint_names(keypoint_count),
            "settings": {
                "use_smoothed": args.use_smoothed,
                "reconstruction_mode": args.reconstruction_mode,
                "visibility_mode": args.visibility_mode,
                "assumed_height_m": args.assumed_height_m,
                "focal_length": args.focal_length or default_focal_length(metadata),
                "depth_alpha": args.depth_alpha,
                "smooth_alpha": args.smooth_alpha,
                "foot_ground_threshold": args.foot_ground_threshold,
                "foot_contact_speed": args.foot_contact_speed,
                "foot_lock_min_frames": args.foot_lock_min_frames,
                "foot_lock_blend": args.foot_lock_blend,
                "hand_contact_threshold": args.hand_contact_threshold,
                "hand_body_threshold": args.hand_body_threshold,
                "collision_radius": args.collision_radius,
            },
        },
        "roles": {},
        "contacts": contacts,
        "metrics": metrics,
    }

    for role_name, role in roles_3d.items():
        records = []
        seq2d = roles_2d[role_name]
        for frame_index in range(len(role.valid)):
            records.append(
                {
                    "frame_index": frame_index,
                    "timestamp": frame_index / fps,
                    "visible": bool(seq2d.visible[frame_index]),
                    "valid_3d": bool(role.valid[frame_index]),
                    "track_id": int(seq2d.track_ids[frame_index]) if seq2d.track_ids[frame_index] >= 0 else None,
                    "bbox": array_to_json(seq2d.bboxes[frame_index]),
                    "scores": array_to_json(seq2d.scores[frame_index]),
                    "joints3d_cam": array_to_json(role.joints_cam[frame_index]) if role.valid[frame_index] else None,
                    "joints3d_world": array_to_json(role.joints_world_smooth[frame_index]) if role.valid[frame_index] else None,
                    "joints3d_world_refined": array_to_json(role.joints_world_refined[frame_index]) if role.valid[frame_index] else None,
                    "trans_world": array_to_json(role.trans_world_smooth[frame_index]) if role.valid[frame_index] else None,
                    "trans_world_refined": array_to_json(role.trans_world_refined[frame_index]) if role.valid[frame_index] else None,
                    "poses_root_world": array_to_json(role.poses_root_world[frame_index]) if role.valid[frame_index] else None,
                    "foot_contacts": {
                        "left": bool(role.foot_contacts[frame_index, 0]),
                        "right": bool(role.foot_contacts[frame_index, 1]),
                    },
                    "poses_body": None,
                    "betas": None,
                    "verts_cam": None,
                }
            )
        payload["roles"][role_name] = records

    (output_root / "motion3d_sequences.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def save_motion_csv(
    output_root: Path,
    roles_2d: Dict[str, Role2DSequence],
    roles_3d: Dict[str, Role3DSequence],
    keypoint_count: int,
) -> None:
    names = keypoint_names(keypoint_count)
    with (output_root / "motion3d_sequences.csv").open("w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        writer.writerow(
            [
                "frame_index",
                "role",
                "visible",
                "valid_3d",
                "track_id",
                "joint_index",
                "joint_name",
                "score",
                "world_x",
                "world_y",
                "world_z",
                "refined_world_x",
                "refined_world_y",
                "refined_world_z",
            ]
        )
        for role_name, role in roles_3d.items():
            seq2d = roles_2d[role_name]
            for frame_index in range(len(role.valid)):
                for joint_index in range(keypoint_count):
                    point = role.joints_world_smooth[frame_index, joint_index]
                    refined = role.joints_world_refined[frame_index, joint_index]
                    writer.writerow(
                        [
                            frame_index,
                            role_name,
                            bool(seq2d.visible[frame_index]),
                            bool(role.valid[frame_index]),
                            int(seq2d.track_ids[frame_index]) if seq2d.track_ids[frame_index] >= 0 else None,
                            joint_index,
                            names[joint_index],
                            float(seq2d.scores[frame_index, joint_index]),
                            *array_to_json(point),
                            *array_to_json(refined),
                        ]
                    )


def save_root_motion_csv(output_root: Path, roles_3d: Dict[str, Role3DSequence], fps: float) -> None:
    with (output_root / "root_motion.csv").open("w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        writer.writerow(
            [
                "frame_index",
                "timestamp",
                "role",
                "valid_3d",
                "root_x",
                "root_y",
                "root_z",
                "refined_root_x",
                "refined_root_y",
                "refined_root_z",
                "root_yaw_axis_angle_y",
            ]
        )
        for role_name, role in roles_3d.items():
            for frame_index in range(len(role.valid)):
                writer.writerow(
                    [
                        frame_index,
                        frame_index / fps,
                        role_name,
                        bool(role.valid[frame_index]),
                        *array_to_json(role.trans_world_smooth[frame_index]),
                        *array_to_json(role.trans_world_refined[frame_index]),
                        float(role.poses_root_world[frame_index, 1]),
                    ]
                )


def save_npz(output_root: Path, roles_3d: Dict[str, Role3DSequence]) -> None:
    arrays = {}
    for role_name, role in roles_3d.items():
        prefix = role_name
        arrays[f"{prefix}_joints_cam"] = role.joints_cam
        arrays[f"{prefix}_joints_world"] = role.joints_world_smooth
        arrays[f"{prefix}_joints_world_refined"] = role.joints_world_refined
        arrays[f"{prefix}_trans_world"] = role.trans_world_smooth
        arrays[f"{prefix}_trans_world_refined"] = role.trans_world_refined
        arrays[f"{prefix}_poses_root_world"] = role.poses_root_world
        arrays[f"{prefix}_foot_contacts"] = role.foot_contacts
        arrays[f"{prefix}_valid"] = role.valid
    np.savez_compressed(output_root / "motion3d_arrays.npz", **arrays)


def main() -> None:
    args = parse_args()
    output_root = ensure_dir(Path(args.output_root).resolve())
    payload = load_json(Path(args.sequence_json).resolve())
    metadata_path = Path(args.metadata_json).resolve()
    metadata = load_json(metadata_path) if metadata_path.exists() else payload.get("metadata", {})
    if args.video_path:
        metadata["video_path"] = args.video_path

    roles_2d, frame_count, keypoint_count = build_role_2d_sequences(
        payload=payload,
        metadata=metadata,
        use_smoothed=args.use_smoothed,
        min_score=args.min_score,
        max_frames=args.max_frames,
        visibility_mode=args.visibility_mode,
    )

    roles_3d = {
        role_name: reconstruct_role_3d(seq, metadata, keypoint_count, args)
        for role_name, seq in roles_2d.items()
    }

    fps = float(metadata.get("fps") or 24.0)
    contacts, metrics = detect_and_refine_interactions(roles_3d, keypoint_count, fps, args)

    if not args.skip_wham_export:
        wham_path = export_wham_observations(roles_2d, output_root, metadata, args.min_score)
        if wham_path is None:
            metrics["wham_adapter_warning"] = "joblib is not installed; WHAM observation export skipped."

    save_motion_json(output_root, roles_2d, roles_3d, metadata, contacts, metrics, keypoint_count, args)
    save_motion_csv(output_root, roles_2d, roles_3d, keypoint_count)
    save_root_motion_csv(output_root, roles_3d, fps)
    save_npz(output_root, roles_3d)
    (output_root / "contacts.json").write_text(json.dumps(contacts, indent=2), encoding="utf-8")
    (output_root / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"Finished 3D reconstruction for {frame_count} frames. Output saved to: {output_root}")


if __name__ == "__main__":
    main()
