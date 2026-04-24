#!/usr/bin/env python3
"""Export a minimum-viable dual-interaction asset package.

This stage converts the reconstructed 3D motion into engine-friendly MVP assets:

1. per-role skeleton definitions
2. segmented animation clips
3. local-rotation animation data + root motion
4. interaction constraints / IK / collision metadata
5. BVH clips for quick DCC import
6. lightweight animated glTF scenes for open-format review/import
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy.signal import find_peaks
from scipy.spatial.transform import Rotation as R


ROLE_NAMES = ("character_A", "character_B")

CUBE_VERTICES = np.asarray(
    [
        [-0.01, -0.01, -0.01],
        [0.01, -0.01, -0.01],
        [0.01, 0.01, -0.01],
        [-0.01, 0.01, -0.01],
        [-0.01, -0.01, 0.01],
        [0.01, -0.01, 0.01],
        [0.01, 0.01, 0.01],
        [-0.01, 0.01, 0.01],
    ],
    dtype=np.float32,
)

CUBE_INDICES = np.asarray(
    [
        0, 1, 2, 0, 2, 3,
        4, 6, 5, 4, 7, 6,
        0, 4, 5, 0, 5, 1,
        1, 5, 6, 1, 6, 2,
        2, 6, 7, 2, 7, 3,
        3, 7, 4, 3, 4, 0,
    ],
    dtype=np.uint16,
)


@dataclass(frozen=True)
class BoneSpec:
    name: str
    joint_index: Optional[int]
    parent: Optional[str]


@dataclass
class RoleMotion:
    name: str
    frame_count: int
    fps: float
    valid: np.ndarray
    root_translation: np.ndarray
    joints_world: np.ndarray
    foot_contacts: np.ndarray


EXPORT_BONES: List[BoneSpec] = [
    BoneSpec("root", None, None),
    BoneSpec("pelvis", 19, "root"),
    BoneSpec("spine", 18, "pelvis"),
    BoneSpec("head", 17, "spine"),
    BoneSpec("nose", 0, "head"),
    BoneSpec("left_shoulder", 5, "spine"),
    BoneSpec("left_elbow", 7, "left_shoulder"),
    BoneSpec("left_wrist", 9, "left_elbow"),
    BoneSpec("right_shoulder", 6, "spine"),
    BoneSpec("right_elbow", 8, "right_shoulder"),
    BoneSpec("right_wrist", 10, "right_elbow"),
    BoneSpec("left_hip", 11, "pelvis"),
    BoneSpec("left_knee", 13, "left_hip"),
    BoneSpec("left_ankle", 15, "left_knee"),
    BoneSpec("left_big_toe", 20, "left_ankle"),
    BoneSpec("left_small_toe", 22, "left_ankle"),
    BoneSpec("left_heel", 24, "left_ankle"),
    BoneSpec("right_hip", 12, "pelvis"),
    BoneSpec("right_knee", 14, "right_hip"),
    BoneSpec("right_ankle", 16, "right_knee"),
    BoneSpec("right_big_toe", 21, "right_ankle"),
    BoneSpec("right_small_toe", 23, "right_ankle"),
    BoneSpec("right_heel", 25, "right_ankle"),
]

NAME_TO_INDEX = {bone.name: idx for idx, bone in enumerate(EXPORT_BONES)}
CHILDREN_BY_NAME: Dict[str, List[str]] = {}
for bone in EXPORT_BONES:
    if bone.parent is None:
        continue
    CHILDREN_BY_NAME.setdefault(bone.parent, []).append(bone.name)


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Export dual-character MVP assets from motion3d_sequences.json."
    )
    parser.add_argument(
        "--motion-json",
        default=str(project_root / "outputs2_improved" / "video_3d" / "motion3d_sequences.json"),
    )
    parser.add_argument(
        "--output-root",
        default=str(project_root / "mvp_delivery" / "video2"),
    )
    parser.add_argument("--source-label", default="video2_mvp")
    parser.add_argument("--min-clip-frames", type=int, default=60)
    parser.add_argument("--max-clip-frames", type=int, default=240)
    parser.add_argument("--contact-gap-tolerance", type=int, default=3)
    parser.add_argument("--speed-smoothing-window", type=int, default=9)
    parser.add_argument("--foot-lock-threshold", type=float, default=0.15)
    parser.add_argument("--root-minima-distance", type=int, default=45)
    return parser.parse_args()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_json(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"Required file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def finite_rows(points: np.ndarray) -> np.ndarray:
    return points[np.isfinite(points).all(axis=1)]


def median_vector(vectors: List[np.ndarray], fallback: np.ndarray) -> np.ndarray:
    if not vectors:
        return fallback.astype(np.float32)
    merged = np.asarray(vectors, dtype=np.float32)
    return np.median(merged, axis=0).astype(np.float32)


def normalize_vector(vector: np.ndarray, fallback: Optional[np.ndarray] = None) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-8:
        if fallback is None:
            return np.array([1.0, 0.0, 0.0], dtype=np.float32)
        return normalize_vector(fallback, None)
    return (vector / norm).astype(np.float32)


def rotation_from_to(source: np.ndarray, target: np.ndarray) -> R:
    src = normalize_vector(source)
    dst = normalize_vector(target)
    dot = float(np.clip(np.dot(src, dst), -1.0, 1.0))
    if dot >= 1.0 - 1e-7:
        return R.identity()
    if dot <= -1.0 + 1e-7:
        axis = np.cross(src, np.array([1.0, 0.0, 0.0], dtype=np.float32))
        if float(np.linalg.norm(axis)) <= 1e-6:
            axis = np.cross(src, np.array([0.0, 1.0, 0.0], dtype=np.float32))
        axis = normalize_vector(axis)
        return R.from_rotvec(axis * math.pi)
    axis = np.cross(src, dst)
    axis = normalize_vector(axis)
    angle = math.acos(dot)
    return R.from_rotvec(axis * angle)


def contiguous_segments(indices: np.ndarray, gap_tolerance: int = 0) -> List[Tuple[int, int]]:
    if len(indices) == 0:
        return []
    segments: List[Tuple[int, int]] = []
    start = int(indices[0])
    last = int(indices[0])
    for value in indices[1:]:
        value = int(value)
        if value <= last + gap_tolerance + 1:
            last = value
            continue
        segments.append((start, last + 1))
        start = value
        last = value
    segments.append((start, last + 1))
    return segments


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(values) == 0:
        return values.astype(np.float32, copy=True)
    radius = max(1, int(window))
    kernel = np.ones((radius,), dtype=np.float32) / float(radius)
    padded = np.pad(values.astype(np.float32), (radius // 2, radius - 1 - radius // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid").astype(np.float32)


def load_role_motion(payload: dict) -> Tuple[Dict[str, RoleMotion], dict]:
    metadata = payload.get("metadata", {})
    fps = float(metadata.get("fps") or 30.0)
    roles_payload = payload.get("roles", {})
    roles: Dict[str, RoleMotion] = {}

    for role_name in ROLE_NAMES:
        records = roles_payload.get(role_name)
        if records is None:
            continue
        frame_count = len(records)
        valid = np.zeros((frame_count,), dtype=bool)
        root_translation = np.full((frame_count, 3), np.nan, dtype=np.float32)
        joints_world = np.full((frame_count, 26, 3), np.nan, dtype=np.float32)
        foot_contacts = np.zeros((frame_count, 2), dtype=bool)

        for frame_index, record in enumerate(records):
            valid[frame_index] = bool(record.get("valid_3d"))
            trans = record.get("trans_world_refined")
            if trans is not None and len(trans) == 3 and all(value is not None for value in trans):
                root_translation[frame_index] = np.asarray(trans, dtype=np.float32)
            points = record.get("joints3d_world_refined")
            if points:
                rows = []
                for point in points[:26]:
                    if point is None or any(value is None for value in point):
                        rows.append([np.nan, np.nan, np.nan])
                    else:
                        rows.append([float(point[0]), float(point[1]), float(point[2])])
                if rows:
                    joints_world[frame_index, : len(rows)] = np.asarray(rows, dtype=np.float32)
            contacts = record.get("foot_contacts") or {}
            foot_contacts[frame_index, 0] = bool(contacts.get("left"))
            foot_contacts[frame_index, 1] = bool(contacts.get("right"))

        roles[role_name] = RoleMotion(
            name=role_name,
            frame_count=frame_count,
            fps=fps,
            valid=valid,
            root_translation=root_translation,
            joints_world=joints_world,
            foot_contacts=foot_contacts,
        )

    if not roles:
        raise SystemExit("No role motion data found in motion JSON.")
    return roles, metadata


def estimate_rest_positions(role: RoleMotion) -> Dict[str, np.ndarray]:
    rest_positions: Dict[str, np.ndarray] = {"root": np.zeros((3,), dtype=np.float32)}
    default_offsets = {
        "pelvis": np.zeros((3,), dtype=np.float32),
        "spine": np.array([0.0, 0.22, 0.0], dtype=np.float32),
        "head": np.array([0.0, 0.18, 0.0], dtype=np.float32),
        "nose": np.array([0.0, 0.06, 0.05], dtype=np.float32),
        "left_shoulder": np.array([0.16, 0.04, 0.0], dtype=np.float32),
        "left_elbow": np.array([0.28, -0.02, 0.0], dtype=np.float32),
        "left_wrist": np.array([0.25, 0.0, 0.0], dtype=np.float32),
        "right_shoulder": np.array([-0.16, 0.04, 0.0], dtype=np.float32),
        "right_elbow": np.array([-0.28, -0.02, 0.0], dtype=np.float32),
        "right_wrist": np.array([-0.25, 0.0, 0.0], dtype=np.float32),
        "left_hip": np.array([0.10, -0.10, 0.0], dtype=np.float32),
        "left_knee": np.array([0.0, -0.42, 0.0], dtype=np.float32),
        "left_ankle": np.array([0.0, -0.42, 0.0], dtype=np.float32),
        "left_big_toe": np.array([0.05, -0.04, 0.12], dtype=np.float32),
        "left_small_toe": np.array([0.02, -0.04, 0.10], dtype=np.float32),
        "left_heel": np.array([0.0, -0.04, -0.08], dtype=np.float32),
        "right_hip": np.array([-0.10, -0.10, 0.0], dtype=np.float32),
        "right_knee": np.array([0.0, -0.42, 0.0], dtype=np.float32),
        "right_ankle": np.array([0.0, -0.42, 0.0], dtype=np.float32),
        "right_big_toe": np.array([-0.05, -0.04, 0.12], dtype=np.float32),
        "right_small_toe": np.array([-0.02, -0.04, 0.10], dtype=np.float32),
        "right_heel": np.array([0.0, -0.04, -0.08], dtype=np.float32),
    }

    for bone in EXPORT_BONES[1:]:
        parent = bone.parent
        assert parent is not None
        parent_bone = EXPORT_BONES[NAME_TO_INDEX[parent]]
        vectors: List[np.ndarray] = []
        for frame_index in range(role.frame_count):
            if not role.valid[frame_index]:
                continue
            if bone.joint_index is None:
                continue
            child_pos = role.joints_world[frame_index, bone.joint_index]
            if parent == "root":
                parent_pos = role.root_translation[frame_index]
            else:
                assert parent_bone.joint_index is not None
                parent_pos = role.joints_world[frame_index, parent_bone.joint_index]
            if np.isfinite(child_pos).all() and np.isfinite(parent_pos).all():
                vectors.append(child_pos - parent_pos)
        fallback = default_offsets.get(bone.name, np.array([0.0, 0.1, 0.0], dtype=np.float32))
        local_offset = median_vector(vectors, fallback)
        if float(np.linalg.norm(local_offset)) <= 1e-4:
            local_offset = fallback
        rest_positions[bone.name] = rest_positions[parent] + local_offset

    return rest_positions


def estimate_capsule_radius(bone_name: str, length: float) -> float:
    if "spine" in bone_name or "pelvis" in bone_name:
        return float(np.clip(length * 0.22, 0.05, 0.14))
    if "hip" in bone_name or "knee" in bone_name or "ankle" in bone_name:
        return float(np.clip(length * 0.16, 0.03, 0.09))
    if "shoulder" in bone_name or "elbow" in bone_name or "wrist" in bone_name:
        return float(np.clip(length * 0.12, 0.025, 0.07))
    if "toe" in bone_name or "heel" in bone_name or "nose" in bone_name or "head" in bone_name:
        return float(np.clip(length * 0.08, 0.015, 0.04))
    return float(np.clip(length * 0.12, 0.02, 0.06))


def compute_global_and_local_rotations(
    role: RoleMotion,
    rest_positions: Dict[str, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    bone_count = len(EXPORT_BONES)
    local_quats = np.zeros((role.frame_count, bone_count, 4), dtype=np.float32)
    global_quats = np.zeros((role.frame_count, bone_count, 4), dtype=np.float32)
    identity = R.identity()

    for frame_index in range(role.frame_count):
        if not role.valid[frame_index]:
            local_quats[frame_index, :, 3] = 1.0
            global_quats[frame_index, :, 3] = 1.0
            continue

        global_rotations: Dict[str, R] = {"root": identity}
        local_rotations: Dict[str, R] = {"root": identity}

        for bone in EXPORT_BONES[1:]:
            parent_name = bone.parent
            assert parent_name is not None
            parent_global = global_rotations[parent_name]
            bone_name = bone.name
            bone_pos = (
                role.root_translation[frame_index]
                if bone_name == "root"
                else role.joints_world[frame_index, bone.joint_index] if bone.joint_index is not None else np.full((3,), np.nan, dtype=np.float32)
            )
            if bone_name != "root" and bone.joint_index is not None:
                bone_pos = role.joints_world[frame_index, bone.joint_index]
            if bone_name == "root":
                bone_pos = role.root_translation[frame_index]
            child_names = CHILDREN_BY_NAME.get(bone_name, [])
            source_vectors = []
            target_vectors = []
            if bone_name == "root":
                bone_pos = role.root_translation[frame_index]

            for child_name in child_names:
                child_bone = EXPORT_BONES[NAME_TO_INDEX[child_name]]
                if child_bone.joint_index is None:
                    continue
                if bone_name == "root":
                    parent_pos = role.root_translation[frame_index]
                elif bone.joint_index is not None:
                    parent_pos = role.joints_world[frame_index, bone.joint_index]
                else:
                    continue
                child_pos = role.joints_world[frame_index, child_bone.joint_index]
                if not (np.isfinite(parent_pos).all() and np.isfinite(child_pos).all()):
                    continue
                observed = child_pos - parent_pos
                rest = rest_positions[child_name] - rest_positions[bone_name]
                if float(np.linalg.norm(observed)) <= 1e-6 or float(np.linalg.norm(rest)) <= 1e-6:
                    continue
                source_vectors.append(normalize_vector(rest))
                target_vectors.append(normalize_vector(observed))

            if target_vectors:
                if len(target_vectors) >= 2:
                    try:
                        rotation_global, _ = R.align_vectors(
                            np.asarray(target_vectors, dtype=np.float32),
                            np.asarray(source_vectors, dtype=np.float32),
                        )
                    except Exception:
                        rotation_global = rotation_from_to(source_vectors[0], target_vectors[0])
                else:
                    rotation_global = rotation_from_to(source_vectors[0], target_vectors[0])
            else:
                rotation_global = parent_global

            rotation_local = parent_global.inv() * rotation_global
            global_rotations[bone_name] = rotation_global
            local_rotations[bone_name] = rotation_local

        for bone_index, bone in enumerate(EXPORT_BONES):
            local_quats[frame_index, bone_index] = local_rotations.get(bone.name, identity).as_quat().astype(np.float32)
            global_quats[frame_index, bone_index] = global_rotations.get(bone.name, identity).as_quat().astype(np.float32)

    for bone_index in range(bone_count):
        for frame_index in range(1, role.frame_count):
            if np.dot(local_quats[frame_index - 1, bone_index], local_quats[frame_index, bone_index]) < 0.0:
                local_quats[frame_index, bone_index] *= -1.0

    return local_quats, global_quats


def compute_root_speed(role: RoleMotion) -> np.ndarray:
    speeds = np.zeros((role.frame_count,), dtype=np.float32)
    for frame_index in range(1, role.frame_count):
        current = role.root_translation[frame_index]
        previous = role.root_translation[frame_index - 1]
        if np.isfinite(current).all() and np.isfinite(previous).all():
            speeds[frame_index] = float(np.linalg.norm(current[[0, 2]] - previous[[0, 2]]) * role.fps)
    return speeds


def build_contact_timeline(contacts: dict, frame_count: int) -> Dict[str, np.ndarray]:
    timeline = {
        "hand_hand": np.zeros((frame_count,), dtype=np.int32),
        "hand_body": np.zeros((frame_count,), dtype=np.int32),
    }
    for item in contacts.get("hand_hand", []):
        frame_index = int(item.get("frame_index", -1))
        if 0 <= frame_index < frame_count:
            timeline["hand_hand"][frame_index] += 1
    for item in contacts.get("hand_body", []):
        frame_index = int(item.get("frame_index", -1))
        if 0 <= frame_index < frame_count:
            timeline["hand_body"][frame_index] += 1
    timeline["interaction"] = ((timeline["hand_hand"] + timeline["hand_body"]) > 0).astype(np.int32)
    return timeline


def merge_short_segments(boundaries: List[int], min_clip_frames: int, frame_count: int) -> List[int]:
    boundaries = sorted(set(boundaries))
    changed = True
    while changed and len(boundaries) > 2:
        changed = False
        for index in range(1, len(boundaries) - 1):
            left = boundaries[index] - boundaries[index - 1]
            right = boundaries[index + 1] - boundaries[index]
            if left < min_clip_frames or right < min_clip_frames:
                boundaries.pop(index)
                changed = True
                break
    if boundaries[0] != 0:
        boundaries.insert(0, 0)
    if boundaries[-1] != frame_count:
        boundaries.append(frame_count)
    return boundaries


def segment_clips(
    roles: Dict[str, RoleMotion],
    contacts: dict,
    args: argparse.Namespace,
) -> List[dict]:
    frame_count = min(role.frame_count for role in roles.values())
    contact_timeline = build_contact_timeline(contacts, frame_count)
    mean_speed = np.mean(
        np.stack([compute_root_speed(role)[:frame_count] for role in roles.values()], axis=0),
        axis=0,
    )
    smoothed_speed = moving_average(mean_speed, args.speed_smoothing_window)
    movement_peaks, _ = find_peaks(smoothed_speed, distance=args.root_minima_distance)
    minima_peaks, _ = find_peaks(-smoothed_speed, distance=args.root_minima_distance)

    boundaries = {0, frame_count}
    interaction = contact_timeline["interaction"]
    change_frames = np.where(np.diff(interaction) != 0)[0] + 1
    for frame_index in change_frames.tolist():
        boundaries.add(int(frame_index))

    last_boundary = 0
    for frame_index in movement_peaks.tolist():
        if frame_index - last_boundary >= args.max_clip_frames:
            nearest_min = None
            for candidate in minima_peaks.tolist():
                if last_boundary + args.min_clip_frames <= candidate <= frame_index:
                    nearest_min = candidate
            if nearest_min is not None:
                boundaries.add(int(nearest_min))
                last_boundary = int(nearest_min)

    sorted_boundaries = merge_short_segments(list(boundaries), args.min_clip_frames, frame_count)
    clips: List[dict] = []
    for clip_index in range(len(sorted_boundaries) - 1):
        start = sorted_boundaries[clip_index]
        end = sorted_boundaries[clip_index + 1]
        if end - start <= 1:
            continue
        local_interaction = interaction[start:end]
        hand_hand_frames = np.where(contact_timeline["hand_hand"][start:end] > 0)[0].tolist()
        hand_body_frames = np.where(contact_timeline["hand_body"][start:end] > 0)[0].tolist()
        dominant = "interaction" if int(local_interaction.sum()) >= max(1, (end - start) // 8) else "transition"
        sync_points = sorted(
            set(
                [0, end - start - 1]
                + hand_hand_frames[:3]
                + hand_body_frames[:3]
            )
        )
        clips.append(
            {
                "clip_index": clip_index,
                "name": f"{clip_index + 1:02d}_{dominant}",
                "start_frame": int(start),
                "end_frame": int(end),
                "duration_frames": int(end - start),
                "duration_seconds": float((end - start) / roles[ROLE_NAMES[0]].fps),
                "dominant_state": dominant,
                "sync_points": [int(point) for point in sync_points],
                "interaction_summary": {
                    "hand_hand_frames": int(np.count_nonzero(contact_timeline["hand_hand"][start:end] > 0)),
                    "hand_body_frames": int(np.count_nonzero(contact_timeline["hand_body"][start:end] > 0)),
                    "interaction_active_frames": int(np.count_nonzero(local_interaction)),
                },
                "loop_candidate": False,
            }
        )
    return clips


def summarize_constraints(contacts: dict, gap_tolerance: int) -> List[dict]:
    grouped: Dict[Tuple[str, ...], List[int]] = {}
    for item in contacts.get("hand_hand", []):
        key = (
            "attach",
            "character_A",
            f"character_A.{item.get('role_a_joint', 'unknown')}",
            f"character_B.{item.get('role_b_joint', 'unknown')}",
        )
        grouped.setdefault(key, []).append(int(item.get("frame_index", -1)))

    for item in contacts.get("hand_body", []):
        source_role = str(item.get("source_role", "unknown"))
        source_joint = str(item.get("source_joint", "unknown"))
        target_role = "character_B" if source_role == "character_A" else "character_A"
        key = (
            "surface_contact",
            source_role,
            f"{source_role}.{source_joint}",
            f"{target_role}.body_core",
        )
        grouped.setdefault(key, []).append(int(item.get("frame_index", -1)))

    constraints = []
    for key, frames in grouped.items():
        kind, owner_role, source, target = key
        valid = np.asarray(sorted(frame for frame in frames if frame >= 0), dtype=np.int32)
        for start, end in contiguous_segments(valid, gap_tolerance):
            constraints.append(
                {
                    "type": kind,
                    "role": owner_role,
                    "source": source,
                    "target": target,
                    "start_frame": int(start),
                    "end_frame": int(end),
                    "stiffness": 0.8 if kind == "attach" else 0.55,
                }
            )

    constraints.append(
        {
            "type": "no_penetration",
            "source": "character_A.body",
            "target": "character_B.body",
            "threshold_m": 0.01,
        }
    )
    return constraints


def build_ik_metadata(
    roles: Dict[str, RoleMotion],
    rest_positions_by_role: Dict[str, Dict[str, np.ndarray]],
) -> dict:
    metadata = {"roles": {}}
    for role_name, rest_positions in rest_positions_by_role.items():
        def pole_direction(a: str, b: str, c: str) -> List[float]:
            ab = rest_positions[b] - rest_positions[a]
            bc = rest_positions[c] - rest_positions[b]
            axis = np.cross(ab, bc)
            if float(np.linalg.norm(axis)) <= 1e-6:
                axis = np.array([0.0, 0.0, 1.0], dtype=np.float32)
            return normalize_vector(axis).round(6).tolist()

        metadata["roles"][role_name] = {
            "effectors": [
                {"name": "left_hand", "bone": "left_wrist"},
                {"name": "right_hand", "bone": "right_wrist"},
                {"name": "left_foot", "bone": "left_ankle"},
                {"name": "right_foot", "bone": "right_ankle"},
            ],
            "pole_vectors": [
                {"joint": "left_elbow", "direction": pole_direction("left_shoulder", "left_elbow", "left_wrist")},
                {"joint": "right_elbow", "direction": pole_direction("right_shoulder", "right_elbow", "right_wrist")},
                {"joint": "left_knee", "direction": pole_direction("left_hip", "left_knee", "left_ankle")},
                {"joint": "right_knee", "direction": pole_direction("right_hip", "right_knee", "right_ankle")},
            ],
        }
    return metadata


def build_collision_capsules(rest_positions_by_role: Dict[str, Dict[str, np.ndarray]]) -> dict:
    payload = {"roles": {}}
    for role_name, rest_positions in rest_positions_by_role.items():
        capsules = []
        for bone in EXPORT_BONES:
            if bone.name == "root":
                continue
            for child_name in CHILDREN_BY_NAME.get(bone.name, []):
                start = rest_positions[bone.name]
                end = rest_positions[child_name]
                length = float(np.linalg.norm(end - start))
                capsules.append(
                    {
                        "bone": bone.name,
                        "child": child_name,
                        "start_rest": start.round(6).tolist(),
                        "end_rest": end.round(6).tolist(),
                        "radius_m": estimate_capsule_radius(child_name, length),
                    }
                )
        payload["roles"][role_name] = capsules
    return payload


def bvh_joint_name(role_name: str, bone_name: str) -> str:
    return f"{role_name}_{bone_name}"


def write_bvh(
    path: Path,
    role_name: str,
    rest_positions: Dict[str, np.ndarray],
    clip_frames: Sequence[int],
    local_quats: np.ndarray,
    root_translation: np.ndarray,
    fps: float,
) -> None:
    lines: List[str] = ["HIERARCHY"]

    def emit_joint(bone_name: str, indent: int) -> None:
        prefix = "\t" * indent
        children = CHILDREN_BY_NAME.get(bone_name, [])
        if bone_name == "root":
            lines.append(f"{prefix}ROOT {bvh_joint_name(role_name, bone_name)}")
        else:
            lines.append(f"{prefix}JOINT {bvh_joint_name(role_name, bone_name)}")
        lines.append(f"{prefix}{{")
        if bone_name == "root":
            offset = np.zeros((3,), dtype=np.float32)
            lines.append(f"{prefix}\tOFFSET 0.000000 0.000000 0.000000")
            lines.append(
                f"{prefix}\tCHANNELS 6 Xposition Yposition Zposition Xrotation Yrotation Zrotation"
            )
        else:
            parent = EXPORT_BONES[NAME_TO_INDEX[bone_name]].parent
            assert parent is not None
            offset = rest_positions[bone_name] - rest_positions[parent]
            lines.append(
                f"{prefix}\tOFFSET {offset[0]:.6f} {offset[1]:.6f} {offset[2]:.6f}"
            )
            lines.append(f"{prefix}\tCHANNELS 3 Xrotation Yrotation Zrotation")

        if children:
            for child_name in children:
                emit_joint(child_name, indent + 1)
        else:
            lines.append(f"{prefix}\tEnd Site")
            lines.append(f"{prefix}\t{{")
            lines.append(f"{prefix}\t\tOFFSET 0.000000 0.050000 0.000000")
            lines.append(f"{prefix}\t}}")
        lines.append(f"{prefix}}}")

    emit_joint("root", 0)
    lines.append("MOTION")
    lines.append(f"Frames: {len(clip_frames)}")
    lines.append(f"Frame Time: {1.0 / fps:.8f}")

    bone_order = [bone.name for bone in EXPORT_BONES]
    for frame_index in clip_frames:
        row: List[str] = []
        root = root_translation[frame_index]
        if not np.isfinite(root).all():
            root = np.zeros((3,), dtype=np.float32)
        root_euler = np.zeros((3,), dtype=np.float32)
        row.extend([f"{root[0]:.6f}", f"{root[1]:.6f}", f"{root[2]:.6f}"])
        row.extend([f"{root_euler[0]:.6f}", f"{root_euler[1]:.6f}", f"{root_euler[2]:.6f}"])

        for bone_name in bone_order[1:]:
            bone_index = NAME_TO_INDEX[bone_name]
            quat = local_quats[frame_index, bone_index]
            euler = R.from_quat(quat).as_euler("XYZ", degrees=True)
            row.extend([f"{euler[0]:.6f}", f"{euler[1]:.6f}", f"{euler[2]:.6f}"])

        lines.append(" ".join(row))

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _component_type_and_dtype(array: np.ndarray) -> Tuple[int, str]:
    if array.dtype == np.float32:
        return 5126, "FLOAT"
    if array.dtype == np.uint16:
        return 5123, "UNSIGNED_SHORT"
    raise ValueError(f"Unsupported dtype for glTF accessor: {array.dtype}")


class GltfBufferBuilder:
    def __init__(self) -> None:
        self.data = bytearray()
        self.buffer_views: List[dict] = []
        self.accessors: List[dict] = []

    def add_array(self, array: np.ndarray, target: Optional[int] = None) -> int:
        array = np.ascontiguousarray(array)
        component_type, accessor_type = _component_type_and_dtype(array)
        if array.ndim == 1:
            count = int(array.shape[0])
            min_value = [float(np.min(array))] if count else [0.0]
            max_value = [float(np.max(array))] if count else [0.0]
        else:
            count = int(array.shape[0])
            width = int(array.shape[1])
            accessor_type = {2: "VEC2", 3: "VEC3", 4: "VEC4"}.get(width, accessor_type)
            min_value = array.min(axis=0).astype(float).tolist() if count else [0.0] * width
            max_value = array.max(axis=0).astype(float).tolist() if count else [0.0] * width

        while len(self.data) % 4 != 0:
            self.data.extend(b"\x00")
        byte_offset = len(self.data)
        raw = array.tobytes()
        self.data.extend(raw)
        view = {
            "buffer": 0,
            "byteOffset": byte_offset,
            "byteLength": len(raw),
        }
        if target is not None:
            view["target"] = target
        view_index = len(self.buffer_views)
        self.buffer_views.append(view)

        accessor = {
            "bufferView": view_index,
            "componentType": component_type,
            "count": count,
            "type": accessor_type,
            "min": min_value,
            "max": max_value,
        }
        accessor_index = len(self.accessors)
        self.accessors.append(accessor)
        return accessor_index

    def encoded_uri(self) -> str:
        payload = base64.b64encode(bytes(self.data)).decode("ascii")
        return f"data:application/octet-stream;base64,{payload}"


def write_dual_clip_gltf(
    path: Path,
    clip_name: str,
    role_names: Sequence[str],
    rest_positions_by_role: Dict[str, Dict[str, np.ndarray]],
    local_quats_by_role: Dict[str, np.ndarray],
    root_translation_by_role: Dict[str, np.ndarray],
    clip_frames: Sequence[int],
    fps: float,
) -> None:
    builder = GltfBufferBuilder()

    position_accessor = builder.add_array(CUBE_VERTICES, target=34962)
    index_accessor = builder.add_array(CUBE_INDICES, target=34963)

    mesh = {
        "primitives": [
            {
                "attributes": {"POSITION": position_accessor},
                "indices": index_accessor,
                "mode": 4,
            }
        ]
    }

    nodes: List[dict] = [{"name": "scene_root", "children": []}]
    role_root_indices: Dict[str, int] = {}

    for role_name in role_names:
        role_root_index = len(nodes)
        role_root_indices[role_name] = role_root_index
        nodes.append({"name": f"{role_name}_group", "children": []})
        nodes[0]["children"].append(role_root_index)

        node_indices: Dict[str, int] = {}
        for bone in EXPORT_BONES:
            node_index = len(nodes)
            node_indices[bone.name] = node_index
            node = {"name": f"{role_name}_{bone.name}"}
            if bone.name != "root":
                node["mesh"] = 0
            if bone.parent is None:
                nodes[role_root_index]["children"].append(node_index)
            else:
                nodes[node_indices[bone.parent]].setdefault("children", []).append(node_index)
            if bone.parent is None:
                node["translation"] = [0.0, 0.0, 0.0]
            else:
                offset = rest_positions_by_role[role_name][bone.name] - rest_positions_by_role[role_name][bone.parent]
                node["translation"] = offset.astype(float).round(6).tolist()
            node["rotation"] = [0.0, 0.0, 0.0, 1.0]
            nodes.append(node)

    gltf = {
        "asset": {"version": "2.0", "generator": "Codex MVP Asset Exporter"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": nodes,
        "meshes": [mesh],
        "materials": [{"pbrMetallicRoughness": {"baseColorFactor": [0.9, 0.75, 0.2, 1.0], "metallicFactor": 0.0, "roughnessFactor": 1.0}}],
        "animations": [],
        "buffers": [],
        "bufferViews": [],
        "accessors": [],
    }
    gltf["meshes"][0]["primitives"][0]["material"] = 0

    animation = {"name": clip_name, "channels": [], "samplers": []}
    time_values = (np.arange(len(clip_frames), dtype=np.float32) / float(fps)).astype(np.float32)
    time_accessor = builder.add_array(time_values)

    current_base = 2
    for role_name in role_names:
        root_node_index = role_root_indices[role_name] + 1  # actual role root bone node
        root_local = root_translation_by_role[role_name][list(clip_frames)]
        if len(root_local):
            root_local = root_local - root_local[0]
        root_accessor = builder.add_array(root_local.astype(np.float32))
        sampler_index = len(animation["samplers"])
        animation["samplers"].append({"input": time_accessor, "output": root_accessor, "interpolation": "LINEAR"})
        animation["channels"].append({"sampler": sampler_index, "target": {"node": root_node_index, "path": "translation"}})

        for bone_index, bone in enumerate(EXPORT_BONES):
            if bone.name == "root":
                continue
            node_index = current_base + bone_index
            rotations = local_quats_by_role[role_name][list(clip_frames), bone_index].astype(np.float32)
            accessor = builder.add_array(rotations)
            sampler_index = len(animation["samplers"])
            animation["samplers"].append({"input": time_accessor, "output": accessor, "interpolation": "LINEAR"})
            animation["channels"].append({"sampler": sampler_index, "target": {"node": node_index, "path": "rotation"}})
        current_base += len(EXPORT_BONES) + 1

    gltf["animations"].append(animation)
    gltf["bufferViews"] = builder.buffer_views
    gltf["accessors"] = builder.accessors
    gltf["buffers"] = [{"byteLength": len(builder.data), "uri": builder.encoded_uri()}]
    path.write_text(json.dumps(gltf, indent=2), encoding="utf-8")


def write_role_clip_json(
    path: Path,
    role_name: str,
    clip: dict,
    local_quats: np.ndarray,
    root_translation: np.ndarray,
    fps: float,
) -> None:
    start = int(clip["start_frame"])
    end = int(clip["end_frame"])
    payload = {
        "role": role_name,
        "clip": clip,
        "fps": fps,
        "bone_order": [bone.name for bone in EXPORT_BONES],
        "root_translation_m": root_translation[start:end].round(6).tolist(),
        "local_rotation_quat_xyzw": local_quats[start:end].round(6).tolist(),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_csv_summary(path: Path, clips: Sequence[dict]) -> None:
    lines = [
        "clip_name,start_frame,end_frame,duration_frames,duration_seconds,dominant_state,interaction_active_frames,hand_hand_frames,hand_body_frames"
    ]
    for clip in clips:
        summary = clip["interaction_summary"]
        lines.append(
            ",".join(
                [
                    clip["name"],
                    str(clip["start_frame"]),
                    str(clip["end_frame"]),
                    str(clip["duration_frames"]),
                    f"{clip['duration_seconds']:.4f}",
                    clip["dominant_state"],
                    str(summary["interaction_active_frames"]),
                    str(summary["hand_hand_frames"]),
                    str(summary["hand_body_frames"]),
                ]
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    motion_path = Path(args.motion_json).resolve()
    output_root = ensure_dir(Path(args.output_root).resolve())
    payload = load_json(motion_path)
    roles, metadata = load_role_motion(payload)
    contacts = payload.get("contacts", {})

    rest_positions_by_role = {
        role_name: estimate_rest_positions(role)
        for role_name, role in roles.items()
    }
    local_quats_by_role: Dict[str, np.ndarray] = {}
    global_quats_by_role: Dict[str, np.ndarray] = {}
    for role_name, role in roles.items():
        local_quats, global_quats = compute_global_and_local_rotations(role, rest_positions_by_role[role_name])
        local_quats_by_role[role_name] = local_quats
        global_quats_by_role[role_name] = global_quats

    clips = segment_clips(roles, contacts, args)
    interaction_constraints = summarize_constraints(contacts, args.contact_gap_tolerance)
    ik_metadata = build_ik_metadata(roles, rest_positions_by_role)
    collision_capsules = build_collision_capsules(rest_positions_by_role)

    skeleton_dir = ensure_dir(output_root / "skeletons")
    clips_dir = ensure_dir(output_root / "clips")
    metadata_dir = ensure_dir(output_root / "metadata")
    full_dir = ensure_dir(output_root / "full_sequence")

    manifest = {
        "source_label": args.source_label,
        "source_motion_json": str(motion_path),
        "source_video": metadata.get("source_video"),
        "fps": float(metadata.get("fps") or roles[ROLE_NAMES[0]].fps),
        "roles": {},
        "clips": [],
        "files": {},
    }

    for role_name, role in roles.items():
        skeleton_payload = {
            "role": role_name,
            "fps": role.fps,
            "bone_order": [bone.name for bone in EXPORT_BONES],
            "bones": [
                {
                    "name": bone.name,
                    "joint_index": bone.joint_index,
                    "parent": bone.parent,
                    "rest_position_m": rest_positions_by_role[role_name][bone.name].round(6).tolist(),
                }
                for bone in EXPORT_BONES
            ],
        }
        skeleton_path = skeleton_dir / f"{role_name}_skeleton.json"
        skeleton_path.write_text(json.dumps(skeleton_payload, indent=2), encoding="utf-8")

        full_json_path = full_dir / f"{role_name}_full_animation.json"
        write_role_clip_json(
            full_json_path,
            role_name=role_name,
            clip={
                "name": "full_sequence",
                "start_frame": 0,
                "end_frame": role.frame_count,
                "duration_frames": role.frame_count,
                "duration_seconds": role.frame_count / role.fps,
            },
            local_quats=local_quats_by_role[role_name],
            root_translation=role.root_translation,
            fps=role.fps,
        )
        full_bvh_path = full_dir / f"{role_name}_full_sequence.bvh"
        write_bvh(
            full_bvh_path,
            role_name=role_name,
            rest_positions=rest_positions_by_role[role_name],
            clip_frames=list(range(role.frame_count)),
            local_quats=local_quats_by_role[role_name],
            root_translation=role.root_translation,
            fps=role.fps,
        )

        manifest["roles"][role_name] = {
            "skeleton_json": str(skeleton_path),
            "full_animation_json": str(full_json_path),
            "full_bvh": str(full_bvh_path),
        }

    for clip in clips:
        clip_dir = ensure_dir(clips_dir / clip["name"])
        clip_frames = list(range(int(clip["start_frame"]), int(clip["end_frame"])))
        clip_manifest = {
            "name": clip["name"],
            "directory": str(clip_dir),
            "start_frame": clip["start_frame"],
            "end_frame": clip["end_frame"],
            "files": {},
        }

        for role_name, role in roles.items():
            clip_json_path = clip_dir / f"{role_name}_animation.json"
            write_role_clip_json(
                clip_json_path,
                role_name=role_name,
                clip=clip,
                local_quats=local_quats_by_role[role_name],
                root_translation=role.root_translation,
                fps=role.fps,
            )
            clip_bvh_path = clip_dir / f"{role_name}.bvh"
            write_bvh(
                clip_bvh_path,
                role_name=role_name,
                rest_positions=rest_positions_by_role[role_name],
                clip_frames=clip_frames,
                local_quats=local_quats_by_role[role_name],
                root_translation=role.root_translation,
                fps=role.fps,
            )
            clip_manifest["files"][role_name] = {
                "animation_json": str(clip_json_path),
                "bvh": str(clip_bvh_path),
            }

        gltf_path = clip_dir / "dual_scene.gltf"
        write_dual_clip_gltf(
            gltf_path,
            clip_name=clip["name"],
            role_names=ROLE_NAMES,
            rest_positions_by_role=rest_positions_by_role,
            local_quats_by_role=local_quats_by_role,
            root_translation_by_role={role_name: roles[role_name].root_translation for role_name in ROLE_NAMES},
            clip_frames=clip_frames,
            fps=roles[ROLE_NAMES[0]].fps,
        )
        clip_meta_path = clip_dir / "clip.meta.json"
        clip_meta_path.write_text(json.dumps(clip, indent=2), encoding="utf-8")
        clip_manifest["files"]["dual_scene_gltf"] = str(gltf_path)
        clip_manifest["files"]["clip_meta_json"] = str(clip_meta_path)
        manifest["clips"].append(clip_manifest)

    constraints_path = metadata_dir / "interaction_constraints.json"
    constraints_path.write_text(json.dumps({"constraints": interaction_constraints}, indent=2), encoding="utf-8")
    ik_path = metadata_dir / "ik_metadata.json"
    ik_path.write_text(json.dumps(ik_metadata, indent=2), encoding="utf-8")
    collision_path = metadata_dir / "collision_capsules.json"
    collision_path.write_text(json.dumps(collision_capsules, indent=2), encoding="utf-8")
    clips_path = metadata_dir / "clips.json"
    clips_path.write_text(json.dumps({"clips": clips}, indent=2), encoding="utf-8")
    clips_csv_path = metadata_dir / "clips.csv"
    write_csv_summary(clips_csv_path, clips)

    export_summary = {
        "source_motion_json": str(motion_path),
        "source_video": metadata.get("source_video"),
        "fps": float(metadata.get("fps") or roles[ROLE_NAMES[0]].fps),
        "frame_count": int(min(role.frame_count for role in roles.values())),
        "deliverables": {
            "skeletons": str(skeleton_dir),
            "clips": str(clips_dir),
            "metadata": str(metadata_dir),
            "full_sequence": str(full_dir),
        },
        "notes": [
            "BVH clips are the most immediately portable artifact for DCC tools.",
            "glTF clip scenes use animated transform nodes with simple cube markers and no skinned mesh.",
            "If Blender is installed later, these BVH clips can be converted to FBX/GLB there.",
        ],
    }
    summary_path = output_root / "export_manifest.json"
    summary_path.write_text(json.dumps(export_summary, indent=2), encoding="utf-8")

    manifest["files"] = {
        "interaction_constraints": str(constraints_path),
        "ik_metadata": str(ik_path),
        "collision_capsules": str(collision_path),
        "clips_json": str(clips_path),
        "clips_csv": str(clips_csv_path),
        "export_manifest": str(summary_path),
    }
    package_manifest_path = output_root / "package_manifest.json"
    package_manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"MVP asset package written to: {output_root}")


if __name__ == "__main__":
    main()
