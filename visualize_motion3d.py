#!/usr/bin/env python3
"""Render a quick 3D motion review video from motion3d_sequences.json."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np


ROLE_COLORS = {
    "character_A": (50, 205, 50),
    "character_B": (30, 144, 255),
}

SKELETONS = {
    17: [
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
    ],
    26: [
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
    ],
}


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parent
    default_output_root = project_root / "outputs" / "two_character_3d"
    parser = argparse.ArgumentParser(description="Render front/top review video for reconstructed 3D motion.")
    parser.add_argument(
        "--motion-json",
        default=str(default_output_root / "motion3d_sequences.json"),
    )
    parser.add_argument(
        "--output-video",
        default=str(default_output_root / "motion3d_review.mp4"),
    )
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--raw", action="store_true", help="Use unrefined world joints instead of refined joints.")
    return parser.parse_args()


def load_motion(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"Motion JSON not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def record_points(record: dict, raw: bool) -> Optional[np.ndarray]:
    key = "joints3d_world" if raw else "joints3d_world_refined"
    points = record.get(key)
    if points is None:
        return None
    rows = []
    for point in points:
        if point is None or any(value is None for value in point):
            rows.append([np.nan, np.nan, np.nan])
        else:
            rows.append([float(point[0]), float(point[1]), float(point[2])])
    arr = np.asarray(rows, dtype=np.float32)
    if arr.size == 0:
        return None
    return arr


def collect_bounds(roles: Dict[str, List[dict]], raw: bool) -> Tuple[np.ndarray, np.ndarray]:
    samples = []
    for records in roles.values():
        for record in records:
            points = record_points(record, raw)
            if points is None:
                continue
            finite = points[np.isfinite(points).all(axis=1)]
            if len(finite):
                samples.append(finite)

    if not samples:
        return np.array([-1.0, -1.0, -1.0], dtype=np.float32), np.array([1.0, 1.0, 1.0], dtype=np.float32)

    merged = np.concatenate(samples, axis=0)
    low = np.percentile(merged, 2, axis=0).astype(np.float32)
    high = np.percentile(merged, 98, axis=0).astype(np.float32)
    span = np.maximum(high - low, 1.0)
    padding = span * 0.15
    return low - padding, high + padding


def project(
    points: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
    rect: Tuple[int, int, int, int],
    axes: Tuple[int, int],
    flip_y: bool,
) -> np.ndarray:
    x0, y0, x1, y1 = rect
    width = max(1, x1 - x0)
    height = max(1, y1 - y0)
    span = np.maximum(high - low, 1e-6)

    coords = np.full((len(points), 2), np.nan, dtype=np.float32)
    finite = np.isfinite(points).all(axis=1)
    x_axis, y_axis = axes
    coords[finite, 0] = x0 + (points[finite, x_axis] - low[x_axis]) / span[x_axis] * width
    normalized_y = (points[finite, y_axis] - low[y_axis]) / span[y_axis]
    if flip_y:
        coords[finite, 1] = y1 - normalized_y * height
    else:
        coords[finite, 1] = y0 + normalized_y * height
    return coords


def draw_grid(canvas: np.ndarray, rect: Tuple[int, int, int, int], title: str) -> None:
    x0, y0, x1, y1 = rect
    cv2.rectangle(canvas, (x0, y0), (x1, y1), (70, 70, 70), 1)
    for i in range(1, 4):
        x = int(round(x0 + (x1 - x0) * i / 4))
        y = int(round(y0 + (y1 - y0) * i / 4))
        cv2.line(canvas, (x, y0), (x, y1), (35, 35, 35), 1)
        cv2.line(canvas, (x0, y), (x1, y), (35, 35, 35), 1)
    cv2.putText(canvas, title, (x0 + 16, y0 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (220, 220, 220), 2, cv2.LINE_AA)


def draw_skeleton(
    canvas: np.ndarray,
    coords: np.ndarray,
    keypoint_count: int,
    color: Tuple[int, int, int],
    radius: int = 4,
) -> None:
    skeleton = SKELETONS.get(keypoint_count, [])
    for start, end in skeleton:
        if start >= len(coords) or end >= len(coords):
            continue
        if not (np.isfinite(coords[start]).all() and np.isfinite(coords[end]).all()):
            continue
        cv2.line(
            canvas,
            tuple(np.round(coords[start]).astype(int)),
            tuple(np.round(coords[end]).astype(int)),
            color,
            2,
            cv2.LINE_AA,
        )
    for point in coords:
        if not np.isfinite(point).all():
            continue
        cv2.circle(canvas, tuple(np.round(point).astype(int)), radius, color, -1, cv2.LINE_AA)


def main() -> None:
    args = parse_args()
    motion_path = Path(args.motion_json).resolve()
    output_path = Path(args.output_video).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = load_motion(motion_path)
    roles: Dict[str, List[dict]] = payload["roles"]
    metadata = payload.get("metadata", {})
    keypoint_count = int(metadata.get("keypoint_count") or 0)
    if keypoint_count <= 0:
        sample = next(
            (
                points
                for records in roles.values()
                for record in records
                for points in [record_points(record, args.raw)]
                if points is not None
            ),
            None,
        )
        keypoint_count = len(sample) if sample is not None else 0

    frame_count = min(len(records) for records in roles.values()) if roles else 0
    if args.max_frames is not None:
        frame_count = min(frame_count, args.max_frames)
    if frame_count <= 0:
        raise SystemExit("No frames to render.")

    fps = float(args.fps or metadata.get("fps") or 24.0)
    low, high = collect_bounds(roles, args.raw)

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (args.width, args.height),
    )
    if not writer.isOpened():
        raise SystemExit(f"Failed to open video writer: {output_path}")

    margin = 48
    gap = 36
    panel_width = (args.width - 2 * margin - gap) // 2
    panel_height = args.height - 2 * margin
    front_rect = (margin, margin, margin + panel_width, margin + panel_height)
    top_rect = (margin + panel_width + gap, margin, margin + 2 * panel_width + gap, margin + panel_height)

    for frame_index in range(frame_count):
        canvas = np.full((args.height, args.width, 3), 18, dtype=np.uint8)
        draw_grid(canvas, front_rect, "Front view: X / Y")
        draw_grid(canvas, top_rect, "Top view: X / Z")

        for role_name, records in roles.items():
            points = record_points(records[frame_index], args.raw)
            if points is None:
                continue
            color = ROLE_COLORS.get(role_name, (230, 230, 230))
            front = project(points, low, high, front_rect, axes=(0, 1), flip_y=True)
            top = project(points, low, high, top_rect, axes=(0, 2), flip_y=False)
            draw_skeleton(canvas, front, keypoint_count, color)
            draw_skeleton(canvas, top, keypoint_count, color)
            cv2.putText(
                canvas,
                role_name,
                (front_rect[0] + 18, front_rect[1] + 62 + 28 * list(roles.keys()).index(role_name)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                color,
                2,
                cv2.LINE_AA,
            )

        cv2.putText(
            canvas,
            f"frame {frame_index:06d}",
            (margin, args.height - 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (220, 220, 220),
            2,
            cv2.LINE_AA,
        )
        writer.write(canvas)

    writer.release()
    print(f"Saved 3D review video to: {output_path}")


if __name__ == "__main__":
    main()
