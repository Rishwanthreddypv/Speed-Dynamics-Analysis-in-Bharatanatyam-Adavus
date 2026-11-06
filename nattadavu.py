"""Extract video frames while tracking 3D foot points with RTMW3D."""

from __future__ import annotations

import argparse
import csv
import math
import time
from pathlib import Path

import cv2
import matplotlib
import numpy as np
from rtmlib import Wholebody3d

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# COCO-WholeBody indices returned as independent 3D points by RTMW3D.
# Small-toe keypoints (18 and 21) are intentionally omitted.
FOOT_KEYPOINTS = {
    17: "left_big_toe",
    19: "left_heel",
    20: "right_big_toe",
    22: "right_heel",
}

DISPLAY_NAMES = {name: name.replace("_", " ").title() for name in FOOT_KEYPOINTS.values()}
POINT_COLORS = {
    17: (255, 80, 40),
    19: (255, 200, 40),
    20: (40, 80, 255),
    22: (40, 200, 255),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Track RTMW3D x/y/z foot coordinates for a video."
    )
    parser.add_argument("--video", type=Path, default=Path("Nattadavu/N01.mp4"))
    parser.add_argument("--seconds", type=float, default=20.0)
    parser.add_argument(
        "--full-video",
        action="store_true",
        help="Process every frame in the video instead of limiting by --seconds.",
    )
    parser.add_argument("--confidence", type=float, default=0.30)
    parser.add_argument("--output-dir", type=Path, default=Path("Nattadavu/results"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--backend", default="onnxruntime")
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument("--display-scale", type=float, default=0.65)
    return parser.parse_args()


def choose_person(
    keypoints_2d: np.ndarray,
    scores: np.ndarray,
    previous_center: np.ndarray | None,
) -> int:
    """Keep the same dancer when more than one person is detected."""
    if len(keypoints_2d) == 1 or previous_center is None:
        return int(np.argmax(np.nanmean(scores, axis=1)))

    # The midpoint between the hip keypoints is a stable body center.
    centers = (keypoints_2d[:, 11, :] + keypoints_2d[:, 12, :]) / 2.0
    distances = np.linalg.norm(centers - previous_center, axis=1)
    return int(np.argmin(distances))


def extract_frames(
    video: Path,
    frames_dir: Path,
    seconds: float | None,
) -> tuple[float, int, int, list[Path]]:
    """Write the requested video segment as numbered JPEG frames."""
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video.resolve()}")

    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if not math.isfinite(fps) or fps <= 0 or width <= 0 or height <= 0:
        capture.release()
        raise RuntimeError("Video has invalid FPS or frame dimensions")

    frames_dir.mkdir(parents=True, exist_ok=True)
    frame_limit = int(math.ceil(seconds * fps)) if seconds is not None else None
    frame_paths: list[Path] = []

    try:
        frame_number = 0
        while frame_limit is None or frame_number < frame_limit:
            ok, frame = capture.read()
            if not ok:
                break
            frame_path = frames_dir / f"frame_{frame_number:06d}.jpg"
            if not cv2.imwrite(str(frame_path), frame):
                raise RuntimeError(f"Could not write frame: {frame_path}")
            frame_paths.append(frame_path)
            frame_number += 1
    finally:
        capture.release()

    return fps, width, height, frame_paths


def plot_keypoint(
    rows: list[dict[str, float | int | str]],
    keypoint_name: str,
    output_path: Path,
) -> None:
    """Plot X/Y image pixels and RTMW3D Z depth against frame number."""
    point_rows = [row for row in rows if row["keypoint"] == keypoint_name]
    frames = np.array([int(row["frame"]) for row in point_rows])
    x_pixels = np.array([float(row["x_pixel"]) for row in point_rows])
    y_pixels = np.array([float(row["y_pixel"]) for row in point_rows])
    z_depth = np.array([float(row["z_depth"]) for row in point_rows])

    figure, pixel_axis = plt.subplots(figsize=(15, 7))
    depth_axis = pixel_axis.twinx()

    pixel_axis.plot(frames, x_pixels, color="#1565C0", linewidth=1.3, label="X pixel")
    pixel_axis.plot(frames, y_pixels, color="#D32F2F", linewidth=1.3, label="Y pixel")
    depth_axis.plot(
        frames,
        z_depth,
        color="#2E7D32",
        linewidth=1.3,
        label="Z relative depth",
    )

    pixel_axis.set_title(f"{DISPLAY_NAMES[keypoint_name]} X/Y/Z Movement")
    pixel_axis.set_xlabel("Frame number")
    pixel_axis.set_ylabel("X and Y location (pixels)")
    depth_axis.set_ylabel("RTMW3D relative Z depth", color="#2E7D32")
    depth_axis.tick_params(axis="y", labelcolor="#2E7D32")
    pixel_axis.grid(True, alpha=0.3)

    lines = pixel_axis.get_lines() + depth_axis.get_lines()
    pixel_axis.legend(lines, [line.get_label() for line in lines], loc="best")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_coordinate_overview(
    rows: list[dict[str, float | int | str]],
    coordinate: str,
    output_path: Path,
) -> None:
    """Plot one coordinate for every tracked foot keypoint against frame number."""
    coordinate_config = {
        "x_pixel": ("X-Coordinate Variation", "X-coordinate position (pixels)"),
        "y_pixel": ("Y-Coordinate Variation", "Y-coordinate position (pixels)"),
        "z_depth": ("Relative Z-Depth Variation", "RTMW3D relative Z-depth"),
    }
    if coordinate not in coordinate_config:
        raise ValueError(f"Unsupported coordinate: {coordinate}")

    title, y_label = coordinate_config[coordinate]
    line_colors = {
        "left_big_toe": "#1565C0",
        "left_heel": "#00ACC1",
        "right_big_toe": "#D32F2F",
        "right_heel": "#F57C00",
    }

    figure, axis = plt.subplots(figsize=(15, 7))
    for keypoint_name in FOOT_KEYPOINTS.values():
        point_rows = [row for row in rows if row["keypoint"] == keypoint_name]
        frames = np.asarray([int(row["frame"]) for row in point_rows])
        values = np.asarray([float(row[coordinate]) for row in point_rows])
        axis.plot(
            frames,
            values,
            linewidth=1.25,
            color=line_colors[keypoint_name],
            label=DISPLAY_NAMES[keypoint_name],
        )

    axis.set_title(f"Foot Keypoint {title}")
    axis.set_xlabel("Frame number")
    axis.set_ylabel(y_label)
    axis.grid(True, alpha=0.25)
    axis.legend(loc="best", ncol=2)
    axis.margins(x=0.01)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if not args.video.is_file():
        raise FileNotFoundError(f"Video not found: {args.video.resolve()}")
    if not args.full_video and args.seconds <= 0:
        raise ValueError("--seconds must be greater than zero")
    if args.display_scale <= 0:
        raise ValueError("--display-scale must be greater than zero")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    segment_seconds = None if args.full_video else args.seconds
    frames_dir = args.output_dir / (
        "frames_full_video" if args.full_video else f"frames_first_{args.seconds:g}_seconds"
    )
    segment_label = "the full video" if args.full_video else f"the first {args.seconds:g} seconds"
    print(f"Extracting {segment_label} into individual frames...", flush=True)
    fps, width, height, frame_paths = extract_frames(
        args.video, frames_dir, segment_seconds
    )
    if not frame_paths:
        raise RuntimeError("No frames were extracted")

    print(
        f"Extracted {len(frame_paths)} frames at {fps:.3f} FPS "
        f"({width}x{height}) into {frames_dir}",
        flush=True,
    )
    print("Initializing RTMLib Wholebody3d (RTMW3D)...", flush=True)
    model = Wholebody3d(mode="balanced", backend=args.backend, device=args.device)

    video_path = args.output_dir / "foot_keypoints_annotated.mp4"
    writer = cv2.VideoWriter(
        str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not create output video: {video_path}")

    rows: list[dict[str, float | int | str]] = []
    previous_center: np.ndarray | None = None
    started = time.perf_counter()
    window_name = "RTMLib Foot Tracking - press Q to stop"

    try:
        for frame_number, frame_path in enumerate(frame_paths):
            frame = cv2.imread(str(frame_path))
            if frame is None:
                raise RuntimeError(f"Could not read extracted frame: {frame_path}")

            keypoints_3d, scores, _simcc, keypoints_2d = model(frame)
            detected = keypoints_2d is not None and scores is not None and len(keypoints_2d) > 0

            if detected:
                person = choose_person(keypoints_2d, scores, previous_center)
                xy = np.asarray(keypoints_2d[person], dtype=float)
                xyz = np.asarray(keypoints_3d[person], dtype=float)
                confidence = np.asarray(scores[person], dtype=float)
                previous_center = (xy[11] + xy[12]) / 2.0

            for point_id, point_name in FOOT_KEYPOINTS.items():
                valid = detected and point_id < min(len(xy), len(xyz), len(confidence))
                score = float(confidence[point_id]) if valid else math.nan
                valid = valid and score >= args.confidence
                x_pixel = float(xy[point_id, 0]) if valid else math.nan
                y_pixel = float(xy[point_id, 1]) if valid else math.nan
                z_depth = float(xyz[point_id, 2]) if valid else math.nan

                rows.append(
                    {
                        "frame": frame_number,
                        "time_seconds": frame_number / fps,
                        "keypoint_id": point_id,
                        "keypoint": point_name,
                        "x_pixel": x_pixel,
                        "y_pixel": y_pixel,
                        "z_depth": z_depth,
                        "confidence": score,
                    }
                )

                if valid:
                    location = (int(round(x_pixel)), int(round(y_pixel)))
                    color = POINT_COLORS[point_id]
                    cv2.circle(frame, location, 10, color, -1)
                    cv2.circle(frame, location, 13, (255, 255, 255), 2)
                    cv2.putText(
                        frame,
                        f"{DISPLAY_NAMES[point_name]} Z={z_depth:.3f}",
                        (location[0] + 14, location[1] - 12),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        color,
                        2,
                        cv2.LINE_AA,
                    )

            cv2.putText(
                frame,
                f"Frame {frame_number}/{len(frame_paths) - 1}",
                (25, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            writer.write(frame)

            if not args.no_display:
                display = cv2.resize(
                    frame,
                    None,
                    fx=args.display_scale,
                    fy=args.display_scale,
                    interpolation=cv2.INTER_AREA,
                )
                cv2.imshow(window_name, display)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    print("Display stopped by user; saving processed results.", flush=True)
                    break

            processed = frame_number + 1
            if processed % 30 == 0 or processed == len(frame_paths):
                elapsed = time.perf_counter() - started
                print(
                    f"Processed {processed}/{len(frame_paths)} frames "
                    f"({processed / fps:.1f}s of video, {elapsed:.1f}s elapsed)",
                    flush=True,
                )
    finally:
        writer.release()
        if not args.no_display:
            cv2.destroyAllWindows()

    if not rows:
        raise RuntimeError("No frames were processed")

    csv_path = args.output_dir / "foot_keypoints_xyz.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    for point_name in FOOT_KEYPOINTS.values():
        graph_path = args.output_dir / f"{point_name}_xyz_by_frame.png"
        plot_keypoint(rows, point_name, graph_path)
        print(f"Saved graph: {graph_path}", flush=True)

    overview_graphs = {
        "x_pixel": "x_coordinates_vs_frame.png",
        "y_pixel": "y_coordinates_vs_frame.png",
        "z_depth": "z_depth_vs_frame.png",
    }
    for coordinate, filename in overview_graphs.items():
        graph_path = args.output_dir / filename
        plot_coordinate_overview(rows, coordinate, graph_path)
        print(f"Saved graph: {graph_path}", flush=True)

    print(f"Saved {len(rows):,} observations: {csv_path}", flush=True)
    print(f"Saved annotated video: {video_path}", flush=True)


if __name__ == "__main__":
    main()
