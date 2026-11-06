"""HTTP adapter for the existing Thattadavu analysis pipeline.

This adapter lives outside the analysis backend and calls the unchanged
``master.process_video`` entry point.
"""

from __future__ import annotations

import contextlib
import io
import os
import sys
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import cv2
import pandas as pd
from flask import Flask, abort, jsonify, request, send_from_directory
from werkzeug.utils import secure_filename


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DIST_DIR = Path(__file__).resolve().parent / "dist"
UPLOAD_DIR = Path(__file__).resolve().parent / "uploads"
RESULTS_DIR = PROJECT_ROOT / "results"
ALLOWED_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

app = Flask(__name__, static_folder=str(DIST_DIR), static_url_path="")
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024 * 1024

jobs: dict[str, dict] = {}
jobs_lock = threading.Lock()
# The pipeline uses module-level paths, so hosted analyses run serially.
executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="thattadavu")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def update_job(job_id: str, **changes) -> None:
    with jobs_lock:
        jobs[job_id].update(changes)
        jobs[job_id]["updated_at"] = utc_now()


class PipelineLog(io.TextIOBase):
    """Collect backend output and translate known messages into stages."""

    def __init__(self, job_id: str):
        self.job_id = job_id
        self.partial = ""

    def write(self, value: str) -> int:
        self.partial += value
        lines = self.partial.split("\n")
        self.partial = lines.pop()
        for line in lines:
            clean = line.strip()
            if not clean:
                continue
            with jobs_lock:
                jobs[self.job_id]["log"].append(clean)
                jobs[self.job_id]["log"] = jobs[self.job_id]["log"][-120:]
            upper = clean.upper()
            if "FRAME EXTRACTION" in upper:
                update_job(self.job_id, stage=0, message="Extracting video frames")
            elif "RTM" in upper and ("POSE" in upper or "DETECT" in upper):
                update_job(self.job_id, stage=1, message="Detecting pose with RTMW3D")
            elif "PEAK" in upper:
                update_job(self.job_id, stage=2, message="Tracking toe and heel movement")
            elif any(word in upper for word in ("MERGED DATASET", "K-MEANS", "GAUSSIAN", "FFT")):
                update_job(self.job_id, stage=3, message="Building statistics and outputs")
        return len(value)

    def flush(self) -> None:
        return None


def collect_artifacts(result_dir: Path) -> list[dict]:
    artifacts = []
    if not result_dir.is_dir():
        return artifacts
    for path in sorted(result_dir.rglob("*")):
        if not path.is_file() or "all_frames" in path.parts:
            continue
        relative = path.relative_to(result_dir).as_posix()
        artifacts.append({
            "name": path.name,
            "path": relative,
            "size": path.stat().st_size,
            "kind": path.suffix.lower().lstrip("."),
        })
    return artifacts


def collect_metrics(video_path: Path, result_dir: Path) -> dict:
    capture = cv2.VideoCapture(str(video_path))
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    capture.release()
    observations = 0
    keypoints_file = result_dir / "RTM_Detected_Frames" / "foot_keypoints.xlsx"
    if keypoints_file.exists():
        try:
            observations = int(len(pd.read_excel(keypoints_file)))
        except Exception:
            observations = 0
    return {
        "frames": frame_count,
        "fps": round(fps, 3),
        "duration": round(frame_count / fps, 2) if fps else 0,
        "keypoints": 4,
        "observations": observations,
    }


def run_pipeline(job_id: str, video_path: Path, result_name: str) -> None:
    update_job(job_id, status="running", message="Starting analysis", stage=0)
    log = PipelineLog(job_id)
    try:
        os.chdir(PROJECT_ROOT)
        import master

        with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            _, success, error = master.process_video(str(video_path))
        if not success:
            raise RuntimeError(error or "The analysis pipeline failed.")
        result_dir = RESULTS_DIR / result_name
        update_job(
            job_id,
            status="complete",
            stage=4,
            message="Analysis complete",
            artifacts=collect_artifacts(result_dir),
            metrics=collect_metrics(video_path, result_dir),
        )
    except Exception as exc:
        update_job(job_id, status="failed", message="Analysis failed", error=str(exc))


@app.get("/")
def index():
    return send_from_directory(DIST_DIR, "index.html")


@app.get("/api/health")
def health():
    return jsonify({"status": "ok", "pipeline": "master.process_video"})


@app.post("/api/analyze")
def analyze():
    upload = request.files.get("video")
    if upload is None or not upload.filename:
        return jsonify({"error": "A video file is required."}), 400
    original_name = secure_filename(upload.filename)
    extension = Path(original_name).suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        return jsonify({"error": "Supported formats: MP4, MOV, AVI, MKV, and WebM."}), 400

    job_id = uuid.uuid4().hex
    result_name = f"web_{job_id[:10]}"
    job_upload_dir = UPLOAD_DIR / job_id
    job_upload_dir.mkdir(parents=True, exist_ok=False)
    video_path = job_upload_dir / f"{result_name}{extension}"
    upload.save(video_path)

    created_at = utc_now()
    with jobs_lock:
        jobs[job_id] = {
            "id": job_id,
            "status": "queued",
            "stage": 0,
            "message": "Waiting for the analysis worker",
            "filename": original_name,
            "created_at": created_at,
            "updated_at": created_at,
            "artifacts": [],
            "metrics": {},
            "log": [],
            "error": None,
            "result_name": result_name,
        }
    executor.submit(run_pipeline, job_id, video_path, result_name)
    return jsonify({"job_id": job_id, "status": "queued"}), 202


@app.get("/api/jobs/<job_id>")
def job_status(job_id: str):
    with jobs_lock:
        job = jobs.get(job_id)
        if job is None:
            abort(404)
        public_job = {key: value for key, value in job.items() if key != "result_name"}
    return jsonify(public_job)


@app.get("/api/jobs/<job_id>/artifacts/<path:artifact_path>")
def artifact(job_id: str, artifact_path: str):
    with jobs_lock:
        job = jobs.get(job_id)
        if job is None or job["status"] != "complete":
            abort(404)
        result_name = job["result_name"]
    return send_from_directory(RESULTS_DIR / result_name, artifact_path, as_attachment=True)


if __name__ == "__main__":
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    app.run(host="127.0.0.1", port=8000, debug=False, threaded=True)
