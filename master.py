"""
==============================================================
MASTER PIPELINE
==============================================================
Processes multiple Bharatanatyam videos in parallel.
Each video is processed independently.

Video
    ↓
Frame Extraction
    ↓
RTMPose
    ↓
Peak Detection
    ↓
Data Loader
    ↓
KMeans
    ↓
Gaussian
    ↓
FFT
==============================================================
"""

import os
import traceback
import importlib
from concurrent.futures import ProcessPoolExecutor, as_completed

import pipeline_paths as paths


# ============================================================
# SETTINGS
# ============================================================

VIDEOS_DIR = "videos"

RESULTS_DIR = "results"

MAX_WORKERS = 4        # Recommended: Start with 4


# ============================================================
# PROCESS A SINGLE VIDEO
# ============================================================

def process_video(video_path):
    


    video_name = os.path.splitext(
        os.path.basename(video_path)
    )[0]
    print(
        f"PID = {os.getpid()} | "
        f"Video = {video_name}"
    )
    try:

        print("\n" + "=" * 70)
        print(f"Starting : {video_name}")
        print("=" * 70)

        # ----------------------------------------------------
        # Initialize paths
        # ----------------------------------------------------

        paths.initialize(
            video_path,
            RESULTS_DIR
        )

        # ----------------------------------------------------
        # Import AFTER initialization
        # ----------------------------------------------------

        import extract_frames
        import rtm_pose
        import peak_detection
        import data_loader
        import kmeans
        import gaussian
        import fft

        # ----------------------------------------------------
        # Reload modules
        # ----------------------------------------------------

        importlib.reload(extract_frames)
        importlib.reload(rtm_pose)
        importlib.reload(peak_detection)
        importlib.reload(data_loader)
        importlib.reload(kmeans)
        importlib.reload(gaussian)
        importlib.reload(fft)

        # ----------------------------------------------------
        # Run Complete Pipeline
        # ----------------------------------------------------

        extract_frames.run()

        rtm_pose.run()

        peak_detection.run()

        data_loader.run()

        kmeans.run()

        gaussian.run()

        fft.run()

        print(f"\n✓ Finished : {video_name}")

        return (
            video_name,
            True,
            None
        )

    except Exception:

        error = traceback.format_exc()

        print(f"\n✗ Failed : {video_name}")

        print(error)

        return (
            video_name,
            False,
            error
        )


# ============================================================
# MAIN
# ============================================================

def main():

    if not os.path.exists(VIDEOS_DIR):

        raise FileNotFoundError(
            f"Videos folder not found:\n{VIDEOS_DIR}"
        )

    videos = sorted(

        [

            os.path.join(VIDEOS_DIR, file)

            for file in os.listdir(VIDEOS_DIR)

            if file.lower().endswith(".mp4")

        ]

    )

    if len(videos) == 0:

        raise RuntimeError(
            "No MP4 videos found."
        )

    print("\n" + "=" * 70)
    print("BHARATANATYAM ANALYSIS PIPELINE")
    print("=" * 70)

    print(f"Videos Found : {len(videos)}")

    print(f"Parallel Workers : {MAX_WORKERS}")

    print("=" * 70)

    successful = []

    failed = []

    # ========================================================
    # PARALLEL EXECUTION
    # ========================================================

    with ProcessPoolExecutor(
        max_workers=MAX_WORKERS
    ) as executor:

        futures = {

            executor.submit(
                process_video,
                video
            ): video

            for video in videos

        }

        for future in as_completed(futures):

            try:
                video_name, success, error = future.result()
            except Exception as e:
                print(f"\nWorker crashed: {e}")
                failed.append("Unknown")
                continue
            if success:

                successful.append(video_name)

            else:

                failed.append(video_name)

    # ========================================================
    # SUMMARY
    # ========================================================

    print("\n" + "=" * 70)

    print("PIPELINE COMPLETED")

    print("=" * 70)

    print(f"Successful : {len(successful)}")

    print(f"Failed     : {len(failed)}")

    if successful:

        print("\nCompleted Videos:")

        for video in successful:

            print(f"  ✓ {video}")

    if failed:

        print("\nFailed Videos:")

        for video in failed:

            print(f"  ✗ {video}")

    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()