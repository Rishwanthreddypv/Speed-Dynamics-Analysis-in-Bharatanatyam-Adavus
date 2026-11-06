import os
import cv2
import pipeline_paths as paths


def run():

    # ============================================================
    # INPUT / OUTPUT
    # ============================================================

    VIDEO_PATH = paths.VIDEO_PATH
    OUTPUT_DIR = paths.FRAMES_DIR

    if not VIDEO_PATH:
        raise ValueError(
            "VIDEO_PATH not initialized.\n"
            "Call pipeline_paths.initialize() first."
        )

    # ============================================================
    # OPEN VIDEO
    # ============================================================

    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        raise FileNotFoundError(
            f"Cannot open video:\n{VIDEO_PATH}"
        )

    # ============================================================
    # VIDEO INFORMATION
    # ============================================================

    fps = cap.get(cv2.CAP_PROP_FPS)

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    width = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    height = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    print("\n======================================")
    print("FRAME EXTRACTION")
    print("======================================")
    print(f"Video        : {os.path.basename(VIDEO_PATH)}")
    print(f"FPS          : {fps:.2f}")
    print(f"Total Frames : {total_frames}")
    print(f"Resolution   : {width} x {height}")
    print(f"Output Folder: {OUTPUT_DIR}")
    print("======================================\n")

    # ============================================================
    # EXTRACT ALL FRAMES
    # ============================================================

    frame_id = 0

    while True:

        success, frame = cap.read()

        if not success:
            print("\nReached end of video.")
            break

        frame_name = f"frame_{frame_id:06d}.jpg"

        frame_path = os.path.join(
            OUTPUT_DIR,
            frame_name
        )

        cv2.imwrite(
            frame_path,
            frame
        )

        if total_frames > 0:
            print(
                f"Saved Frame {frame_id + 1}/{total_frames}",
                end="\r"
            )
        else:
            print(
                f"Saved Frame {frame_id + 1}",
                end="\r"
            )

        frame_id += 1

    # ============================================================
    # RELEASE
    # ============================================================

    cap.release()

    # ============================================================
    # OUTPUT
    # ============================================================

    print("\n======================================")
    print("FRAME EXTRACTION COMPLETED")
    print("======================================")
    print(f"Total Frames Extracted : {frame_id}")
    print(f"Frames Saved To        : {OUTPUT_DIR}")
    print("======================================")

    return OUTPUT_DIR


# ============================================================
# STANDALONE TEST
# ============================================================

if __name__ == "__main__":

    paths.initialize(
        video_path="videos/S01.mp4"
    )

    run()