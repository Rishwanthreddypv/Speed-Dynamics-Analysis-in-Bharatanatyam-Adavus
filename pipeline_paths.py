import os

BASE_DIR = ""

VIDEO_PATH = ""

FRAMES_DIR = ""

RTM_DIR = ""

OUTPUT1_DIR = ""

STATISTICS_DIR = ""

FOOT_KEYPOINTS = ""


def initialize(video_path, results_root="results"):

    global BASE_DIR
    global VIDEO_PATH
    global FRAMES_DIR
    global RTM_DIR
    global OUTPUT1_DIR
    global STATISTICS_DIR
    global FOOT_KEYPOINTS

    VIDEO_PATH = video_path

    video_name = os.path.splitext(
        os.path.basename(video_path)
    )[0]

    BASE_DIR = os.path.join(
        results_root,
        video_name
    )

    FRAMES_DIR = os.path.join(
        BASE_DIR,
        "all_frames"
    )

    RTM_DIR = os.path.join(
        BASE_DIR,
        "RTM_Detected_Frames"
    )

    OUTPUT1_DIR = os.path.join(
        BASE_DIR,
        "output_1"
    )

    STATISTICS_DIR = os.path.join(
        BASE_DIR,
        "Statistics_Output"
    )

    FOOT_KEYPOINTS = os.path.join(
        RTM_DIR,
        "foot_keypoints.xlsx"
    )

    os.makedirs(FRAMES_DIR, exist_ok=True)
    os.makedirs(RTM_DIR, exist_ok=True)
    os.makedirs(OUTPUT1_DIR, exist_ok=True)
    os.makedirs(STATISTICS_DIR, exist_ok=True)