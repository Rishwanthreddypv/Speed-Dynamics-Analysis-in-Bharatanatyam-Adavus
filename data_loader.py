"""
==============================================================
Data Loader
==============================================================
Reads all detected peak Excel files and merges the
Elapsed_Frames column into a single dataset.
==============================================================
"""

import os
import pandas as pd
import numpy as np

import config
import pipeline_paths as paths


# ============================================================
# READ A SINGLE LANDMARK FILE
# ============================================================

def read_landmark(file_path, landmark_name):

    print(f"Reading {landmark_name}...")

    df = pd.read_excel(file_path)

    if "Elapsed_Frames" not in df.columns:

        raise ValueError(
            f"'Elapsed_Frames' column missing in {file_path}"
        )

    data = df[["Elapsed_Frames"]].copy()

    data = data.dropna()

    data["Elapsed_Frames"] = data["Elapsed_Frames"].astype(float)

    data["Landmark"] = landmark_name

    return data


# ============================================================
# LOAD ALL LANDMARKS
# ============================================================

def load_all_data():

    output_dir = paths.OUTPUT1_DIR

    left_big_toe = os.path.join(
        output_dir,
        "detected_peaks_Left_Big_Toe.xlsx"
    )

    left_heel = os.path.join(
        output_dir,
        "detected_peaks_Left_Heel.xlsx"
    )

    right_big_toe = os.path.join(
        output_dir,
        "detected_peaks_Right_Big_Toe.xlsx"
    )

    right_heel = os.path.join(
        output_dir,
        "detected_peaks_Right_Heel.xlsx"
    )

    datasets = [

        read_landmark(
            left_big_toe,
            "Left Big Toe"
        ),

        read_landmark(
            left_heel,
            "Left Heel"
        ),

        read_landmark(
            right_big_toe,
            "Right Big Toe"
        ),

        read_landmark(
            right_heel,
            "Right Heel"
        )

    ]

    merged = pd.concat(
        datasets,
        ignore_index=True
    )

    merged_file = os.path.join(
        paths.STATISTICS_DIR,
        "merged_elapsed_frames.xlsx"
    )

    merged.to_excel(
        merged_file,
        index=False
    )

    print()

    print("=" * 60)

    print("Merged Dataset Created")

    print("=" * 60)

    print(f"Samples : {len(merged)}")

    print()

    print(
        merged.groupby("Landmark").size()
    )

    print("=" * 60)

    return merged


# ============================================================
# RETURN NUMPY ARRAY
# ============================================================

def get_feature_vector(df):

    return df["Elapsed_Frames"].values.reshape(-1, 1)


# ============================================================
# RUN
# ============================================================

def run():

    merged = load_all_data()

    X = get_feature_vector(merged)

    print()

    print("Feature Vector Shape")

    print(X.shape)

    return merged


# ============================================================
# STANDALONE TEST
# ============================================================

if __name__ == "__main__":

    paths.initialize(
        video_path="videos/S01.mp4"
    )

    run()