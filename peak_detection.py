import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

import pipeline_paths as paths

# ============================================================
# PARAMETERS
# ============================================================

MAX_Y = 700          # Ignore minima above this Y value
MIN_DISTANCE = 20    # Minimum frame separation
PROMINENCE = 40      # Peak prominence

# ============================================================
# KEYPOINTS TO PROCESS
# ============================================================

KEYPOINTS = [
    "Left_Big_Toe",
    "Left_Heel",
    "Right_Big_Toe",
    "Right_Heel"
]


def run():

    # ============================================================
    # INPUT / OUTPUT
    # ============================================================

    INPUT_FILE = os.path.join(
        paths.RTM_DIR,
        "foot_keypoints.xlsx"
    )

    OUTPUT_FOLDER = paths.OUTPUT1_DIR

    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(
            f"Input file not found:\n{INPUT_FILE}"
        )

    os.makedirs(
        OUTPUT_FOLDER,
        exist_ok=True
    )

    # ============================================================
    # LOAD EXCEL
    # ============================================================

    df = pd.read_excel(INPUT_FILE)

    frames = df["Frame"].to_numpy()

    # ============================================================
    # PROCESS EACH KEYPOINT
    # ============================================================

    for keypoint in KEYPOINTS:

        column_name = f"{keypoint}_Y"

        if column_name not in df.columns:
            print(f"Column '{column_name}' not found. Skipping...")
            continue

        y = df[column_name].to_numpy()

        # --------------------------------------------------------
        # FIND MINIMA
        # --------------------------------------------------------

        signal = -y

        peaks, properties = find_peaks(
            signal,
            distance=MIN_DISTANCE,
            prominence=PROMINENCE
        )

        # --------------------------------------------------------
        # FILTER PEAKS
        # --------------------------------------------------------

        valid_peaks = []

        for p in peaks:
            if y[p] <= MAX_Y:
                valid_peaks.append(p)

        valid_peaks = np.array(valid_peaks)

        # --------------------------------------------------------
        # SAVE EXCEL
        # --------------------------------------------------------

        peak_frames = frames[valid_peaks]

        elapsed_frames = [""] + list(np.diff(peak_frames))

        results = pd.DataFrame({

            "Peak_Number": np.arange(
                1,
                len(valid_peaks) + 1
            ),

            "Frame": peak_frames,

            column_name: y[valid_peaks],

            "Elapsed_Frames": elapsed_frames

        })

        excel_path = os.path.join(
            OUTPUT_FOLDER,
            f"detected_peaks_{keypoint}.xlsx"
        )

        results.to_excel(
            excel_path,
            index=False
        )

        # --------------------------------------------------------
        # GRAPH
        # --------------------------------------------------------

        plt.figure(figsize=(18, 7))

        plt.plot(
            frames,
            y,
            color="blue",
            linewidth=1.2,
            label=f"{keypoint} Y"
        )

        plt.scatter(
            frames[valid_peaks],
            y[valid_peaks],
            color="red",
            s=80,
            label="Detected Peaks",
            zorder=5
        )

        for p in valid_peaks:

            plt.annotate(

                str(frames[p]),

                (frames[p], y[p]),

                textcoords="offset points",

                xytext=(0, -18),

                ha="center",

                fontsize=8,

                color="red",

                fontweight="bold"

            )

        plt.title(f"Detected Peaks - {keypoint}")

        plt.xlabel("Frame Number")

        plt.ylabel("Y Coordinate (Pixels)")

        plt.grid(True)

        plt.legend()

        plt.tight_layout()

        graph_path = os.path.join(

            OUTPUT_FOLDER,

            f"detected_peaks_{keypoint}.png"

        )

        plt.savefig(
            graph_path,
            dpi=300
        )

        plt.close()

        # --------------------------------------------------------
        # PRINT
        # --------------------------------------------------------

        print(f"\n{keypoint}")
        print("-" * 40)
        print(results)
        print(f"Number of Peaks : {len(valid_peaks)}")
        print(f"Excel Saved : {excel_path}")
        print(f"Graph Saved : {graph_path}")

    print("\n===================================")
    print("Processing Complete.")
    print(f"All outputs saved in '{OUTPUT_FOLDER}'")
    print("===================================")

    return OUTPUT_FOLDER


# ============================================================
# STANDALONE TEST
# ============================================================

if __name__ == "__main__":

    paths.initialize(
        video_path="videos/S01.mp4"
    )

    run()