import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import config
import pipeline_paths as paths


def run():

    # ============================================================
    # INPUT / OUTPUT
    # ============================================================

    INPUT_FILE = os.path.join(
        paths.STATISTICS_DIR,
        "merged_elapsed_frames.xlsx"
    )

    OUTPUT_EXCEL = os.path.join(
        paths.STATISTICS_DIR,
        "fft_results.xlsx"
    )

    OUTPUT_GRAPH = os.path.join(
        paths.STATISTICS_DIR,
        "06_FFT.png"
    )

    # ============================================================
    # LOAD DATA
    # ============================================================

    df = pd.read_excel(INPUT_FILE)

    signal = df["Elapsed_Frames"].values

    N = len(signal)

    if N < 2:
        raise ValueError(
            "Not enough samples for FFT."
        )

    # ============================================================
    # FFT
    # ============================================================

    fft_values = np.fft.fft(signal)

    frequencies = np.fft.fftfreq(
        N,
        d=1
    )

    # ============================================================
    # POSITIVE FREQUENCIES
    # ============================================================

    positive = frequencies >= 0

    frequencies = frequencies[positive]

    magnitude = np.abs(
        fft_values
    )[positive]

    # ============================================================
    # DOMINANT FREQUENCY
    # ============================================================

    if len(magnitude) > 1:

        dominant_index = np.argmax(
            magnitude[1:]
        ) + 1

        dominant_frequency = frequencies[
            dominant_index
        ]

        dominant_amplitude = magnitude[
            dominant_index
        ]

    else:

        dominant_frequency = 0

        dominant_amplitude = 0

    print("\n=========================================")
    print("FFT ANALYSIS")
    print("=========================================")

    print(f"Number of Samples      : {N}")
    print(f"Dominant Frequency     : {dominant_frequency:.4f}")
    print(f"Dominant Amplitude     : {dominant_amplitude:.4f}")

    # ============================================================
    # SAVE RESULTS
    # ============================================================

    fft_df = pd.DataFrame({

        "Frequency": frequencies,

        "Magnitude": magnitude

    })

    fft_df.to_excel(
        OUTPUT_EXCEL,
        index=False
    )

    # ============================================================
    # PLOT
    # ============================================================

    plt.figure(
        figsize=(
            config.FIGURE_WIDTH,
            config.FIGURE_HEIGHT
        )
    )

    plt.plot(
        frequencies,
        magnitude,
        linewidth=2
    )

    plt.scatter(
        dominant_frequency,
        dominant_amplitude,
        color="red",
        s=120,
        label="Dominant Frequency"
    )

    plt.text(
        dominant_frequency,
        dominant_amplitude,
        f"{dominant_frequency:.3f}",
        fontsize=10,
        ha="left"
    )

    plt.xlabel("Normalized Frequency")

    plt.ylabel("Magnitude")

    plt.title("FFT Spectrum of Elapsed Frames")

    plt.grid(True)

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        OUTPUT_GRAPH,
        dpi=config.DPI
    )

    plt.close()

    print("\n======================================")
    print("FFT Completed")
    print("======================================")
    print(f"Excel : {OUTPUT_EXCEL}")
    print(f"Graph : {OUTPUT_GRAPH}")
    print("======================================")

    return fft_df


if __name__ == "__main__":

    paths.initialize(
        video_path="videos/S01.mp4"
    )

    run()