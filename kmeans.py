import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans

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
        "kmeans_results.xlsx"
    )

    OUTPUT_GRAPH = os.path.join(
        paths.STATISTICS_DIR,
        "04_1D_KMeans.png"
    )

    # ============================================================
    # LOAD DATA
    # ============================================================

    df = pd.read_excel(INPUT_FILE)

    X = df["Elapsed_Frames"].values.reshape(-1, 1)

    # ============================================================
    # KMEANS
    # ============================================================

    kmeans = KMeans(
        n_clusters=config.NUMBER_OF_CLUSTERS,
        random_state=config.RANDOM_STATE,
        n_init=config.N_INIT,
        max_iter=config.MAX_ITER
    )

    labels = kmeans.fit_predict(X)

    centroids = kmeans.cluster_centers_.flatten()

    # ============================================================
    # SORT CENTROIDS
    # ============================================================

    order = np.argsort(centroids)

    speed_map = {
        order[0]: "Fast",
        order[1]: "Medium",
        order[2]: "Slow"
    }

    df["Cluster"] = labels
    df["Speed"] = df["Cluster"].map(speed_map)

    # ============================================================
    # SAVE EXCEL
    # ============================================================

    df.to_excel(
        OUTPUT_EXCEL,
        index=False
    )

    # ============================================================
    # PRINT CENTROIDS
    # ============================================================

    print("\nCentroids\n")

    sorted_centroids = []

    for cluster in order:

        centroid = centroids[cluster]

        speed = speed_map[cluster]

        sorted_centroids.append(
            (speed, centroid)
        )

        print(f"{speed:8s} : {centroid:.2f} frames")

    # ============================================================
    # GRAPH
    # ============================================================

    plt.figure(figsize=(18, 3))

    colors = {

        "Fast": config.FAST_COLOUR,

        "Medium": config.MEDIUM_COLOUR,

        "Slow": config.SLOW_COLOUR

    }

    for speed in ["Fast", "Medium", "Slow"]:

        subset = df[df["Speed"] == speed]

        plt.scatter(

            subset["Elapsed_Frames"],

            np.zeros(len(subset)),

            s=120,

            color=colors[speed],

            edgecolor="black",

            label=speed

        )

    for speed, centroid in sorted_centroids:

        plt.axvline(

            centroid,

            linestyle="--",

            linewidth=2,

            color=config.CENTROID_COLOUR

        )

        plt.scatter(

            centroid,

            0,

            s=250,

            color="yellow",

            edgecolor="black",

            zorder=10

        )

        plt.text(

            centroid,

            0.06,

            f"{speed}\n{centroid:.2f}",

            ha="center",

            fontsize=10,

            fontweight="bold"

        )

    plt.yticks([])

    plt.xlabel("Elapsed Frames")

    plt.title("1D K-Means Clustering")

    plt.legend()

    plt.grid(axis="x")

    plt.tight_layout()

    plt.savefig(
        OUTPUT_GRAPH,
        dpi=config.DPI
    )

    plt.close()

    print("\n======================================")
    print("K-Means Completed")
    print("======================================")
    print(f"Excel : {OUTPUT_EXCEL}")
    print(f"Graph : {OUTPUT_GRAPH}")
    print("======================================")

    return df


if __name__ == "__main__":

    paths.initialize(
        video_path="videos/S01.mp4"
    )

    run()