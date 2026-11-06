import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import norm

import config
import pipeline_paths as paths


def run():

    # ============================================================
    # INPUT / OUTPUT
    # ============================================================

    INPUT_FILE = os.path.join(
        paths.STATISTICS_DIR,
        "kmeans_results.xlsx"
    )

    OUTPUT_EXCEL = os.path.join(
        paths.STATISTICS_DIR,
        "gaussian_statistics.xlsx"
    )

    OUTPUT_GRAPH = os.path.join(
        paths.STATISTICS_DIR,
        "05_Gaussian.png"
    )

    # ============================================================
    # LOAD DATA
    # ============================================================

    df = pd.read_excel(INPUT_FILE)

    clusters = ["Fast", "Medium", "Slow"]

    colors = {
        "Fast": config.FAST_COLOUR,
        "Medium": config.MEDIUM_COLOUR,
        "Slow": config.SLOW_COLOUR
    }

    statistics = []

    plt.figure(figsize=(15, 7))

    # ============================================================
    # PROCESS EACH CLUSTER
    # ============================================================

    for cluster in clusters:

        data = df[df["Speed"] == cluster]["Elapsed_Frames"].values

        total = len(data)

        if total < 2:
            print(f"Skipping {cluster}: insufficient samples.")
            continue

        mu = np.mean(data)

        sigma = np.std(
            data,
            ddof=1
        )

        variance = np.var(
            data,
            ddof=1
        )

        lower1 = mu - sigma
        upper1 = mu + sigma

        lower2 = mu - 2 * sigma
        upper2 = mu + 2 * sigma

        lower3 = mu - 3 * sigma
        upper3 = mu + 3 * sigma

        inside1 = np.sum(
            (data >= lower1) &
            (data <= upper1)
        )

        inside2 = np.sum(
            (data >= lower2) &
            (data <= upper2)
        )

        inside3 = np.sum(
            (data >= lower3) &
            (data <= upper3)
        )

        coverage1 = (inside1 / total) * 100
        coverage2 = (inside2 / total) * 100
        coverage3 = (inside3 / total) * 100

        statistics.append({

            "Cluster": cluster,
            "Samples": total,
            "Mean": mu,
            "Variance": variance,
            "Std Dev": sigma,

            "-1 Sigma": lower1,
            "+1 Sigma": upper1,

            "-2 Sigma": lower2,
            "+2 Sigma": upper2,

            "-3 Sigma": lower3,
            "+3 Sigma": upper3,

            "Points inside 1 Sigma": inside1,
            "Points inside 2 Sigma": inside2,
            "Points inside 3 Sigma": inside3,

            "1 Sigma Coverage (%)": coverage1,
            "2 Sigma Coverage (%)": coverage2,
            "3 Sigma Coverage (%)": coverage3

        })

        x = np.linspace(
            lower3 - 20,
            upper3 + 20,
            600
        )

        y = norm.pdf(
            x,
            mu,
            sigma
        )

        plt.plot(
            x,
            y,
            color=colors[cluster],
            linewidth=3,
            label=cluster
        )

        peak = norm.pdf(mu, mu, sigma)

        plt.scatter(
            mu,
            peak,
            color="black",
            s=140,
            zorder=5
        )

        plt.text(
            mu,
            peak + 0.001,
            f"μ={mu:.2f}",
            ha="center",
            fontsize=10,
            fontweight="bold"
        )

        for value, style in [

            (lower1, ":"),

            (upper1, ":"),

            (lower2, "--"),

            (upper2, "--"),

            (lower3, "-."),

            (upper3, "-.")

        ]:

            plt.axvline(
                value,
                color=colors[cluster],
                linestyle=style,
                linewidth=1.5
            )

        mask = (x >= lower3) & (x <= upper3)

        plt.fill_between(
            x[mask],
            y[mask],
            color=colors[cluster],
            alpha=0.15
        )

        plt.text(
            mu,
            peak * 0.75,
            f"1σ : {coverage1:.1f}%\n"
            f"2σ : {coverage2:.1f}%\n"
            f"3σ : {coverage3:.1f}%",
            ha="center",
            fontsize=9,
            bbox=dict(
                facecolor="white",
                alpha=0.8
            )
        )

    # ============================================================
    # SAVE EXCEL
    # ============================================================

    stats_df = pd.DataFrame(statistics)

    stats_df.to_excel(
        OUTPUT_EXCEL,
        index=False
    )

    # ============================================================
    # GRAPH
    # ============================================================

    plt.title(
        "Gaussian Distribution of K-Means Clusters",
        fontsize=16,
        fontweight="bold"
    )

    plt.xlabel("Elapsed Frames")

    plt.ylabel("Probability Density")

    plt.grid(True)

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        OUTPUT_GRAPH,
        dpi=config.DPI
    )

    plt.close()

    print("\n======================================")
    print("Gaussian Analysis Completed")
    print("======================================")
    print(f"Excel : {OUTPUT_EXCEL}")
    print(f"Graph : {OUTPUT_GRAPH}")
    print("======================================")

    return stats_df


if __name__ == "__main__":

    paths.initialize(
        video_path="videos/S01.mp4"
    )

    run()