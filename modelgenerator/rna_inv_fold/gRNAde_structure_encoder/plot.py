import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def plot_multiline_from_tsv(tsv_file):
    # Read the TSV file into a DataFrame
    df = pd.read_csv(tsv_file, sep="\t")

    # Check if specified columns exist in the DataFrame

    df_dict = {}
    for val in df["num_denoise_step"].unique():
        df_dict[val] = df[df["num_denoise_step"] == val]
        plt.plot(
            df_dict[val]["MASK_RATIO"],
            df_dict[val]["Best"],
            label="num_denoise_step=" + str(val),
        )
    plt.plot(
        df_dict[val]["MASK_RATIO"],
        [0.5278] * 7,
        label="Baseline (gRNAde)",
        color="black",
        linestyle="--",
    )

    # Add labels and title
    plt.xlabel("MASK_RATIO")
    plt.ylabel("Recovery rate")
    plt.title("Different num_denoise_step")
    plt.legend()
    plt.savefig("num_denoise_step.jpeg")
    plt.clf()

    df_dict = {}
    for val in df["MASK_RATIO"].unique():
        df_dict[val] = df[df["MASK_RATIO"] == val]
        plt.plot(
            df_dict[val]["num_denoise_step"],
            df_dict[val]["Best"],
            label="MASK_RATIO=" + str(val),
        )
    plt.plot(
        df_dict[val]["num_denoise_step"],
        [0.5278] * 5,
        label="Dashed Line",
        color="black",
        linestyle="--",
    )
    # Add labels and title
    plt.xlabel("num_denoise_step")
    plt.ylabel("Recovery rate")
    plt.title("Different MASK_RATIO")
    plt.legend()
    plt.savefig("MASK_RATIO.jpeg")
    plt.clf()

    num_denoise_steps = [1, 5, 10, 30, 50]
    MASK_RATIOs = [0.01, 0.05, 0.10, 0.20, 0.30, 0.50, 0.70]

    grid = df["Best"].to_numpy().reshape((7, 5))[::-1]
    plt.imshow(grid, cmap="Reds_r")
    plt.xticks(ticks=np.arange(5), labels=num_denoise_steps)
    plt.yticks(ticks=[6, 5, 4, 3, 2, 1, 0], labels=MASK_RATIOs)
    plt.colorbar(label="Recovery rate")
    plt.xlabel("num_denoise_step")
    plt.ylabel("MASK_RATIO")
    plt.title("Recovery rate heatmap")
    plt.savefig("Grid_view.jpeg")
    plt.clf()


# Example usage
tsv_file = "/home/shuxian.zou/downstream_szn/Inverse_folding/geometric-rna-design/Log_mlm_demo2 copy.tsv"  # Replace with the path to your TSV file

plot_multiline_from_tsv(tsv_file)
