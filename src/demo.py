# Code to demonstrate the code

import os
import sys
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from time import perf_counter


from aecm import AECM_BOS, AECM_GOD


def process_data(data_path):
    """
    Process data from csv file
    """
    # Check if the file is a valid csv file
    if not data_path.endswith(".csv"):
        print("Data file should be a csv file!")
        with open("demo_failure.txt", "w") as f:
            f.write("Data file should be a csv file!")
        sys.exit(1)

    # Check if file has header
    try:
        first_line = open(data_path).readline().strip().split(",")
        has_header = False
        for i in range(len(first_line)):
            try:
                int(first_line[i])
            except ValueError:
                has_header = True
                break

        if has_header:
            dataframe = pd.read_csv(data_path, header="infer")
        else:
            dataframe = pd.read_csv(
                data_path,
                names=["Feature {}".format(i) for i in range(len(first_line))],
            )
    except Exception as e:
        print("Error reading the file: ", e)
        with open("demo_failure.txt", "w") as f:
            f.write("Invalid file format!")
        sys.exit(1)
    data = dataframe.values

    ma, mi = np.max(data, axis=0), np.min(data, axis=0)
    n_cat = ma - mi + 1
    shifted_data = data - mi[np.newaxis, :] + 1
    return shifted_data, n_cat, dataframe


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument(
        "--data_path", type=str, default="data/input.csv", help="path to data"
    )
    parser.add_argument(
        "--output_dir", type=str, default="output", help="path to output"
    )
    parser.add_argument(
        "--method", type=str, default="bos", help="method to use (bos or god)"
    )
    parser.add_argument(
        "--init",
        type=str,
        default="random",
        help="initialization method (random or kmeans)",
    )
    parser.add_argument("--n_clusters", type=int, default=3, help="number of clusters")
    parser.add_argument("--n_iter", type=int, default=100, help="number of iterations")
    parser.add_argument(
        "--eps", type=float, default=1e-8, help="epsilon for convergence"
    )
    parser.add_argument("--verbose", type=bool, default=False, help="verbose")

    args = parser.parse_args()

    if not os.path.exists(args.data_path):
        print("Data file does not exist!")
        print(args.data_path)
        with open("demo_failure.txt", "w") as f:
            f.write("Data file can't be found!")
        sys.exit(1)

    data, n_cat, df = process_data(args.data_path)
    d = data.shape[1]
    m = n_cat

    t = perf_counter()
    if args.method == "bos":
        clustering = AECM_BOS(
            nb_clusters=args.n_clusters,
            nb_features=d,
            ms=m,
            data=data,
            verbose=args.verbose,
        )
    else:
        clustering = AECM_GOD(
            nb_clusters=args.n_clusters,
            nb_features=d,
            ms=m,
            data=data,
            verbose=args.verbose,
        )
    ll_list = clustering.fit(
        epsilon_aecm=args.eps, max_iter_aecm=args.n_iter, initialization=args.init
    )
    labels = clustering.labels

    print("Method: " + "BOS model" if args.method == "bos" else "GOD model")
    print("Number of clusters: ", args.n_clusters)
    print("Initialization: ", args.init)
    print(f"Time: {perf_counter() - t:.2f}s")

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(args.output_dir, "estimated_params.txt"), "w") as f:
        for i in range(args.n_clusters):
            f.write("Cluster {}\n".format(i))
            f.write("alpha: {}\n".format(clustering.alphas[i]))
            f.write("mu: {}\n".format(clustering.mus[i]))
            f.write("pi: {}\n".format(clustering.pis[i]))
            f.write("\n")

    # Plot parameters of each cluster as a bar plot
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid")
    patterns = ["/", "o", "\\", "*", "-", "|", "+", "O", "x", "."]

    fig, ax = plt.subplots()
    width = 0.7 / d
    for i in range(d):
        ax.bar(
            np.arange(args.n_clusters) + width * i,
            clustering.mus[:, i],
            width,
            label=df.columns[i],
            hatch=patterns[i % len(patterns)],
        )
    ax.set_ylabel("Mean")
    ax.set_xlabel("Cluster")
    ax.set_title("Estimated means for each cluster")
    # Center the labels
    ax.set_xticks(np.arange(args.n_clusters) + width * (d - 1) / 2)
    ax.set_xticklabels(np.arange(args.n_clusters))
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "estimated_mean.png"))

    # plot weight of each cluster in the dataset
    fig, ax = plt.subplots()
    ax.bar(np.arange(args.n_clusters), clustering.alphas)
    ax.set_ylabel("Proportion")
    ax.set_xlabel("Cluster")
    ax.set_title("Estimated proportion of each cluster")
    ax.set_xticks(np.arange(args.n_clusters))
    ax.set_xticklabels(np.arange(args.n_clusters))
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "estimated_proportion.png"))

    # write the dataset with the cluster labels
    df["Cluster"] = labels
    df.to_csv(os.path.join(args.output_dir, "clustered_data.csv"), index=False)
