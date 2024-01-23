# Code to demonstrate the code

import os
import sys
import argparse
from pathlib import Path
import numpy as np


from aecm import AECM_BOS, AECM_GOD


def process_data(data_path):
    """
    Process data from csv file
    """
    data = np.loadtxt(data_path, delimiter=",", dtype=int)

    ma, mi = np.max(data, axis=0), np.min(data, axis=0)
    n_cat = ma - mi + 1
    shifted_data = data - mi[np.newaxis, :] + 1
    # TODO decide if we want to handle missing values
    return shifted_data, n_cat


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
        sys.exit(1)

    data, n_cat = process_data(args.data_path)
    d = data.shape[1]
    m = n_cat

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
    ll_list = clustering.fit(epsilon_aecm=args.eps, max_iter_aecm=args.n_iter)
    labels = clustering.labels

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(args.output_dir, "estimated_params.txt"), "w") as f:
        f.write(
            "Estimated alpha: {}, Estimated mu: {}, Estimated pi: {}".format(
                clustering.alphas, clustering.mus, clustering.pis
            )
        )
