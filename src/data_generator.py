# Generate synthetic multivariate ordinal data

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import time


# Naive implementation (can be improved)
def bos_model(n_cat, mu, pi):
    """
    Generate a single feature from BOS model
    :param n_cat: number of categories
    :param mu: position parameter
    :param pi: precision parameter
    :return: a single feature
    """
    # Perform stochastic binary search algorithm
    cur_e = list(np.arange(1, n_cat + 1))
    for i in range(n_cat):
        y = np.random.choice(cur_e, 1)[0]
        z = np.random.binomial(1, pi)
        e_minus = [e for e in cur_e if e < y]
        e_plus = [e for e in cur_e if e > y]
        e_equal = [e for e in cur_e if e == y]
        if z == 0:
            p = [
                len(e_minus) / len(cur_e),
                len(e_plus) / len(cur_e),
                len(e_equal) / len(cur_e),
            ]
            id = np.random.choice(3, 1, p=p)[0]
            cur_e = [e_minus, e_plus, e_equal][id]
        else:
            min_e = e_equal
            min_dist = abs(mu - e_equal[0])
            if len(e_minus) != 0:
                d_e_minus = min([abs(mu - e_minus[0]), abs(mu - e_minus[-1])])
                if d_e_minus < min_dist:
                    min_e = e_minus
                    min_dist = d_e_minus
            if len(e_plus) != 0:
                d_e_plus = min([abs(mu - e_plus[0]), abs(mu - e_plus[-1])])
                if d_e_plus < min_dist:
                    min_e = e_plus
                    min_dist = d_e_plus
            cur_e = min_e
        if len(cur_e) == 1:
            return cur_e[0]
    return cur_e[0]


def generate_data(n, p, n_cat, k, alpha, mu, pi, seed):
    """
    Generate synthetic multivariate ordinal dataset
    :param n: number of samples
    :param p: number of features
    :param n_cat: number of categories for each feature
    :param k: number of groups
    :param alpha: coefficient of the group indicator
    :param mu: position parameters of the BOS models
    :param pi: precision parameters of the BOS models
    :param seed: random seed
    :return: synthetic data of size (n, p) and group indicator of size (n,)
    """
    np.random.seed(seed)
    # generate group indicator with probability alphas
    w = np.random.choice(k, n, p=alpha, replace=True)
    # generate features
    x = np.zeros((n, p), dtype=int)
    for i in range(n):
        for j in range(p):
            x[i, j] = bos_model(n_cat[j], mu[w[i]][j], pi[w[i]][j])

    return x, w


def plot_hist_bos_model(n_cat, mu, pi, n_sample=10000):
    x = [bos_model(n_cat, mu, pi) for _ in range(n_sample)]
    sns.histplot(x, stat="density", discrete=True)
    plt.show()


if __name__ == "__main__":
    # Test bos_model
    # plot_hist_bos_model(5, 3, 0.5)

    seed = 0

    n = 10000
    p = 2
    n_cat = [5, 5]
    k = 2
    alpha = [0.5, 0.5]
    mu = [[2, 4], [4, 2]]
    pi = [[0.4, 0.4], [0.4, 0.4]]

    output_file = "../data/synthetic.csv"

    x, w = generate_data(n, p, n_cat, k, alpha, mu, pi, seed)
    # save data
    df = pd.DataFrame(x)
    df["w"] = w
    df.to_csv(os.path.join(os.path.dirname(__file__), output_file), index=False)

    # plot data
    sns.pairplot(
        df,
        hue="w",
        kind="hist",
        plot_kws={"alpha": 0.6, "bins": n_cat},
        diag_kws={"bins": n_cat[0]},
    )
    plt.show()
