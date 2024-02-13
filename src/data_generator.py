# Generate synthetic multivariate ordinal data

from typing import Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import time

try:
    from .god_model_generator import god_model_generator
except ImportError:
    from god_model_generator import god_model_generator


# Naive implementation (can be improved)
def bos_model(n_cat: int, mu: int, pi: float) -> int:
    """
    Generate a single feature from BOS model

    Parameters
    ----------
    n_cat : int
        number of categories
    mu : int
        position parameter
    pi : float
        precision parameter

    Returns
    -------
    int
        x ~ BOS(n_cat, mu, pi)
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


def bos_model_sample(
    m: int,
    mu: int,
    pi: float,
    n_sample: int,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Generate n_sample x in [[1, n_cat]] from BOS model with parameters mu and pi

    Args:
        m: number of categories
        mu: true category
        pi: probability of error
        n_sample: number of samples
        seed: random seed

    Return:
        x: generated categories (n_sample)
    """
    assert 1 <= mu <= m, f"mu must be in [[1, m]] but got {mu}"
    assert 0 < pi <= 1, f"pi must be in ]0.5, 1] but got {pi}"
    assert n_sample > 0, f"n_sample must be > 0 but got {n_sample}"
    if seed is not None:
        np.random.seed(seed)
    x = np.empty(n_sample, dtype=int)
    for i in range(n_sample):
        x[i] = bos_model(m, mu, pi)
    return x


def generate_data(
    n: int,
    p: int,
    n_cat: list[int],
    k: int,
    alpha: list[float],
    mu: list[list[int]],
    pi: list[list[float]],
    seed: int,
    model: str = "bos",
):
    """
    Generate synthetic multivariate ordinal dataset

    Parameters
    ----------
    n : int
        number of samples
    p : int
        number of features
    n_cat : list[int] of length p
        number of categories for each feature
    k : int
        number of groups or clusters
    alpha : list[float] of length k
        coefficient of the group indicator
    mu : list[list[int]] of size (k, p)
        position parameters of the BOS models, mu[i, j] : mu for group i, feature j
    pi : list[list[float]] of size (k, p)
        precision parameters of the BOS models, pi[i, j] : pi for group i, feature j
    seed : int
        random seed
    model : str
        model type, "bos" or "god"

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        synthetic data of size (n, p) and group indicator of size (n,)
    """
    np.random.seed(seed)
    # generate group indicator with probability alphas
    w = np.random.choice(k, n, p=alpha, replace=True)
    # w[i] = j means sample i belongs to group j
    # generate features
    x = np.zeros((n, p), dtype=int)
    for i in range(n):
        for j in range(p):
            if model == "bos":
                x[i, j] = bos_model(n_cat[j], mu[w[i]][j], pi[w[i]][j])
            else:
                x[i, j] = god_model_generator(n_cat[j], mu[w[i]][j], pi[w[i]][j])
    return x, w


def plot_hist_bos_model(n_cat, mu, pi, n_sample=10000, show=True):
    x = [bos_model(n_cat, mu, pi) for _ in range(n_sample)]
    sns.histplot(x, stat="density", discrete=True, label="bos model")
    if show:
        plt.show()


def plot_hist_god_model(n_cat, mu, pi, n_sample=10000, show=True):
    x = [god_model_generator(n_cat, mu, pi) for _ in range(n_sample)]
    sns.histplot(x, stat="density", discrete=True, label="god model")
    if show:
        plt.show()


if __name__ == "__main__":
    # Test bos_model
    # plot_hist_bos_model(5, 3, 0.5)

    seed = 0
    """plot_hist_bos_model(5, 3, 0.7, show=False)
    plot_hist_god_model(5, 3, 0.5 + 0.7 * 0.5, show=False)
    plt.legend()
    plt.show()"""

    n = 10000
    p = 5
    n_cat = [10, 10, 5, 5, 5]
    k = 4
    alpha = [0.1, 0.2, 0.3, 0.4]
    mu = [[2, 3, 2, 3, 2], [3, 10, 3, 2, 3], [10, 1, 4, 1, 4], [1, 4, 1, 4, 1]]
    pi = [
        [0.8, 0.7, 0.7, 0.6, 0.9],
        [0.7, 0.8, 0.8, 0.7, 0.6],
        [0.6, 0.9, 0.9, 0.8, 0.7],
        [0.9, 0.6, 0.6, 0.9, 0.8],
    ]

    output_file = "../data/synthetic.csv"

    x, w = generate_data(n, p, n_cat, k, alpha, mu, pi, seed, model="god")
    # save data
    df = pd.DataFrame(x)
    df.to_csv(
        os.path.join(os.path.dirname(__file__), output_file), index=False, header=False
    )

"""    df["w"] = w

    # plot data
    sns.pairplot(
        df,
        hue="w",
        kind="hist",
        plot_kws={"alpha": 0.6, "bins": n_cat},
        diag_kws={"bins": n_cat[0]},
    )
    plt.show()"""
