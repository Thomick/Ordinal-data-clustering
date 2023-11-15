import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from data_generator import generate_data


# Recusively compute the probabilities
def compute_p_list(x, mu, pi, m, cur_values=[], cur_prob=1.0, it=0):
    """
    Compute the probabilities
    :param data: a single feature
    :param mu: position parameter
    :param pi: precision parameter
    :return: a list of probabilities
    """

    if len(cur_values) == 0:
        cur_e = list(np.arange(1, m + 1))
    else:
        cur_e = cur_values[-1][2]
    if len(cur_e) == 0:
        return [[cur_values, 0.0]]
    if len(cur_e) == 1:
        # print("    " * it, cur_values, cur_prob)
        return [[cur_values, (cur_e[0] == x).astype(float) * cur_prob]]

    p_list = []
    for y in cur_e:
        e_minus = [e for e in cur_e if e < y]
        e_plus = [e for e in cur_e if e > y]
        e_equal = [e for e in cur_e if e == y]
        for z in [0, 1]:
            if z == 0:
                p_list += compute_p_list(
                    x,
                    mu,
                    pi,
                    m,
                    cur_values + [[y, z, e_minus]],
                    cur_prob * len(e_minus) / (len(cur_e) ** 2) * (1 - pi),
                    it=it + 1,
                )
                p_list += compute_p_list(
                    x,
                    mu,
                    pi,
                    m,
                    cur_values + [[y, z, e_plus]],
                    cur_prob * len(e_plus) / (len(cur_e) ** 2) * (1 - pi),
                    it=it + 1,
                )
                p_list += compute_p_list(
                    x,
                    mu,
                    pi,
                    m,
                    cur_values + [[y, z, e_equal]],
                    cur_prob * len(e_equal) / (len(cur_e) ** 2) * (1 - pi),
                    it=it + 1,
                )
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
                p_list += compute_p_list(
                    x,
                    mu,
                    pi,
                    m,
                    cur_values + [[y, z, min_e]],
                    cur_prob * pi / len(cur_e),
                    it=it + 1,
                )
    return p_list


# Use EM algorithm to find the parameters of BOS model of a single feature
def univariate_EM(data, m, mu, n_iter=100, eps=1e-3):
    """
    Use EM algorithm to find the parameters of BOS model
    :param data: a single feature
    :param n_iter: number of iterations
    :return: mu and pi
    """
    # Initialization
    mu = mu
    pi = 0.5
    pi_list = [pi]
    for _ in range(n_iter):
        # E step
        p_lists = [compute_p_list(data[j], mu, pi, m) for j in range(len(data))]
        # M step
        p_tots = [sum([p for c, p in p_lists[j]]) for j in range(len(data))]
        s = 0.0
        for i in range(len(data)):
            si = 0.0
            for c, p in p_lists[i]:
                l = len(c)
                for j in range(m - 2):
                    if j < l:
                        if c[j][1] == 1:
                            si += p
                    else:
                        si += p
            s += si / p_tots[i]
        pi = s / (m - 1) / len(data)
        pi_list.append(pi)
        if abs(pi - pi_list[-2]) < eps:
            break

    return mu, pi_list


def compute_loglikelihood(data, m, mu, pi, p_lists=None):
    """
    Compute the loglikelihood of the data
    :param data: a single feature
    :param m: number of clusters
    :param mu: position parameter
    :param pi: precision parameter
    :return: loglikelihood
    """
    loglikelihood = 0.0
    for i in range(len(data)):
        if p_lists is not None:
            p_list = p_lists[i]
        else:
            p_list = compute_p_list(data[i], mu, pi, m)
        for c, p in p_list:
            # print(c, p)
            pass
        p_tot = sum([p for c, p in p_list])
        loglikelihood += (
            np.sum([p * np.log(p) if p > 0 else 0 for c, p in p_list]) / p_tot
        )
    return loglikelihood


if __name__ == "__main__":
    n = 100
    m = 5
    true_mu = 3
    true_pi = 0.5
    data = generate_data(n, 1, [m], 1, [1], [[true_mu]], [[true_pi]], 0)[0].flatten()
    print(data)
    print("True mu: {}, True pi: {}".format(true_mu, true_pi))
    ll_list = []
    pi_list = []
    mu_list = list(range(1, m + 1))
    for mu in tqdm(mu_list):
        mu, pl = univariate_EM(data, m, mu, 100)
        pi_list.append(pl[-1])
        ll_list.append(compute_loglikelihood(data, m, mu, pl[-1]))

    print(
        "Estimated mu: {}, Estimated pi: {}".format(
            mu_list[np.argmax(ll_list)], pi_list[np.argmax(ll_list)]
        )
    )

    plt.plot(mu_list, ll_list)
    plt.xlabel("mu")
    plt.ylabel("log-likelihood")
    plt.figure()
    plt.plot(mu_list, pi_list)
    plt.xlabel("mu")
    plt.ylabel("pi")
    plt.show()
