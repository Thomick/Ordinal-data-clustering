from typing import Optional
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from src.data_generator import generate_data


type_trajectory = list[tuple[int, int, list[int]]]


# Recursively compute the probabilities
def compute_p_list(x: int,
                   mu: int,
                   pi: float,
                   m: int,
                   cur_values: Optional[type_trajectory] = None,
                   cur_prob=1.0,
                   it=0
                   ) -> list[tuple[type_trajectory, float]]:
    """
    Compute the probabilities of the (C_is, x) over all possible trajectories.
    Starting with the entire set of categories (probability 1), check every
    possible trajectory and update the probabilities accordingly.
    :param x: a single feature
    :param mu: position parameter
    :param pi: precision parameter
    :param m: number of categories
    :param cur_values: current trajectories (optional)
    :param cur_prob: current probability (optional)
    :param it: current iteration (optional)
    :return: a list of probabilities :
        Every element is a trajectory and its probability (assuming x is attained)
          (list of objects [(y, z, e), p])
    """

    if cur_values is None:
        cur_e: list[int] = list(range(1, m + 1))
        cur_values: type_trajectory = []
    else:
        cur_e = cur_values[-1][2]
    if len(cur_e) == 0:
        return [(cur_values, 0.0)]
    if len(cur_e) == 1:
        # We have reached the end of the trajectory because only one element remains
        # If the element is x, then the probability is 1 (normalized with the trajectory probability)
        # Otherwise, the probability is 0
        # print("    " * it, cur_values, cur_prob)
        return [(cur_values, (cur_e[0] == x) * cur_prob)]

    p_list = []
    for y in cur_e:
        y: int
        e_minus = [e for e in cur_e if e < y]
        e_plus = [e for e in cur_e if e > y]
        e_equal = [e for e in cur_e if e == y]  # e_equal = [y]
        for z in (0, 1):
            if z == 0:
                p_list += compute_p_list(
                    x,
                    mu,
                    pi,
                    m,
                    cur_values + [(y, z, e_minus)],
                    cur_prob * len(e_minus) / (len(cur_e) ** 2) * (1 - pi),
                    # probability to pick y then to pick z and finally to pick ejp1
                    it=it + 1,
                )
                p_list += compute_p_list(
                    x,
                    mu,
                    pi,
                    m,
                    cur_values + [(y, z, e_plus)],
                    cur_prob * len(e_plus) / (len(cur_e) ** 2) * (1 - pi),
                    it=it + 1,
                )
                p_list += compute_p_list(
                    x,
                    mu,
                    pi,
                    m,
                    cur_values + [(y, z, e_equal)],
                    cur_prob * len(e_equal) / (len(cur_e) ** 2) * (1 - pi),
                    it=it + 1,
                )
            else:
                min_e = e_equal  # e_equal = [y]
                min_dist = abs(mu - e_equal[0])  # min_dist = abs(mu - y)
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
                    cur_values + [(y, z, min_e)],
                    cur_prob * pi / len(cur_e),
                    it=it + 1,
                )
    return p_list


# Use EM algorithm to find the parameters of BOS model of a single feature
def univariate_EM(data, m, mu, n_iter=100, eps=1e-3, pi=None, weights=None):
    """
    Use EM algorithm to find the parameters of BOS model
    :param data: a single feature
    :param n_iter: number of iterations
    :return: mu and pi
    """
    # Initialization
    mu = mu
    if pi is None:
        pi = 0.5
    pi_list = [pi]
    lls_list = []
    old_ll = -np.inf
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
                if m <= 2:
                    stop = m - 1
                else:
                    stop = m - 2
                if m == 3:
                    l = len(c) - 1
                for j in range(stop):  # changing this to -1 fixed for n_cat=3
                    if j < l:
                        if c[j][1] == 1:
                            si += p
                    else:
                        si += p
            if weights is not None:
                s += (si / (p_tots[i] + 1e-10)) * weights[i]
            else:
                s += si / (p_tots[i] + 1e-10)
        if weights is not None:
            s = s / np.mean(weights)
        pi = s / (m - 1) / len(data)
        pi_list.append(pi)
        lls_list.append(
            compute_loglikelihood(data, m, mu, pi, p_lists=p_lists, weights=weights)
        )
        if abs(lls_list[-1] - old_ll) < eps:  # threshold on log-likelihood
            break
        old_ll = lls_list[-1]

    return mu, pi_list, lls_list


def compute_loglikelihood(data, m, mu, pi, p_lists=None, weights=None):
    """
    Compute the loglikelihood of the data
    :param data: a single feature
    :param m: number of categories
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
        # for c, p in p_list:
        # print(c, p)
        # pass
        p_tot = sum([p for c, p in p_list])
        # loglikelihood += (
        #     np.sum([p * np.log(p) if p > 0 else 0 for c, p in p_list]) / p_tot
        # )
        weight = 1 if weights is None else weights[i]
        loglikelihood += np.log(p_tot + 1e-10) * weight
    return loglikelihood


class OrdinalClustering:
    def __init__(self, n_clusters, n_iter=100, eps=1e-1, silent=True):
        self.n_clusters = n_clusters
        self.n_iter = n_iter
        self.eps = eps
        self.silent = silent

    def fit(self, data, m):
        d = data.shape[1]
        m = np.array(m).astype(int)

        mu = np.random.randint(np.ones(m.shape[0]), m + 1, (self.n_clusters, d))
        pi = np.random.random((self.n_clusters, d))
        alpha = np.ones((self.n_clusters)) / self.n_clusters

        def expectation(data, mu, pi, m, alpha):
            p_list = []
            p_list_sum = []
            for i, x in enumerate(data):
                p_list.append([])
                p_list_sum.append([])
                for k in range(self.n_clusters):
                    p_list[-1].append([])
                    p_list_sum[-1].append([])
                    for j in range(d):
                        dimension_p_list = compute_p_list(
                            x[j],
                            mu[k, j],
                            pi[k, j],
                            m[j],
                        )
                        p_list[-1][-1].append(dimension_p_list)
                        p_list_sum[-1][-1].append(sum([p for c, p in dimension_p_list]))

            p_list_x = np.prod(np.array(p_list_sum), axis=2)
            pw1_x = (alpha * p_list_x) / np.sum(alpha * p_list_x, axis=1).reshape(-1, 1)
            return pw1_x, p_list_x

        log_likelihood_old = -np.inf
        ll_list = []
        for _ in range(self.n_iter):
            if not self.silent:
                print("Iteration {}".format(_))
                print("-" * 20)
                print(f"Current mu: {mu}")
                print(f"Current pi: {pi}")
            # E step
            pw1_x, p_list_x = expectation(data, mu, pi, m, alpha)

            # log_likelihood = np.sum(pw1_x * np.log(alpha * p_list_x))
            log_likelihood = np.sum(np.log(np.sum(alpha * p_list_x, axis=1)))
            if not self.silent:
                print("Log-likelihood: {}".format(log_likelihood))
                print()
            if np.abs(log_likelihood - log_likelihood_old) < self.eps:
                if not self.silent:
                    print("Converged, stopping...")
                break

            # M step
            # Update alpha
            alpha = np.mean(pw1_x, axis=0)

            # Internal EM (with threshold): Update pi and mu
            old_pi = pi.copy()
            for k in range(self.n_clusters):
                weights = pw1_x[:, k]
                for j in range(d):
                    lls_local = []
                    pis_local = []
                    mus = np.arange(1, m[j] + 1)
                    for mu_test in mus:
                        mu_r, pis, lls_run = univariate_EM(
                            data[:, j],
                            m[j],
                            mu_test,
                            10,
                            pi=pi[k, j],
                            weights=weights,
                            eps=1e-1,
                        )
                        pis_local.append(pis[-1])
                        lls_local.append(lls_run[-1])
                    # pi is updated according to the previous mu
                    mu_, pi_update, lls_run = univariate_EM(
                        data[:, j],
                        m[j],
                        mu[k, j],
                        10,
                        pi=pi[k, j],
                        weights=weights,
                        eps=1e-1,
                    )
                    pi[k, j] = pi_update[-1]
                    # pi is updated according to the best mu
                    # pi[k, j] = pis_local[np.argmax(lls_local)]
                    mu[k, j] = mus[np.argmax(lls_local)]
            log_likelihood_old = log_likelihood
            ll_list.append(log_likelihood)

            if np.linalg.norm(pi - old_pi) < self.eps:
                if not self.silent:
                    print("Converged, stopping...")
                break

        self.alpha = alpha
        self.mu = mu
        self.pi = pi
        self.ll_list = ll_list
        self.bic = log_likelihood + 0.5 * np.log(data.shape[0]) * (
            self.n_clusters + d * self.n_clusters
        )

        self.labels_ = np.argmax(pw1_x, axis=1)

        return alpha, mu, pi, ll_list

    def fit_transform(self, data, m):
        self.fit(data, m)
        return self.labels_


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--p", type=int, default=2)
    parser.add_argument("--n_cat", type=int, default=3)
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--type", type=str, default="multivariate")
    parser.add_argument("--n_iter", type=int, default=20)
    parser.add_argument("--eps", type=float, default=1e-1)
    args = parser.parse_args()

    if args.type == "univariate":
        n = args.n
        m = args.n_cat
        true_mu = np.random.randint(1, m + 1)
        true_pi = np.random.random()
        data = generate_data(n, 1, [m], 1, [1], [[true_mu]], [[true_pi]], 0)[
            0
        ].flatten()
        print(data)
        print("True mu: {}, True pi: {}".format(true_mu, true_pi))
        ll_list = []
        pi_list = []
        mu_list = list(range(1, m + 1))
        for mu in tqdm(mu_list):
            mu, pl, lls = univariate_EM(data, m, mu, args.n_iter, args.eps)
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

    elif args.type == "multivariate":
        # Test multivariate_AECM
        n = args.n
        d = args.p
        n_clusters = args.k
        m = np.ones(d).astype(int) * args.n_cat
        true_mu = np.random.randint(1, m + 1, (n_clusters, d))
        true_pi = np.random.random((n_clusters, d))
        true_alpha = np.ones(n_clusters) / n_clusters
        data = generate_data(n, d, m, n_clusters, true_alpha, true_mu, true_pi, 0)

        print(
            "True alpha: {}, True mu: {}, True pi: {}".format(
                true_alpha, true_mu, true_pi
            )
        )

        clustering = OrdinalClustering(n_clusters, n_iter=args.n_iter, eps=args.eps)

        alpha_hat, mu_hat, pi_hat, ll_list = clustering.fit(data[0], m)

        labels = clustering.labels_

        print(
            "Estimated alpha: {}, Estimated mu: {}, Estimated pi: {}".format(
                alpha_hat, mu_hat, pi_hat
            )
        )

        plt.plot(ll_list)
        plt.ylabel("log-likelihood")
        plt.xlabel("iteration")
        plt.title("Log-likelihood of the AECM algorithm")
        plt.show()


if __name__ == "__main__":
    main()
