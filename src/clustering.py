from typing import Optional
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from src.data_generator import generate_data


compact_type_trajectory = list[tuple[int, int, int, int]]
# y, z, e_min, e_max such that e = [e_min ... e_max[
type_trajectory = list[tuple[int, int, list[int]]]


# Recursively compute the probabilities
def compute_p_list(
    x: int,
    mu: int,
    pi: float,
    m: int,
) -> list[tuple[type_trajectory, float]]:
    """
    Compute the probabilities of the (C_is, x) over all possible trajectories.
    Starting with the entire set of categories (probability 1), check every
    possible trajectory and update the probabilities accordingly.
    :param x: a single feature
    :param mu: position parameter
    :param pi: precision parameter
    :param m: number of categories
    :return: a list of probabilities :
        Every element is a trajectory and its probability (assuming x is attained)
          (list of objects [(y, z, e), p])
        [(c, P(c, x | mu, pi)) for all possible trajectories c]
    """

    def recursive_compute_p_list(
        cur_e_min: int,
        cur_e_max: int,
        cur_values: compact_type_trajectory,
        cur_prob: float = 1.0,
        it: int = 0,
    ) -> list[tuple[type_trajectory, float]]:
        """
        Auxiliary function to compute_p_list

        Parameters
        ----------
        cur_e_min : int
            Minimum value of the current interval
        cur_e_max : int
            Maximum value of the current interval (excluded)
        cur_values : compact_type_trajectory
            Current trajectories
        cur_prob : float
            Current probability
        it : int
            Current iteration

        Returns
        -------
        list[tuple[type_trajectory, float]]
            List of trajectories and their probabilities
        """
        if cur_e_max == cur_e_min:
            new_cur_values = []
            for y, z, e_min, e_max in cur_values:
                new_cur_values.append((y, z, list(range(e_min, e_max))))
            return [(new_cur_values, 0.0)]
        if cur_e_min + 1 == cur_e_max:
            # We have reached the end of the trajectory because only one element remains
            # If the element is x, then the probability is 1 (normalized with the trajectory probability)
            # Otherwise, the probability is 0
            # print("    " * it, cur_values, cur_prob)

            # reconstruct the list of e
            new_cur_values = []
            for y, z, e_min, e_max in cur_values:
                new_cur_values.append((y, z, list(range(e_min, e_max))))
            return [(new_cur_values, (cur_e_min == x) * cur_prob)]

        p_list = []
        for y in range(cur_e_min, cur_e_max):
            y: int  # pivot
            len_cur_e = cur_e_max - cur_e_min

            len_e_minus = y - cur_e_min
            len_e_plus = cur_e_max - (y + 1)

            # z = 0
            p_list.extend(
                recursive_compute_p_list(
                    cur_e_min,
                    y,
                    cur_values + [(y, 0, cur_e_min, y)],
                    cur_prob * len_e_minus / len_cur_e**2 * (1 - pi),
                    # probability to pick y then to pick z and finally to pick ejp1
                    it=it + 1,
                )
            )

            p_list.extend(
                recursive_compute_p_list(
                    y + 1,
                    cur_e_max,
                    cur_values + [(y, 0, y + 1, cur_e_max)],
                    cur_prob * len_e_plus / len_cur_e**2 * (1 - pi),
                    it=it + 1,
                )
            )
            p_list.extend(
                recursive_compute_p_list(
                    y,
                    y + 1,
                    cur_values + [(y, 0, y, y + 1)],
                    cur_prob * 1 / len_cur_e**2 * (1 - pi),
                    it=it + 1,
                )
            )

            # z = 1
            min_e = (y, y + 1)
            min_dist = abs(mu - y)
            if cur_e_min != y:
                d_e_minus = min(abs(mu - cur_e_min), abs(mu - (y - 1)))
                if d_e_minus < min_dist:
                    min_e = (cur_e_min, y)
                    min_dist = d_e_minus
            if y + 1 != cur_e_max:
                d_e_plus = min(abs(mu - (y + 1)), abs(mu - (cur_e_max - 1)))
                if d_e_plus < min_dist:
                    min_e = (y + 1, cur_e_max)
                    min_dist = d_e_plus
            p_list.extend(
                recursive_compute_p_list(
                    min_e[0],
                    min_e[1],
                    cur_values + [(y, 1, min_e[0], min_e[1])],
                    cur_prob * pi / len_cur_e,
                    it=it + 1,
                )
            )
        return p_list

    return recursive_compute_p_list(1, m + 1, [], 1, 0)


def compute_loglikelihood(
    data: list[int],
    p_tots: list[float],
    weights: Optional[list[float]] = None,
) -> float:
    """
    Compute the loglikelihood of the data

    Parameters
    ----------
    data : list[int]
        List of observations from a single feature
    m : int
        Number of categories
    mu : int
        Position parameter
    pi : float
        Precision parameter
    p_tots : list[float]
        List of p_tots, p_tots[i] = p(x_i | parameters)
    weights : list[float], optional
        Weights of the observations, by default None
        In the case of univariate data, weights[i] = 1
        In the case of multivariate data, weights[i] is the weight of the ith observation
        probability that the ith observation is in the cluster k is proportional to weights[i]

    Returns
    -------
    float
        Loglikelihood of the data
    """
    loglikelihood = 0.0
    for i in range(len(data)):
        weight = 1 if weights is None else weights[i]
        loglikelihood += np.log(p_tots[i] + 1e-10) * weight
    return loglikelihood


# Use EM algorithm to find the parameters of BOS model of a single feature
def univariate_em(
    data: list[int],
    m: int,
    mu: int,
    n_iter: int = 100,
    eps: float = 1e-3,
    pi: float = 0.5,
    weights=None,
) -> tuple[list[float], list[float]]:
    """
    Use EM algorithm to find the parameter pi of BOS model
    for a fixed mu

    Parameters
    ----------
    data : list[int]
        List of observations from a single feature
    m : int
        Number of categories
    mu : int
        Position parameter
    n_iter : int, optional
        Number of iterations, by default 100
    eps : float, optional
        Threshold on log-likelihood, by default 1e-3
    pi : float, optional
        Precision parameter, by default None
    weights : list[float], optional
        Weights of the observations, by default None


    Returns
    -------
    list[float]
        List of estimated precision parameters
    list[float]
        List of log-likelihoods
    """
    assert m >= 1, "m must be >= 1"
    if m == 1:
        return [1], [0]
    # Initialization
    pi_list: list[float] = [pi]
    lls_list = []
    old_ll = -np.inf

    for _ in range(n_iter):
        # E step
        p_lists = [compute_p_list(x, mu, pi, m) for x in data]
        # p_lists[i][j] = p(x_i, c_j | mu, pi)
        # M step
        p_tots: list[float] = [sum(p for _, p in p_lists[j]) for j in range(len(data))]
        # p_tots[i] = p(x_i | mu, pi) = sum_j p(x_i, c_j | mu, pi)
        s = 0.0
        # s = sum_i^n sum_j^{m-1} p(z_ij = 1 | x_i, mu, pi_q)
        # we compute s = sum_i^n s'_i where
        # s'_i = sum_j^{m-1} p(z_ij = 1 | x_i, mu, pi_q)
        # warning: s'_i != si
        for i in range(len(data)):
            si = 0.0
            # to compute p(z_ij = 1 | x_i, mu, pi_q), we use Bayes' rule:
            # p(z_ij = 1 | x_i, mu, pi_q) = sum_{c} p(z_ij = 1 | c_j, x_i, mu, pi_q) p(c_j | x_i, mu, pi_q)
            # where p(c_j | x_i, mu, pi_q) = p(x_i, c_j | mu, pi_q) / p(x_i | mu, pi_q)
            # hence we can factorize 1 / p(x_i | mu, pi_q) and get:
            # p(z_ij = 1 | x_i, mu, pi_q)
            # = 1 / p(x_i | mu, pi_q) sum_c p(z_ij = 1 | c_j, x_i, mu, pi_q) p(x_i, c_j | mu, pi_q)
            # we can also sum over j = 1 to m - 1 to get si:
            # with si = [sum_j^{m-1} p(z_ij = 1 | x_i, mu, pi_q)] * p(x_i | mu, pi_q)
            # si is computable because p_lists[i][j] = p(x_i, c_j | mu, pi_q) and
            # p(z_ij = 1 | c, x_i, mu, pi_q) = 1 if z_ij = 1 in c_j, 0 otherwise

            for c, p in p_lists[i]:
                c: type_trajectory
                p: float  # p(x_i, ci | mu, pi)
                len_c = len(c)
                stop = m - 2
                if m == 2:
                    stop = 1
                if m == 3:
                    len_c = len(c) - 1
                # m >= 2 by assertion thus m - 2 >= 0
                for j in range(stop):  # changing this to -1 fixed for n_cat=3
                    if j < len_c:
                        if c[j][1] == 1:  # if z = 1
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
        lls_list.append(compute_loglikelihood(data, p_tots=p_tots, weights=weights))
        if abs(lls_list[-1] - old_ll) < eps:  # threshold on log-likelihood
            break
        old_ll = lls_list[-1]

    return pi_list, lls_list, p_tots


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

        def maximize_mu(
            data, pi, m, weights, em_func=univariate_em, n_iter=10, eps=1e-1
        ):
            pis_local, lls_local = [], []
            mus = np.arange(1, m + 1)
            for mu_test in mus:
                pis, lls_run, _ = em_func(
                    data, m, mu_test, n_iter, pi=pi, weights=weights, eps=eps
                )
                pis_local.append(pis[-1])
                lls_local.append(lls_run[-1])
            return pis_local[np.argmax(lls_local)], mus[np.argmax(lls_local)]

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
                    # Search for the best mu
                    best_pi_for_mu, best_mu = maximize_mu(
                        data[:, j],
                        pi[k, j],
                        m[j],
                        weights,
                        em_func=univariate_em,
                        n_iter=10,
                        eps=1e-1,
                    )

                    # pi is updated according to the previous mu
                    pi_update, _ = univariate_em(
                        data[:, j],
                        m[j],
                        mu[k, j],
                        10,
                        pi=pi[k, j],
                        weights=weights,
                        eps=1e-1,
                    )
                    pi[k, j] = pi_update[-1]

                    # # pi is updated according to the best mu
                    # pi[k, j] = best_pi_for_mu

                    mu[k, j] = best_mu
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
            pl, _, p_tots = univariate_em(data, m, mu, args.n_iter, args.eps)
            pi_list.append(pl[-1])
            ll_list.append(compute_loglikelihood(data, p_tots))

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
