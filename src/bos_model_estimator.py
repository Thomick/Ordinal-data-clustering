from typing import Any, Optional
import numpy as np
try:
    from .god_model_tools import group_sum
except ImportError:
    from god_model_tools import group_sum


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
        if it == m - 1:
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
            if y != cur_e_min:
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

            if y + 1 != cur_e_max:
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
def univariate_em_pi(
    data: list[int],
    m: int,
    mu: int,
    n_iter: int = 100,
    eps: float = 1e-3,
    pi: float = 0.5,
    weights=None,
) -> tuple[list[float], list[float], list[float]]:
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
    list[float]
        List of p(x_i | mu, pi_q)
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

                for j in range(len_c):
                    if c[j][1] == 1:  # if z = 1
                        si += p
            if weights is not None:
                s += (si / (m - 1) / (p_tots[i] + 1e-20)) * weights[i]
            else:
                s += si / (m - 1) / (p_tots[i] + 1e-20)
        if weights is not None:
            s = s / np.mean(weights)
        pi = s / len(data)
        pi_list.append(pi)
        lls_list.append(compute_loglikelihood(data, p_tots=p_tots, weights=weights))
        if abs(lls_list[-1] - old_ll) < eps:  # threshold on log-likelihood
            break
        old_ll = lls_list[-1]

    return pi_list, lls_list, p_tots


def univariate_em(
    m: int,
    data: list[int],
    weights: Optional[np.ndarray] = None,
    n_iter: int = 100,
    eps: float = 1e-3,
    pi: float = 0.5,
) -> tuple[int, float, float, np.ndarray]:
    """
    Estimate mu and pi using EM algorithm
    
    Use the same parameters as univariate_em_pi
    Return the best mu and pi according to the log-likelihood computed for each mu with univariate_em_pi
    and [ P(x | mu, pi) for x in [[1, m]] ] (TODO)
    """
    # sum of each group to reduce the complexity of the algorithm
    weights = group_sum(m, data, weights)
    data = np.arange(1, m + 1)

    best_pi = None
    best_mu = None
    best_ll = float("-inf")
    best_probs = None

    for mu in range(1, m + 1):
        pi_list, lls_list, probs = univariate_em_pi(
            m=m, data=data, weights=weights, mu=mu, n_iter=n_iter, eps=eps, pi=pi)
        if lls_list[-1] > best_ll:
            best_ll = lls_list[-1]
            best_pi = pi_list[-1]
            best_mu = mu
            best_probs = probs
    return best_mu, best_pi, best_ll, np.array(best_probs)


def observation_likelihood(
        m : int,
        mu: int,
        pi: float) -> np.ndarray:
    """
    Compute the observation likelihood for each observation
    
    Parameters
    ----------
    m : int
        Number of categories
    mu : int
        Position parameter
    pi : float
        Precision parameter
    
    Returns
    -------
    np.ndarray
        Observation likelihood
        [ P(x | mu, pi) for x in [[1, m]] ]
    """
    pxc: list[list[Any, float]] = [compute_p_list(x, mu, pi, m) for x in range(1, m + 1)]
    # p_lists[i][j] = p(x_i, c_j | mu, pi)

    return np.array([sum(p for _, p in pxc[j]) for j in range(m)])
