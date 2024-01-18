from functools import cache
from typing import Any, Optional
import numpy as np
try:
    from .model_tools import estimate_mu_pi_trichotomy, trichotomy_maximization, group_sum, compute_log_likelihood
    from .bos_model_polynomials import compute_polynomials
except ImportError:
    from model_tools import estimate_mu_pi_trichotomy, trichotomy_maximization, group_sum, compute_log_likelihood
    from bos_model_polynomials import compute_polynomials


# Recursively compute the probabilities
def compute_p_list(
    x: int,
    mu: int,
    pi: float,
    m: int,
) -> list[tuple[int, float]]:
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
        [(sum(z_i), P(c, x | mu, pi)) for all possible trajectories c]
    """
    @cache
    def recursive_compute_p_list(
        cur_e_min: int,
        cur_e_max: int,
        cur_zi_sum: int,
        cur_prob: float = 1.0,
        it: int = 0,
    ) -> list[tuple[int, float]]:
        """
        Auxiliary function to compute_p_list

        Parameters
        ----------
        cur_e_min : int
            Minimum value of the current interval
        cur_e_max : int
            Maximum value of the current interval (excluded)
        cur_zi_sum : int
            Current sum of z_i
        cur_prob : float
            Current probability
        it : int
            Current iteration

        Returns
        -------
        list[tuple[int, float]]
            List of sum of z_i and probability for each trajectory
        """
        if it == m - 1:
            # We have reached the end of the trajectory because only one element remains
            # If the element is x, then the probability is 1 (normalized with the trajectory probability)
            # Otherwise, the probability is 0
            # print("    " * it, cur_values, cur_prob)

            # reconstruct the list of e
            return [(cur_zi_sum, (cur_e_min == x) * cur_prob)]

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
                        cur_zi_sum,
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
                        cur_zi_sum,
                        cur_prob * len_e_plus / len_cur_e**2 * (1 - pi),
                        it=it + 1,
                    )
                )

            p_list.extend(
                recursive_compute_p_list(
                    y,
                    y + 1,
                    cur_zi_sum,
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
                    cur_zi_sum + 1,
                    cur_prob * pi / len_cur_e,
                    it=it + 1,
                )
            )
        return p_list

    return recursive_compute_p_list(1, m + 1, 0, 1, 0)


def sum_probability_zi(m: int, x: int, mu: int, pi: float) -> float:
    """
    Compute sum_{j = 1}^{m - 1} P(z_j = 1 | x, mu, pi)
    """ 
    s = 0
    p_list = compute_p_list(x, mu, pi, m)
    p_tot = 0  # P(x | mu, pi)
    for sum_zi, p in p_list:
        sum_zi: int  # sum_{j = 1}^{m - 1} z_j
        p: float  # p(x_i, ci | mu, pi)
        p_tot += p
        s += sum_zi * p  # z_j * p(c)
    return s / p_tot


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
    # print(f"univariate_em_pi, {data=}, {m=}, {mu=}, {n_iter=}, {eps=}, {pi=}, {weights=}")
    assert m >= 1, "m must be >= 1"
    if m == 1:
        return [1], [0], [1]
    
    # Initialization
    u = compute_polynomials(m)
    if np.array_equal(data, np.arange(1, m + 1)) and weights is not None:
        pass
    else:
        if weights is None:
            weights = np.ones(len(data))
        weights = group_sum(m, data, weights=weights)
    pi_list: list[float] = [pi]
    log_likelihood = compute_log_likelihood(m=m, weights=weights, mu=mu, pi=pi, u=u, probability_x_given_mu_pi=probability_x_given_mu_pi_using_u)
    lls_list: list[float] = [log_likelihood]

    for iteration_index in range(n_iter):
        s = 0
        for x in range(1, m + 1):
            s += sum_probability_zi(m, x, mu, pi) * weights[x - 1]
        pi = s / np.sum(weights) / (m - 1)
        pi_list.append(pi)

        # print(f"Iteration {iteration_index + 1}: pi={pi:.6e}, ll={log_likelihood:.6e}")

        new_log_likelihood = compute_log_likelihood(m=m, weights=weights, mu=mu, pi=pi, u=u, probability_x_given_mu_pi=probability_x_given_mu_pi_using_u)
        lls_list.append(new_log_likelihood)
        if abs(new_log_likelihood - log_likelihood) < eps:
            break
        log_likelihood = new_log_likelihood

    # print(f"EM algorithm for {mu} converged after {iteration_index + 1} iterations")
    return pi_list, lls_list, [probability_x_given_mu_pi_using_u(m=m, x=x, mu=mu, pi=pi, u=u) for x in data]


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
    # print(f"univariate_em, original: {data=}, {weights=}")
    weights = group_sum(m, data, weights)
    data = np.arange(1, m + 1)
    # print(f"univariate_em, grouped: {data=}, {weights=}")
    

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
    # we can compute the probabilities using the best mu and pi
    return best_mu, best_pi, best_ll, np.array(best_probs)

"""
Useful functions:
"""


def probability_x_given_mu_pi_using_u(
        m: int,
        x: int,
        mu: int,
        pi: float,
        u: np.ndarray) -> float:
    """
    Compute P(x | mu, pi) = sum_d=0^{m-1} u(mu, x, d) * pi^(m - 1 - d)

    Complexity: O(m)

    Arguments:
    ----------
        m: int with m >= 1
            number of categories
        x: int in [[1, m]]
            observed category
        mu: int in [[1, m]]
            supposed category
        pi: float in [1/2, 1]
            probability of error
        u: np.ndarray of int of shape (m, m, m)
            coefficients of the polynomial u(mu, x, d)

    Return:
    -------
        p: probability P(x | mu, pi)
    """
    return np.polyval(u[mu - 1, x - 1], pi)


def estimate_mu_pi_bos(m: int,
    data: np.ndarray,
    weights: Optional[np.ndarray] = None,
    epsilon: float = 1e-5,
    u: Optional[np.ndarray] = None,
) -> tuple[int, float, float, np.ndarray]:
    """
    Estimate mu and pi given xs for the GOD model using the grid search algorithm

    Parameters
    ----------
    m : int
        Number of categories
    data : list[int]
        Observed categories
    weights: np.ndarray
        weights of each observation
    epsilon : float
        Precision of the estimation
    u : np.ndarray
        u coefficients of the polynomials

    Return
    ------
    mu : int
        Estimated mu
    pi : float
        Estimated pi
    log_likelihood : float
        Log-likelihood of the model : log P(X | mu, pi)
    probability : np.ndarray
        Probability of each category : [ P(x | mu, pi) for x in [[1, m]] ]
    """
    return estimate_mu_pi_trichotomy(
        m=m,
        probability_x_given_mu_pi=probability_x_given_mu_pi_using_u,
        data=data,
        weights=weights,
        epsilon=epsilon,
        u=u,
        compute_polynomials=compute_polynomials,
        pi_min=0,
        pi_max=1
    )
