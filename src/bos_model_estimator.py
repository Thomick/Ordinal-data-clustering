"""
 Copyright (c) 2024 Théo Rudkiewicz, Thomas Michel, Ali Ramlaoui

This program is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License along with this program. If not, see <http://www.gnu.org/licenses/>. 
"""
from typing import Any, Optional
import numpy as np
try:
    from .model_tools import estimate_mu_pi_trichotomy, group_sum, compute_log_likelihood
    from .bos_model_polynomials import compute_polynomials, sum_probability_zi
except ImportError:
    from model_tools import estimate_mu_pi_trichotomy, group_sum, compute_log_likelihood
    from bos_model_polynomials import compute_polynomials, sum_probability_zi


# Use EM algorithm to find the parameters of BOS model of a single feature
def univariate_em_pi(
    data: list[int],
    m: int,
    mu: int,
    n_iter: int = 100,
    eps: float = 1e-3,
    pi: float = 0.5,
    weights=None,
    u: Optional[np.ndarray] = None,
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
    u : np.ndarray, optional
        Coefficients of the polynomials, by default None

    Returns
    -------
    list[float]
        List of estimated precision parameters
    list[float]
        List of log-likelihoods
    list[float]
        List of p(x_i | mu, pi_q)
    """
    # print(f"univariate_em_pi, {data=}, {m=}, {mu=}, {n_iter=}, {eps=}, {pi=}, {weights=}")
    assert m >= 1, "m must be >= 1"
    if m == 1:
        return [1], [0], [1]
    
    # Initialization
    if u is None:
        u = compute_polynomials(m)
    if np.array_equal(data, np.arange(1, m + 1)) and weights is not None:
        pass
    else:
        if weights is None:
            weights = np.ones(len(data))
        weights = group_sum(m, data, weights=weights)
    pi_list: list[float] = [pi]
    log_likelihood = compute_log_likelihood(m=m,
                                            weights=weights,
                                            mu=mu,
                                            pi=pi,
                                            u=u,
                                            probability_x_given_mu_pi=probability_x_given_mu_pi_using_u)
    lls_list: list[float] = [log_likelihood]

    for iteration_index in range(n_iter):
        s = 0
        for x in range(1, m + 1):
            s += sum_probability_zi(m, x, mu, pi) * weights[x - 1]
        pi = s / np.sum(weights) / (m - 1)
        pi_list.append(pi)

        # print(f"Iteration {iteration_index + 1}: pi={pi:.6e}, ll={log_likelihood:.6e}")

        new_log_likelihood = compute_log_likelihood(m=m,
                                                    weights=weights,
                                                    mu=mu,
                                                    pi=pi,
                                                    u=u,
                                                    probability_x_given_mu_pi=probability_x_given_mu_pi_using_u)
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
    u: Optional[np.ndarray] = None,
) -> tuple[int, float, float, np.ndarray]:
    """
    Estimate mu and pi using EM algorithm
    
    Use the same parameters as univariate_em_pi
    Return the best mu and pi according to the log-likelihood computed for each mu with univariate_em_pi
    and [ P(x | mu, pi) for x in [[1, m]] ] (TODO)
    """
    assert m >= 1, "m must be >= 1"
    if m == 1:
        return 1, 1, 0, np.array([1])

    if u is None:
        u = compute_polynomials(m)

    # sum of each group to reduce the complexity of the algorithm
    # print(f"univariate_em, original: {data=}, {weights=}")
    weights = group_sum(m, data, weights)
    data = np.arange(1, m + 1)
    # print(f"univariate_em, grouped: {data=}, {weights=}")

    best_pi = None
    best_mu = None
    best_ll = float("-inf")
    best_probs = None

    for mu in range(1, m + 1):
        pi_list, lls_list, probs = univariate_em_pi(
            m=m, data=data, weights=weights, mu=mu, n_iter=n_iter, eps=eps, pi=pi, u=u)
        if lls_list[-1] > best_ll:
            best_ll = lls_list[-1]
            best_pi = pi_list[-1]
            best_mu = mu
            best_probs = probs
    # we can compute the probabilities using the best mu and pi
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
