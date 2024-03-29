"""
 Copyright (c) 2024 Théo Rudkiewicz, Thomas Michel, Ali Ramlaoui

This program is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License along with this program. If not, see <http://www.gnu.org/licenses/>. 
"""
from typing import Callable, Optional
import numpy as np
from numba import njit


@njit
def count_errors(order_info: np.ndarray) -> np.ndarray:
    """
    Count errors in order_info for each mu in [[1, m]]
    Complexity: O(m)

    Args:
        order_info: array of order information c in {0, 1}^(m - 1)

    Return:
        [|| E_k - c||_1 for k in [[1, m]]] np.ndarray in [[0, m - 1]]^m
        with E_k = [1] * (k - 1) + [0] * (m - k)

    Algorithm:
    || E_1 - c ||_1 = || c ||_1
    || E_k - c ||_1 = || E_{k - 1} - c ||_1 + 1 - 2 * c[k - 1]
    """
    m = order_info.shape[0] + 1

    current_error = np.sum(order_info)
    errors = np.empty(m, dtype=np.int64)
    errors[0] = current_error
    for i in range(1, m):
        current_error = current_error + 1 - 2 * order_info[i - 1]
        errors[i] = current_error
    return errors


@njit
def evaluate_polynomial(p: np.ndarray, x: float) -> float:
    """
    Evaluate a polynomial
    Complexity: O(p.shape[0])

    Args:
        p: polynomial coefficients [a_0, ..., a_n]
        x: value to evaluate
    
    Return:
        p(x) = a_0 + a_1 * x + ... + a_n * x^n
    """
    y = 0
    for i in range(p.shape[0] - 1, -1, -1):
        y = y * x + p[i]
    return y


def trichotomy_maximization(f: Callable[[float], float], 
                            a: float, 
                            b: float, 
                            epsilon: float = 1e-5,
                            max_iter: int = 1_000
                            ) -> tuple[float, float]:
    """
    Find the maximum of a function f on [a, b] using trichotomy
    Complexity: O( (lg(epislon) - lg(b - a)) / (1 - lg(3)) ) 

    Arguments:
    ----------
        f: function to maximize
        a: left bound of the interval
        b: right bound of the interval
        epsilon: convergence criterion
    
    Return:
    -------
        x_max: maximum of f on [a, b]
        f(x_max)
    """
    assert a < b, f"a={a} >= b={b}"
    i = 0
    while b - a > epsilon and i < max_iter:
        h = (b - a) / 3
        c = a + h
        d = b - h
        if f(c) <= f(d):
            a = c
        else:
            b = d
        i += 1
    return (a + b) / 2, f((a + b) / 2)


@njit
def group_sum(m: int, data: np.ndarray, weights: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Compute the sum of weights grouped by unique values in data
    Complexity: O(n + m)

    Args:
        data, array (n) of int in [[1, m]]: data group
        weights, array (n) of floats: weights for each data point of shape (n)

    Return:
        array of sums of shape (m)
    """
    if weights is None:
        weights = np.ones(data.shape[0])
    assert data.shape[0] == weights.shape[0]
    sums = np.zeros(m)
    for i in range(data.shape[0]):
        sums[data[i] - 1] += weights[i]
    return sums


def compute_log_likelihood(
    m: int,
    weights: np.ndarray,
    mu: int,
    pi: float,
    u: np.ndarray,
    probability_x_given_mu_pi: Callable[[int, int, int, float, np.ndarray], float],
) -> float:
    """
    Compute the log-likelihood of the model

    log P(X | mu, pi) = sum_i=1^m w_i log(P(i | mu, pi))

    Complexity: O(m * C(probability_x_given_mu_pi)) but for BOS and GOD models,
    C(probability_x_given_mu_pi) = O(m) so the complexity is O(m^2)

    Arguments:
    ----------
        m, int: 
            number of categories
        weights, np.ndarray of shape m: 
            weights[i] is the weight of the observation i
        mu, int in [[1, m]]: 
            supposed category
        pi, float in [0, 1]: 
            probability of error
        u, np.ndarray[int] of shape (m, m, m): 
            coefficients of the polynomials u(mu, x, d)
        probability_x_given_mu_pi, Callable[[int, int, int, float, np.ndarray], float]:
            probability of x given mu and pi
            takes as arguments: m, x, mu, pi and u
        
    Return:
    -------
        log_likelihood: log-likelihood of the model
    """
    assert m == weights.shape[0], f"m={m} != weights.shape[0]={weights.shape[0]}"

    log_likelihood = 0
    for x, w in enumerate(weights):
        if w != 0:
            p = probability_x_given_mu_pi(m=m, x=x + 1, mu=mu, pi=pi, u=u)
            assert p >= 0, f"p should be > 0: {x=}, {u[mu - 1, x - 1]=}, {pi=}, {p=}"
            log_likelihood += w * np.log(p)
    assert (
        log_likelihood <= 0
    ), f"Log-likelihood should be negative, but {log_likelihood} > 0"
    return log_likelihood


def estimate_mu_pi_trichotomy(
    m: int,
    probability_x_given_mu_pi: Callable[[int, int, int, float, np.ndarray], float],
    data: np.ndarray,
    weights: Optional[np.ndarray] = None,
    epsilon: float = 1e-5,
    u: Optional[np.ndarray] = None,
    compute_polynomials: Optional[Callable[[int], np.ndarray]] = None,
    pi_min: float = 0,
    pi_max: float = 1
) -> tuple[int, float, float, np.ndarray]:
    """
    Estimate mu and pi given xs for the GOD model using the grid search algorithm

    Parameters
    ----------
    m : int
        Number of categories
    probability_x_given_mu_pi : Callable[[int, int, int, float, np.ndarray], float]
        Probability of x given mu and pi for the model takes u as argument
    data : np.ndarray of int in [[1, m]]
        Observed categories
    weights: np.ndarray
        weights of each observation
    epsilon : float
        Precision of the estimation
    u : np.ndarray
        u coefficients of the polynomials
    compute_polynomials : Callable[[int], np.ndarray]
        Function to compute the polynomials u(mu, x, d)

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
    if u is None:
        assert compute_polynomials is not None, "u or compute_polynomials should be given"
        u = compute_polynomials(m)
    
    # sum of each group to reduce the complexity of the algorithm
    weights = group_sum(m, data, weights)

    best_mu = -1
    best_pi = -1
    best_likelihood = -np.inf
    for mu in range(1, m + 1):
        log_likelihood_function = lambda t: compute_log_likelihood(
            m=m, weights=weights, mu=mu, pi=t, u=u, probability_x_given_mu_pi=probability_x_given_mu_pi
        )
        pi, log_likelihood = trichotomy_maximization(
            log_likelihood_function, pi_min, pi_max, epsilon
        )
        if log_likelihood > best_likelihood:
            best_likelihood = log_likelihood
            best_mu = mu
            best_pi = pi
    
    probability = np.array(
        [probability_x_given_mu_pi(m=m, x=x, mu=best_mu, pi=best_pi, u=u) for x in range(1, m + 1)]
    )

    return best_mu, best_pi, best_likelihood, probability
