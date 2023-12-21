from typing import Iterable, Optional, Union
import numpy as np
from numba import njit
from god_model_generator import count_errors


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
        

def enumerate_order_info(n_cat: int) -> Iterable[tuple[int, np.ndarray]]:
    """
    Enumerate all possible order information for n_cat categories
    Complexity: O(2^n_cat * n_cat)

    Args:
        n_cat: number of categories
    
    Return:
        order_info: all possible order information (n_cat, 2^n_cat)
    """
    order_info = np.zeros(n_cat, dtype=int)
    for i in range(2 ** n_cat):
        yield i, order_info
        for j in range(n_cat):
            order_info[j] = (order_info[j] + 1) % 2
            if order_info[j] == 1:
                break


def get_all_errors(n_cat: int) -> np.ndarray:
    """
    Generate all possible errors for n_cat categories
    Complexity: O(2^n_cat * n_cat)

    Args:
        n_cat: number of categories
    
    Return:
        errors: all possible errors (n_cat, n_cat)
    """
    all_errors = np.zeros((2 ** n_cat, n_cat), dtype=int)
    for i, order_info in enumerate_order_info(n_cat):
        all_errors[i] = count_errors(order_info)
    
    return all_errors


def probability_distribution_x_given_pi(x: int, pi: float, n_cat: int) -> np.ndarray:
    """
    Compute P(x | mu, pi)
    Complexity: O(2^n_cat * n_cat)

    Args:
        x: observed category
        pi: probability of error
        n_cat: number of categories
    
    Return:
        [ P(x | mu, pi) for mu in [[1, n_cat]] ]
    """
    all_errors = get_all_errors(n_cat)
    all_mins: np.ndarray = np.min(all_errors, axis=1)[:, None] == all_errors
    nb_mins = np.sum(all_mins, axis=1)

    p = np.zeros(n_cat)
    
    for mu in range(1, n_cat + 1):
        for i in range(2 ** n_cat):
            if all_mins[i, x - 1]:
                loc_p = pi ** (n_cat - all_errors[i, mu - 1]) * (1 - pi) ** all_errors[i, mu - 1]
                loc_p /= nb_mins[i]
                p[mu - 1] += loc_p
    return p


def probability_distribution_xs_given_pi(xs: np.ndarray, pi: float, n_cat: int) -> np.ndarray:
    """
    Compute P(x^1, ..., x^n | mu, pi)
    Complexity: O(2^n_cat * n_cat * n) with n = xs.shape[0]

    Args:
        xs: observed categories (n,)
        mu: true category
        pi: probability of error
        n_cat: number of categories
    
    Return:
        [ P(x^1, ..., x^n | mu, pi) for mu in [[1, n_cat]] ]
    """
    p = np.ones(n_cat)
    for x in xs:
        p *= probability_distribution_x_given_pi(x, pi, n_cat)
    
    return p


def estimate_mu_given_pi(xs: np.ndarray, pi: float, n_cat: int) -> np.ndarray:
    """
    Compute P(mu | x^1, ..., x^n, pi)
    Complexity: O(2^n_cat * n_cat * n) with n = xs.shape[0]

    Args:
        xs: observed categories (n,)
        pi: probability of error
        n_cat: number of categories
    
    Return:
        [ P(mu | x^1, ..., x^n, pi) for mu in [[1, n_cat]] ]
    """
    p = probability_distribution_xs_given_pi(xs, pi, n_cat)
    return p.argmax() + 1


"""
We aim to compute u(., mu, x).

First, we compute;
distance in [[0, m]](2^m, m) with distance[i, j] = ||E_j - c_i||_1
is_minimal in [[0, m]](2^m, m) with is_minimal[i, j] = (j in argmin_{k in [[1, m]]} ||E_k - c_i||_1)
card_min in [[0, m - 1]](2^m) with card_min[i] = |argmin_{k in [[1, m]]} ||E_k - c_i||_1| = sum(is_minimal[i, :])

This is done in O(2^m * m) time.

Then, we compute u(mu, x, .) in O(m^2 2^m) time.

For mu, x in [[1, m]]^2:
    Init: u[mu, x] = [0, ..., 0] * (m + 1)
    For i in [[1, 2^m]]:  # c_i in {0, 1}^m
        If is_minimal[i, x]:  # ci in C_x
            d = distance[i, mu]
            u[d] += 1 / card_min[i]


We aim to estimate pi and mu given xs.
            
We then compute:
For mu in [[1, m]]:
    P^N(mu, X) = sum_i=1^n m * u(., mu, x^i)
    P^{N - W}(mu, X) = sum_i=1^n u(., mu, x^i) * (m - np.arange(0, m + 1))

    We can then perform the EM algorithm to estimate pi:
        while pi has not converged:
            t = pi / (1 - pi)
            pi = max(1/2, P^{N - W}(mu, X)(t) / P^N(mu, X)(t))
    
    We can then compute the log-likelihood:
        t = pi / (1 - pi)
        p^N(mu, x^i)(t) = (m * u(., mu, x^i))(t)
        log_likelihood = sum_i=1^n log(p^N(mu, x^i)(t))

    Store the best pi and log_likelihood.
Return the best pi and corresponding to the best log_likelihood.
"""


def compute_u(m: int) -> np.ndarray:
    """
    Compute u(., mu, x) for all mu in [[1, m]] and x in [[1, m]]

    Complexity: O(m^2 2^m)

    Arguments:
    ----------
        m: number of categories
    
    Return:
    -------
        u: u(., ., .) coefficients of the polynomials (m, m, m + 1)
        u[mu, x] = u(., mu, x)
    """
    distance = get_all_errors(m)
    is_minimal = np.min(distance, axis=1)[:, None] == distance
    card_min = np.sum(is_minimal, axis=1)
    u = np.zeros((m, m, m + 1))
    for mu in range(1, m + 1):
        for x in range(1, m + 1):
            for i in range(2 ** m):
                if is_minimal[i, x - 1]:
                    d = distance[i, mu - 1]
                    u[mu - 1, x - 1, d] += 1 / card_min[i]
    return u


def compute_log_likelihood(m: int,
                           xs: list[int],
                           pi: float,
                           u_mu: np.ndarray
                           ) -> float:
    """
    Compute the log-likelihood of the model

    log P(X | mu, pi) = sum_i=1^n log(m * u(., mu, x^i)(pi / (1 - pi)))
    where u(., mu, x^i) is the polynomial of degree m with coefficients u_mu

    Complexity: O(n * m)

    Arguments:
    ----------
        m: number of categories
        xs: observed categories
        pi: probability of error
        u_mu: u(., mu, .) coefficients of the polynomials
    
    Return:
    -------
        log_likelihood: log-likelihood of the model
    """
    log_likelihood = 0
    t = pi / (1 - pi)
    for x in xs:
        log_likelihood += np.log(m * evaluate_polynomial(p=u_mu[x], x=t))
    return log_likelihood


def compute_polynomial(m: int,
                       xs: list[int],
                       u_mu: np.ndarray,
                       ) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the polynomial P^N(mu, xs) and P^{N - W}(mu, xs) for a given mu and xs

    Complexity: O(m * n)

    Arguments:
    ----------
        m: number of categories
        u_mu: u(., mu, .) coefficients of the polynomials
        xs: observed categories
    
    Return:
    -------
        p_n: P^N(mu, xs) polynomial
        p_n_w: P^{N - W}(mu, xs) polynomial
    """
    p_n = np.zeros(m + 1)
    p_n_w = np.zeros(m + 1)

    for x in xs:
        p_n += m * u_mu[x]
        p_n_w += u_mu[x] * (m - np.arange(0, m + 1))
    
    return p_n, p_n_w


def estimate_pi(m: int,
                xs: list[int],
                mu: int,
                u_mu: np.ndarray,
                pi: float = 0.75,
                epsilon: float = 1e-6,
                n_iter_max: int = 100,
                evolution: bool = False
                ) -> Union[tuple[list[float], list[float]], tuple[float, float]]:
    """
    Estimate pi given mu and xs for the GOD model using the EM algorithm

    Algorithm:
        
    P^N(mu, X) = sum_i=1^n m * u(., mu, x^i)
    P^{N - W}(mu, X) = sum_i=1^n u(., mu, x^i) * (m - np.arange(0, m + 1))

    We can then perform the EM algorithm to estimate pi:
        while pi has not converged:
            t = pi / (1 - pi)
            pi = max(1/2, P^{N - W}(mu, X)(t) / P^N(mu, X)(t))
    
    We can then compute the log-likelihood:
        t = pi / (1 - pi)
        p^N(mu, x^i)(t) = (m * u(., mu, x^i))(t)
        log_likelihood = sum_i=1^n log(p^N(mu, x^i)(t))
    
    Return pi and log_likelihood

    Complexity: O(m n + m * n_iter + n * m) if the likelihood is only computed at the end
                O(m n + m * n * n_iter + n * m) if the likelihood is computed at each iteration

    Arguments:
    ----------
        m: number of categories
        xs: observed categories
        mu: supposed category
        u_mu: u(., mu, .) coefficients of the polynomials
        pi: initial value of pi
        epsilon: convergence threshold
        n_iter_max: maximum number of iterations
        evolution: whether to compute the log-likelihood at each iteration

    Return:
    -------
        [pi]: estimated probability of error at each iteration
        [log_likelihood of pi]: log-likelihood of the model at each iteration
    """
    if evolution:
        pi_history = [pi]
        log_likelihood_history = [compute_log_likelihood(m=m, xs=xs, pi=pi, u_mu=u_mu)]
    i = 0
    p_n, p_n_w = compute_polynomial(m, xs, u_mu)
    while True:
        i += 1
        t = pi / (1 - pi)
        new_pi = max(1/2, evaluate_polynomial(p_n_w, t) / evaluate_polynomial(p_n, t))
        if evolution:
            pi_history.append(new_pi)
            log_likelihood_history.append(compute_log_likelihood(m=m, xs=xs, pi=new_pi, u_mu=u_mu))
            assert log_likelihood_history[-1] >= log_likelihood_history[-2], "Log-likelihood should increase"
        if abs(pi - new_pi) < epsilon or i >= n_iter_max:
            break
        pi = new_pi
    if evolution:
        return pi_history, log_likelihood_history
    else:
        return pi, compute_log_likelihood(m=m, xs=xs, pi=pi, u_mu=u_mu)


def estimate_mu_pi(m: int,
                   xs: list[int],
                   epsilon: float = 1e-6,
                   pi: float = 0.75,
                   n_iter_max: int = 100,
                   evolution: bool = False,
                   u: Optional[np.ndarray] = None
                   ) -> Union[
                              tuple[int, float, float, list[list[float]], list[list[float]]],
                              tuple[int, float, float]]:
    """
    Estimate mu and pi given xs for the GOD model using the EM algorithm

    Algorithm:
    if u is not None:
        u = u
    pis = np.zeros(m)
    log_likelihoods = np.zeros(m)
    for mu in [[1, m]]:
        pi, log_likelihood = estimate_pi(mu, ...)
        pis[mu - 1] = pi
        log_likelihoods[mu - 1] = log_likelihood
    mu = argmax_{mu in [[1, m]]} log_likelihoods[mu - 1]
    pi = pis[mu - 1]
    log_likelihood = log_likelihoods[mu - 1]
    Return mu, pi and log_likelihood

    Arguments:
    ----------
        m: number of categories
        xs: observed categories
        epsilon: convergence threshold
        pi: initial value of pi
        n_iter_max: maximum number of iterations
        evolution: whether to compute the log-likelihood at each iteration
        u: u(., mu, .) coefficients of the polynomials
    
    Return:
    -------
        if evolution:
            mu, pi, log_likelihood, [pi_history], [log_likelihood_history]: estimated mu, pi and log-likelihood 
            of the model and estimated probability of error and log-likelihood at each iteration for each mu
        else:
            mu, pi, log_likelihood: estimated mu, pi and log-likelihood of the model
    """
    if u is None:
        u = compute_u(m)
    if evolution:
        pis_history = []
        log_likelihoods_history = []
    best_likelihood = -np.inf
    best_mu = 1
    best_pi = 1
    for mu in range(1, m + 1):
        if evolution:
            pi_history, log_likelihood_history = estimate_pi(m, xs, mu, u[mu - 1], pi, epsilon, n_iter_max, evolution)
            pis_history.append(pi_history)
            log_likelihoods_history.append(log_likelihood_history)
            pi = pi_history[-1]
            log_likelihood = log_likelihood_history[-1]
        else:
            pi, log_likelihood = estimate_pi(m, xs, mu, u[mu - 1], pi, epsilon, n_iter_max, evolution)
        if log_likelihood > best_likelihood:
            best_likelihood = log_likelihood
            best_mu = mu
            best_pi = pi
    if evolution:
        return best_mu, best_pi, best_likelihood, pis_history, log_likelihoods_history
    else:
        return best_mu, best_pi, best_likelihood
