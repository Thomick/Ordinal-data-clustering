from typing import Optional, Union
import numpy as np
import torch
from .god_model_tools import evaluate_polynomial
from .compute_u import get_all_errors, compute_u


def probability_distribution_x_given_pi(m: int, x: int, pi: float) -> np.ndarray:
    """
    Compute P(x | mu, pi)
    Complexity: O(2^n_cat * n_cat)

    Args:
        x: observed category
        pi: probability of error
        m: number of categories

    Return:
        [ P(x | mu, pi) for mu in [[1, n_cat]] ]
    """
    distance = get_all_errors(m)
    is_minimal: np.ndarray = np.min(distance, axis=1)[:, None] == distance
    card_min = np.sum(is_minimal, axis=1)

    p = np.zeros(m)

    for mu in range(1, m + 1):
        for i in range(2**m):
            if is_minimal[i, x - 1]:
                loc_p = (
                    pi ** (m - distance[i, mu - 1]) * (1 - pi) ** distance[i, mu - 1]
                )
                loc_p /= card_min[i]
                p[mu - 1] += loc_p
    return p


def probability_distribution_xs_given_pi(
    m: int, data: np.ndarray, pi: float
) -> np.ndarray:
    """
    Compute P(x^1, ..., x^n | mu, pi)
    Complexity: O(2^n_cat * n_cat * n) with n = len(xs)

    Args:
        m: number of categories
        data: observed categories (n) [x^1, ..., x^n]
        pi: probability of error

    Return:
        [ P(x^1, ..., x^n | mu, pi) for mu in [[1, n_cat]] ]
    """
    p = np.ones(m)
    for x in data:
        p *= probability_distribution_x_given_pi(m, x, pi)
    return p


def likelihood_distribution_xs_given_pi(
    m: int, data: np.ndarray, pi: float
) -> np.ndarray:
    """
    Compute [log P(x^1, ..., x^n | mu, pi) for mu in [[1, m]]]
    Complexity: O(2^m * m * n) with n = len(xs)

    Args:
        m: number of categories
        data: observed categories (n) [x^1, ..., x^n]
        pi: probability of error

    Return:
        [log P(x^1, ..., x^n | mu, pi) for mu in [[1, m]] ]
    """
    p = np.zeros(m)
    for x in data:
        p += np.log(probability_distribution_x_given_pi(m, x, pi))
    return p


def estimate_mu_given_pi(
    m: int,
    data: np.ndarray,
    pi: float,
) -> int:
    """
    Compute P(mu | x^1, ..., x^n, pi)
    Complexity: O(2^n_cat * n_cat * n) with n = xs.shape[0]

    Args:
        m: number of categories
        data: observed categories (n)
        pi: probability of error

    Return:
        argmax [ P(mu | x^1, ..., x^n, pi) for mu in [[1, n_cat]] ]
    """
    p = likelihood_distribution_xs_given_pi(m, data, pi)
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
This is done by compute_u(m) in O(m^2 2^m) time.


We aim to estimate pi and mu given xs.
            
We then compute:
For mu in [[1, m]]:
    P^N(mu, X) = sum_i=1^n m * u(., mu, x^i)
    P^{N - W}(mu, X) = sum_i=1^n u(., mu, x^i) * (m - np.arange(0, m + 1))

    We can then perform the EM algorithm to estimate pi:
        while pi has not converged:
            t = (1 - pi) / pi
            pi = max(1/2, P^{N - W}(mu, X)(t) / P^N(mu, X)(t))
    
    We can then compute the log-likelihood:
        t = (1 - pi) / pi
        p^N(mu, x^i)(t) = (m * u(., mu, x^i))(t)
        log_likelihood = sum_i=1^n log(p^N(mu, x^i)(t))

    Store the best pi and log_likelihood.
Return the best pi and corresponding to the best log_likelihood.
"""


def compute_log_likelihood(
    m: int,
    data: list[int],
    pi: float,
    u_mu: np.ndarray,
) -> float:
    """
    Compute the log-likelihood of the model

    log P(X | mu, pi) = sum_i=1^n log(m * u(., mu, x^i)((1 - pi) / pi))
    where u(., mu, x^i) is the polynomial of degree m with coefficients u_mu

    Complexity: O(n * m)

    Arguments:
    ----------
        m: number of categories
        data: observed categories
        pi: probability of error
        u_mu: u(., mu, .) coefficients of the polynomials

    Return:
    -------
        log_likelihood: log-likelihood of the model
    """
    t = (1 - pi) / pi

    # version 1
    log_likelihood = m * len(data) * np.log(pi)
    for x in data:
        # assert 1 <= x <= m, f"Category should be in [[1, m]], but {x} is not"
        log_likelihood += np.log(evaluate_polynomial(p=u_mu[x - 1], x=t))

    assert log_likelihood <= 0, f"Log-likelihood should be negative, but {log_likelihood} > 0"
    return log_likelihood


def compute_polynomial(
    m: int,
    data: list[int],
    u_mu: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the polynomial P^N(mu, xs) and P^{N - W}(mu, xs) for a given mu and xs

    Complexity: O(m * n)

    Arguments:
    ----------
        m: number of categories
        u_mu: u(., mu, .) coefficients of the polynomials
        data: observed categories

    Return:
    -------
        p_n: P^N(mu, xs) polynomial
        p_n_w: P^{N - W}(mu, xs) polynomial
    """
    p_n = np.zeros(m + 1)
    p_n_w = np.zeros(m + 1)

    for x in data:
        assert 1 <= x <= m, f"Category should be in [[1, m]], but {x} is not"
        p_n += m * u_mu[x - 1]
        p_n_w += u_mu[x - 1] * (m - np.arange(0, m + 1))

    return p_n, p_n_w


def estimate_pi(
    m: int,
    data: list[int],
    mu: int,
    u_mu: np.ndarray,
    pi: float = 0.75,
    epsilon: float = 1e-6,
    n_iter_max: int = 100,
    evolution: bool = False,
) -> Union[tuple[list[float], list[float]], tuple[float, float]]:
    """
    Estimate pi given mu and xs for the GOD model using the EM algorithm

    Algorithm:

    P^N(mu, X) = sum_i=1^n m * u(., mu, x^i)
    P^{N - W}(mu, X) = sum_i=1^n u(., mu, x^i) * (m - np.arange(0, m + 1))

    We can then perform the EM algorithm to estimate pi:
        while pi has not converged:
            t = (1 - pi) / pi
            pi = max(1/2, P^{N - W}(mu, X)(t) / P^N(mu, X)(t))

    We can then compute the log-likelihood:
        t = (1 - pi) / pi
        p^N(mu, x^i)(t) = (m * u(., mu, x^i))(t)
        log_likelihood = sum_i=1^n log(p^N(mu, x^i)(t))

    Return pi and log_likelihood

    Complexity: O(m n + m * n_iter + n * m) if the likelihood is only computed at the end
                O(m n + m * n * n_iter + n * m) if the likelihood is computed at each iteration

    Arguments:
    ----------
        m: number of categories
        data: observed categories
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
        log_likelihood_history = [
            compute_log_likelihood(m=m, data=data, pi=pi, u_mu=u_mu)
        ]
    i = 0
    p_n, p_n_w = compute_polynomial(m, data, u_mu)
    while True:
        i += 1
        t = (1 - pi) / pi  # should be the correct value but it does not work to estimate mu
        # t = pi / (1 - pi) # should be false but it works to estimate mu
        new_pi = max(0.51, evaluate_polynomial(p_n_w, t) / evaluate_polynomial(p_n, t))
        if evolution:
            pi_history.append(new_pi)
            log_likelihood_history.append(compute_log_likelihood(m=m, data=data, pi=new_pi, u_mu=u_mu))
            # assert (log_likelihood_history[-1] >= log_likelihood_history[-2]), (
            #            f"Log-likelihood should increase at each iteration"
            #            f", but {log_likelihood_history[-1]} < {log_likelihood_history[-2]}")
        if abs(pi - new_pi) < epsilon or i >= n_iter_max:
            break
        pi = new_pi
    if evolution:
        return pi_history, log_likelihood_history
    else:
        return pi, compute_log_likelihood(m=m, data=data, pi=pi, u_mu=u_mu)


def estimate_mu_pi(
    m: int,
    data: list[int],
    epsilon: float = 1e-6,
    pi_zero: float = 0.51,
    n_iter_max: int = 100,
    evolution: bool = False,
    u: Optional[np.ndarray] = None,
) -> Union[
    tuple[int, float, float, list[list[float]], list[list[float]]],
    tuple[int, float, float],
]:
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
        data: observed categories
        epsilon: convergence threshold
        pi_zero: initial value of pi
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
            pi_history, log_likelihood_history = estimate_pi(
                m, data, mu, u[mu - 1], pi_zero, epsilon, n_iter_max, evolution
            )
            pis_history.append(pi_history)
            log_likelihoods_history.append(log_likelihood_history)
            pi = pi_history[-1]
            log_likelihood = log_likelihood_history[-1]
        else:
            pi, log_likelihood = estimate_pi(
                m, data, mu, u[mu - 1], pi_zero, epsilon, n_iter_max, evolution
            )
        if log_likelihood > best_likelihood:
            best_likelihood = log_likelihood
            best_mu = mu
            best_pi = pi
    if evolution:
        return best_mu, best_pi, best_likelihood, pis_history, log_likelihoods_history
    else:
        return best_mu, best_pi, best_likelihood


def grid_log_likelihood(m: int,
                        data: list[int],
                        u: np.ndarray,
                        pi_min: float = 0.5,
                        pi_max: float = 0.99,
                        nb_pi: int = 100
                        ) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the log-likelihood of the data given the model for different values of pi and all possible values of mu.

    Parameters
    ----------
    m : int
        Number of categories
    data : list[int]
        Observed categories
    u : np.ndarray
        u coefficients of the polynomials
    pi_min : float
        Minimum value of pi
    pi_max : float
        Maximum value of pi
    nb_pi : int
        Number of values of pi to test
    
    Return
    ------
    log_likelihood : np.ndarray
        log-likelihood of the model for different values of pi and all possible values of mu
        log_likelihood[mu - 1, i] is the log-likelihood of the model for mu and pi = pi_min + i * (pi_max - pi_min) / nb_pi
    pi_range : np.ndarray
        pi values tested
    """
    pi_range = np.linspace(pi_min, pi_max, nb_pi)
    log_likelihood = np.zeros((m, nb_pi))
    for mu in range(1, m + 1):
        for i, pi in enumerate(pi_range):
            log_likelihood[mu - 1, i] = compute_log_likelihood(m, data, pi, u[mu - 1])
    return log_likelihood, pi_range


def estimate_mu_pi_grid(m: int,
                        data: list[int],
                        pi_min: float = 0.5,
                        pi_max: float = 1,
                        nb_pi: int = 100,
                        u: Optional[np.ndarray] = None
                        ) -> tuple[int, float, float]:
    """
    Estimate mu and pi given xs for the GOD model using the grid search algorithm

    Parameters
    ----------
    m : int
        Number of categories
    data : list[int]
        Observed categories
    pi_min : float
        Minimum value of pi
    pi_max : float
        Maximum value of pi
    nb_pi : int
        Number of values of pi to test
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
    """
    if u is None:
        u = compute_u(m)
    log_likelihood, pi_range = grid_log_likelihood(m, data, u, pi_min, pi_max, nb_pi)
    best_pis = pi_range[np.argmax(log_likelihood, axis=1)]
    best_ll = np.max(log_likelihood, axis=1)
    mu = np.argmax(best_ll) + 1
    pi = best_pis[mu - 1]
    log_likelihood = best_ll[mu - 1]
    return mu, pi, log_likelihood


def compute_log_likelihood_torch(
        m: int,
        data: list[int],
        pi: torch.Tensor,
        u_mu: np.ndarray,
        ) -> float:
    """
    Compute the log-likelihood of the model

    log P(X | mu, pi) = sum_i=1^n log(m * u(., mu, x^i)((1 - pi) / pi))
    where u(., mu, x^i) is the polynomial of degree m with coefficients u_mu

    Complexity: O(n * m)

    Arguments:
    ----------
        m: number of categories
        data: observed categories
        pi: probability of error
        u_mu: u(., mu, .) coefficients of the polynomials

    Return:
    -------
        log_likelihood: log-likelihood of the model
    """
    # version 1
    # t = (1 - pi) / pi
    # log_likelihood = m * len(data) * torch.tensor(torch.log(pi))
    # for x in data:
    #     p_t = torch.tensor(0.)
    #     for d in range(M + 1):
    #         p_t += u_mu[x - 1, d] * ((1 - pi) / pi) ** d
    #     log_likelihood += torch.log(p_t)
    
    # version 2
    assert 0 <= pi <= 1, f"pi should be in [0, 1], but {pi} is not"
    log_likelihood = torch.tensor(0.)
    for x in data:
        p_t = torch.tensor(0.)
        for d in range(m + 1):
            p_t += u_mu[x - 1, d] * (1 - pi) ** d * pi ** (m - d)
        log_likelihood += torch.log(p_t)

    assert log_likelihood <= 0, f"Log-likelihood should be negative, but {log_likelihood} > 0"
    return log_likelihood


def optimize_log_likelihood(
    m: int,
    data: list[int],
    u_mu: np.ndarray,
    pi_zero: float = 0.5,
    n_iter_max: int = 100,
    epsilon: float = 1e-3,
    ) -> tuple[float, float]:
    """
    Optimize the log-likelihood of the model over pi

    Arguments:
    ----------
        m: number of categories
        data: observed categories
        u_mu: u(mu, ., .) coefficients of the polynomials
        pi_zero: initial value of pi
        n_iter_max: maximum number of iterations
        epsilon: convergence criterion
    
    Return:
    -------
        pi_hat: optimal value of pi
        log_likelihood: log-likelihood of the model
    """
    pi = torch.tensor(pi_zero, requires_grad=True)
    pi_old = -1
    optimizer = torch.optim.SGD([pi], lr=0.001)
    for _ in range(n_iter_max):
        optimizer.zero_grad()
        try:
            n_log_likelihood = -compute_log_likelihood_torch(m, data, pi, u_mu)
        except AssertionError:
            return pi.item(), -np.inf
        n_log_likelihood.backward()
        optimizer.step()
        # pi = pi.clamp(0, 1)
        if abs(pi - pi_old) < epsilon:
            break
        pi_old = pi.item()
    return pi.item(), -n_log_likelihood.item()


def estimate_mu_pi_torch(
        m: int,
        data: list[int],
        pi_zero: float = 0.5,
        n_iter_max: int = 100,
        epsilon: float = 1e-3,
        ) -> tuple[int, float, float]:
    """
    Estimate mu and pi given xs for the GOD model using SGD to optimize the log-likelihood

    Arguments:
    ----------
        m: number of categories
        data: observed categories
        pi_zero: initial value of pi
        n_iter_max: maximum number of iterations
        epsilon: convergence criterion
    
    Return:
    -------
        mu: estimated mu
        pi: estimated pi
        log_likelihood: log-likelihood of the model
    """
    u = compute_u(m)
    log_likelihoods = np.zeros(m)
    pi_hats = np.zeros(m)
    for mu in range(1, m + 1):
        pi_hat, log_likelihood = optimize_log_likelihood(m, data, u[mu - 1], pi_zero, n_iter_max, epsilon)
        pi_hats[mu - 1] = pi_hat
        log_likelihoods[mu - 1] = log_likelihood
    mu = np.argmax(log_likelihoods) + 1
    pi = pi_hats[mu - 1]
    log_likelihood = log_likelihoods[mu - 1]
    return mu, pi, log_likelihood


if __name__ == "__main__":
    from god_model_generator import god_model_sample

    xs: list[int] = god_model_sample(m=5, mu=2, pi=0.7, n_sample=200)

    print("xs:", xs)
    mu_hat_t, pi_hat_t, _ = estimate_mu_pi_torch(m=5, data=xs)

    mu_hat_g, pi_hat_g, _ = estimate_mu_pi_grid(m=5, data=xs, nb_pi=100)
    print(f"mu = 2, pi = 0.7")
    print(f"Torch: mu_hat = {mu_hat_t}, pi_hat = {pi_hat_t}")
    print(f"Grid: mu_hat = {mu_hat_g}, pi_hat = {pi_hat_g}")
