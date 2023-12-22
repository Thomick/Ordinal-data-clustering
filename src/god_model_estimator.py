from typing import Optional, Union
import numpy as np
import matplotlib.pyplot as plt

try:
    from .god_model_tools import evaluate_polynomial, trichotomy_maximization
    from .compute_u import get_all_errors, compute_u
    from .god_model_generator import god_model_sample
except ImportError:
    from god_model_tools import evaluate_polynomial, trichotomy_maximization
    from god_model_generator import god_model_sample
    from compute_u import get_all_errors, compute_u


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
    weights: np.ndarray = None,
) -> float:
    """
    Compute the log-likelihood of the model

    log P(X | mu, pi) = sum_i=1^n log(m * u(., mu, x^i)((1 - pi) / pi))
    where u(., mu, x^i) is the polynomial of degree m with coefficients u_mu

    Complexity: O(n * m)

    The p can be computed with evaluate_polynomial(u_mu[x - 1], (1 - pi) / pi)
    but this cause issues with pi = 0 or pi = 1.

    Arguments:
    ----------
        m: number of categories
        data: observed categories
        pi: probability of error
        u_mu: u(., mu, .) coefficients of the polynomials
        weights: weights of the observations, optional
        only used for AECM

    Return:
    -------
        log_likelihood: log-likelihood of the model
    """

    # version 1
    log_likelihood = 0
    for i, x in enumerate(data):
        p = 0
        for d in range(m + 1):
            p += u_mu[x - 1, d] * pi ** (m - d) * (1 - pi) ** d
        assert p >= 0, f"p should be > 0: {x=}, {u_mu[x - 1]=}, {pi=}, {p=}"
        if weights is None:
            log_likelihood += np.log(p)
        else:
            if weights[i] == 0 and p == 0:
                log_likelihood += 0
            else:
                log_likelihood += weights[i] * np.log(p)
    assert (
        log_likelihood <= 0
    ), f"Log-likelihood should be negative, but {log_likelihood} > 0"
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
        t = (
            1 - pi
        ) / pi  # should be the correct value but it does not work to estimate mu
        # t = pi / (1 - pi) # should be false but it works to estimate mu
        new_pi = max(0.51, evaluate_polynomial(p_n_w, t) / evaluate_polynomial(p_n, t))
        if evolution:
            pi_history.append(new_pi)
            log_likelihood_history.append(
                compute_log_likelihood(m=m, data=data, pi=new_pi, u_mu=u_mu)
            )
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


def estimate_mu_pi_em(
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


def grid_log_likelihood(
    m: int,
    data: list[int],
    u: np.ndarray,
    pi_min: float = 0.5,
    pi_max: float = 0.99,
    nb_pi: int = 100,
    weights: Optional[np.ndarray] = None,
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
    weights : np.ndarray
        Weights of the observations

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
            log_likelihood[mu - 1, i] = compute_log_likelihood(
                m, data, pi, u[mu - 1], weights
            )
    return log_likelihood, pi_range


def estimate_mu_pi_grid(
    m: int,
    data: list[int],
    pi_min: float = 0.5,
    pi_max: float = 1,
    nb_pi: int = 100,
    u: Optional[np.ndarray] = None,
    weights: Optional[np.ndarray] = None,
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
    log_likelihood, pi_range = grid_log_likelihood(
        m, data, u, pi_min, pi_max, nb_pi, weights
    )
    best_pis = pi_range[np.argmax(log_likelihood, axis=1)]
    best_ll = np.max(log_likelihood, axis=1)
    mu = np.argmax(best_ll) + 1
    pi = best_pis[mu - 1]
    log_likelihood = best_ll[mu - 1]
    return mu, pi, log_likelihood


def estimate_mu_pi(
    m: int,
    data: list[int],
    epsilon: float = 1e-4,
    u: Optional[np.ndarray] = None,
    weights: Optional[np.ndarray] = None,
    pi_min: float = 0.5,
    pi_max: float = 1,
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
    epsilon : float
        Precision of the estimation
    u : np.ndarray
        u coefficients of the polynomials
    weights: np.ndarray
        weights of each observation

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

    best_mu = -1
    best_pi = -1
    best_likelihood = -np.inf
    for mu in range(1, m + 1):
        log_likelihood_function = lambda t: compute_log_likelihood(
            m=m, data=data, pi=t, u_mu=u[mu - 1], weights=weights
        )
        pi, log_likelihood = trichotomy_maximization(
            log_likelihood_function, pi_min, pi_max, epsilon
        )
        if log_likelihood > best_likelihood:
            best_likelihood = log_likelihood
            best_mu = mu
            best_pi = pi

    return best_mu, best_pi, best_likelihood


def plot_log_likelihoods(
    m: int,
    mu: int,
    pi: float,
    u: np.ndarray,
    nb_pi: int = 1_000,
    data: np.ndarray = None,
    nb_sample: int = None,
    seed: int = 0,
) -> None:
    """ "
    Plot the log-likelihoods P(X | pi, mu) over pi in [0, 1] for each mu in [1, m]

    Parameters
    ----------
    m : int
        Number of categories
    mu : int
        True category
    pi : float
        Probability of error
    u : np.ndarray
        u coefficients computed with compute_u
    nb_pi : int
        Number of points to plot
    data : np.ndarray
        Data observed if None the data are generated from the god model
    nb_sample : int
        Number of samples to generate from the god model if data is None
    seed : int
        Seed for the random number generator if data is None
    """
    assert 1 <= mu <= m, f"mu={mu} not in [1, m]"
    assert 0 <= pi <= 1, f"pi={pi} not in [0, 1]"
    assert u.shape == (m, m, m + 1), f"u.shape={u.shape} != (m, m, m+1)"
    assert (data is None) ^ (
        nb_sample is None
    ), f"data={data} and nb_sample={nb_sample} are not consistent"
    if data is None:
        data = god_model_sample(m=m, mu=mu, pi=pi, n_sample=nb_sample, seed=seed)
    log_likelihoods, pi_range = grid_log_likelihood(
        m=m, data=data, nb_pi=nb_pi, u=u, pi_min=0.5, pi_max=1
    )
    plt.figure(figsize=(10, 5))
    for i in range(1, m + 1):
        plt.plot(
            pi_range,
            log_likelihoods[i - 1],
            label=f"mu={i}",
            linewidth=2 if i == mu else 1,
        )

    plt.xlabel("pi")
    plt.ylabel("log likelihood")
    plt.axvline(pi, color="r", linestyle="--", label="pi")
    plt.title(f"True parameters mu={mu}, pi={pi}")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    from god_model_generator import god_model_sample

    xs: list[int] = god_model_sample(m=5, mu=2, pi=0.7, n_sample=200)

    print("xs:", xs)

    mu_hat_g, pi_hat_g, ll_g = estimate_mu_pi_grid(m=5, data=xs, nb_pi=100)
    mu_hat_t, pi_hat_t, ll_t = estimate_mu_pi(m=5, data=xs)
    print(f"mu = 2, pi = 0.7")
    print(f"Grid: mu_hat = {mu_hat_g}, pi_hat = {pi_hat_g}, {ll_g=}")
    print(f"Trichotomy: mu_hat = {mu_hat_t}, pi_hat = {pi_hat_t}, {ll_t=}")

    plot_log_likelihoods(m=5, mu=2, pi=0.7, u=compute_u(5), nb_pi=100, data=xs)
