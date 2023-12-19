from typing import Iterable
import numpy as np
from god_model_generator import count_errors



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
        mu: true category
        pi: probability of error
        n_cat: number of categories
    
    Return:
        [ P(x | mu, pi) for mu in [[1, n_cat]] ]
    """
    all_errors = get_all_errors(n_cat)
    all_mins = np.min(all_errors, axis=1)[:, None] == all_errors
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
