from typing import Iterable
import numpy as np
from god_model_tools import count_errors


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
    is_minimal: np.ndarray = np.min(distance, axis=1)[:, None] == distance
    card_min = np.sum(is_minimal, axis=1)
    u = np.zeros((m, m, m + 1))
    for mu in range(1, m + 1):
        for x in range(1, m + 1):
            for i in range(2 ** m):
                if is_minimal[i, x - 1]:
                    d = distance[i, mu - 1]
                    u[mu - 1, x - 1, d] += 1 / card_min[i]
    return u
