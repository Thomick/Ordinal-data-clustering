from typing import Iterable
import numpy as np
try:
    from .god_model_tools import count_errors
except ImportError:
    from god_model_tools import count_errors


def enumerate_order_info(m: int) -> Iterable[tuple[int, np.ndarray]]:
    """
    Enumerate all possible order information for n_cat categories
    Complexity: O(m * 2^m)

    Args:
        m: number of categories

    Return:
        order_info: all possible order information (m, 2^m)
    """
    order_info = np.zeros(m, dtype=int)
    for i in range(2 ** m):
        yield i, order_info
        for j in range(m):
            order_info[j] = (order_info[j] + 1) % 2
            if order_info[j] == 1:
                break


def get_all_errors(m: int) -> np.ndarray:
    """
    Generate all possible errors for n_cat categories
    Complexity: O(m * 2^m)

    Args:
        m: number of categories

    Return:
        errors: all possible errors (m, m)
    """
    all_errors = np.zeros((2 ** m, m), dtype=int)
    for i, order_info in enumerate_order_info(m):
        all_errors[i] = count_errors(order_info)

    return all_errors


def compute_u(m: int) -> np.ndarray:
    """
    Compute u(., mu, x) for all mu in [[1, m]] and x in [[1, m]]

    Complexity: O(m^2 2^m) in all generality
    but O(m 2^m) as the if condition is only true on average less than 2 times

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
    for x in range(1, m + 1):
        for i in range(2 ** m):
            if is_minimal[i, x - 1]:
                for mu in range(1, m + 1):
                    d = distance[i, mu - 1]
                    u[mu - 1, x - 1, d] += 1 / card_min[i]
    return u
