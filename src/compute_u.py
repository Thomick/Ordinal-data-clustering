from typing import Iterable
import numpy as np
try:
    from .model_tools import count_errors
except ImportError:
    from model_tools import count_errors


def enumerate_order_info(m: int) -> Iterable[tuple[int, np.ndarray]]:
    """
    Enumerate all possible order information for m categories
    Complexity: O(m * 2^m)

    Args:
        m: number of categories

    Return:
        order_info: all possible order information (m - 1, 2 ** (m - 1))
    """
    order_info = np.zeros(m - 1, dtype=int)
    for i in range(2 ** (m - 1)):
        yield i, order_info
        for j in range(m -1):
            order_info[j] = (order_info[j] + 1) % 2
            if order_info[j] == 1:
                break


def get_all_errors(m: int) -> np.ndarray:
    """
    Generate all possible errors for m categories
    Complexity: O(m * 2^m)

    Args:
        m: number of categories

    Return:
        errors: all possible errors (m, 2 ** (m - 1))
    """
    all_errors = np.zeros((2 ** (m - 1), m), dtype=int)
    for i, order_info in enumerate_order_info(m):
        all_errors[i] = count_errors(order_info)
    return all_errors


def compute_u(m: int) -> np.ndarray:
    """
    Compute u(mu, x, d) of GOD model for 
    all mu in [[1, m]], x in [[1, m]] and d in [[0, m - 1]]

    Complexity: O(m^2 2^m) in all generality
    but O(m 2^m) as the if condition is only true on average less than 2 times

    Arguments:
    ----------
        m: number of categories

    Return:
    -------
        u: u(., ., .) coefficients of the polynomials (m, m, m)
        u[mu, x] = u(mu, x, .)
    """
    distance = get_all_errors(m)
    is_minimal: np.ndarray = np.min(distance, axis=1)[:, None] == distance
    card_min = np.sum(is_minimal, axis=1)
    u = np.zeros((m, m, m))
    for i in range(2 ** (m - 1)):
        for x in range(1, m + 1):
            if is_minimal[i, x - 1]:
                for mu in range(1, m + 1):
                    d = distance[i, mu - 1]
                    u[mu - 1, x - 1, d] += 1 / card_min[i]
    return u


if __name__ == "__main__":
    from pickle import dump, load
    from time import time
    from datetime import datetime

    m_min = 1
    m_max = 30
    for m in range(m_min, m_max + 1):
        try:
            with open(f"GOD_coefficients.pkl", "rb") as f:
                coefficients = load(f)
        except FileNotFoundError:
            coefficients = dict()
        if m in coefficients:
            print(f"u_{m}.pkl already computed")
        else:
            del coefficients
            print(f"Computing u_{m} started at {datetime.now()}")
            t = time()
            u = compute_u(m)
            print(f"Time for m={m}: {time() - t:.3e}")
            try:
                with open(f"GOD_coefficients.pkl", "rb") as f:
                    coefficients = load(f)
            except FileNotFoundError:
                coefficients = dict()
            coefficients[m] = u
            with open(f"GOD_coefficients.pkl", "wb") as f:
                dump(coefficients, f)
