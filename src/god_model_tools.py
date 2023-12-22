from typing import Callable
import numpy as np
from numba import njit


@njit
def count_errors(order_info: np.ndarray) -> np.ndarray:
    """
    Count errors in order_info for each mu in [[1, m]]
    Complexity: O(m)

    Args:
        order_info: array of order information c in {0, 1}^m

    Return:
        [|| E_k - c||_1 for k in [[1, m]]] np.ndarray in [[0, m]]^m
        with E_k = [1] * k + [0] * (m - k)
    """
    m = order_info.shape[0]
    
    errors = np.zeros(m)
    current_error = np.sum(order_info)

    for i in range(m):
        current_error = current_error + 1 - 2 * order_info[i]
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
