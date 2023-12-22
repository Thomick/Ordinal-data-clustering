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
