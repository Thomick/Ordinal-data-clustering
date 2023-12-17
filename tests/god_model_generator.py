import numpy as np
from numba import njit
import matplotlib.pyplot as plt


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


def god_model_generator(n_cat: int, mu: int, pi: float) -> int:
    """
    Generate a x in [[1, n_cat]] from GOD model with parameters mu and pi

    Args:
        n_cat: number of categories
        mu: true category
        pi: probability of true comparison (pi > 0.5)
    
    Return:
        x: generated category
    """
    false_comparaisons = np.random.binomial(1, 1 - pi, size=n_cat)
    order = np.arange(1, n_cat + 1) <= mu
    # observed_order[i] = (i <= mu) if false_comparaisons[i] == 0 else ((i <= mu) + 1) % 2
    observed_order = (false_comparaisons + order) % 2

    nb_errors = count_errors(observed_order)
    x = np.random.choice(np.where(nb_errors == nb_errors.min())[0]) + 1

    return x


def god_model_sample(n_cat: int, mu: int, pi: float, n_sample: int) -> np.ndarray:
    """
    Generate n_sample x in [[1, n_cat]] from GOD model with parameters mu and pi

    Args:
        n_cat: number of categories
        mu: true category
        pi: probability of error
        n_sample: number of samples
    
    Return:
        x: generated categories (n_sample,)
    """
    x = np.empty(n_sample, dtype=int)
    for i in range(n_sample):
        x[i] = god_model_generator(n_cat, mu, pi)
    
    return x
