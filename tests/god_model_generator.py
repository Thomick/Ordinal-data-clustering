from typing import Optional
import numpy as np
from god_model_tools import count_errors


def god_model_generator(m: int, mu: int, pi: float) -> int:
    """
    Generate x in [[1, n_cat]] from GOD model with parameters mu and pi

    Args:
        m: number of categories
        mu: true category
        pi: probability of true comparison (pi > 0.5)
    
    Return:
        x: generated category
    """
    false_comparisons = np.random.binomial(1, 1 - pi, size=m)
    order = np.arange(1, m + 1) <= mu
    # observed_order[i] = (i <= mu) if false_comparisons[i] == 0 else ((i <= mu) + 1) % 2
    observed_order = (false_comparisons + order) % 2

    nb_errors = count_errors(observed_order)
    x = np.random.choice(np.where(nb_errors == nb_errors.min())[0]) + 1

    return x


def god_model_sample(m: int,
                     mu: int,
                     pi: float,
                     n_sample: int,
                     seed: Optional[int] = None,
                     ) -> np.ndarray:
    """
    Generate n_sample x in [[1, n_cat]] from GOD model with parameters mu and pi

    Args:
        m: number of categories
        mu: true category
        pi: probability of error
        n_sample: number of samples
        seed: random seed
    
    Return:
        x: generated categories (n_sample)
    """
    if seed is not None:
        np.random.seed(seed)
    x = np.empty(n_sample, dtype=int)
    for i in range(n_sample):
        x[i] = god_model_generator(m, mu, pi)
    return x
