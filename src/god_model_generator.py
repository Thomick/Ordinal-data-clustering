"""
 Copyright (c) 2024 Théo Rudkiewicz, Thomas Michel, Ali Ramlaoui

This program is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License along with this program. If not, see <http://www.gnu.org/licenses/>. 
"""
from typing import Optional
import numpy as np
try:    
    from .model_tools import count_errors
except ImportError:
    from model_tools import count_errors


def god_model_generator(m: int, mu: int, pi: float) -> int:
    """
    Generate x in [[1, m]] from GOD model with parameters mu and pi

    Args:
        m: int
            number of categories
        mu: int in [[1, m]]
            true category
        pi: int
            probability of true comparison (pi >= 0.5)

    Return:
        x: int in [[1, m]]
            generated category
    """
    false_comparisons = np.random.binomial(1, 1 - pi, size=m - 1)
    order = np.arange(1, m) < mu
    # observed_order[i] = (i < mu) if false_comparisons[i] == 0 else ((i < mu) + 1) % 2
    observed_order = (false_comparisons + order) % 2

    nb_errors = count_errors(observed_order)
    x = np.random.choice(np.where(nb_errors == nb_errors.min())[0]) + 1
    return x


def god_model_sample(
    m: int,
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
    assert 1 <= mu <= m, f"mu must be in [[1, m]] but got {mu}"
    assert 0.5 < pi <= 1, f"pi must be in ]0.5, 1] but got {pi}"
    assert n_sample > 0, f"n_sample must be > 0 but got {n_sample}"
    if seed is not None:
        np.random.seed(seed)
    x = np.empty(n_sample, dtype=int)
    for i in range(n_sample):
        x[i] = god_model_generator(m, mu, pi)
    return x
