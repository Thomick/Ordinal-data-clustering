"""
 Copyright (c) 2024 Théo Rudkiewicz, Thomas Michel, Ali Ramlaoui

This program is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License along with this program. If not, see <http://www.gnu.org/licenses/>. 
"""
import numpy as np
try:
    from compute_u import compute_u
except ImportError:
    from src.compute_u import compute_u


def check_unimodality(seq: np.ndarray) -> bool:
    """
    Check if the sequence is unimodal ie only negative values followed by some zeros and then only positive values
    (possibly with an empty positive or negative part).

    Parameters
    ----------
    seq : np.ndarray
        The sequence to check.
    
    Returns
    -------
    bool
        True if the sequence is unimodal, False otherwise.
    """
    seq = np.sign(seq)
    i = 0
    while i < len(seq) and seq[i] > 0:
        i += 1
    while i < len(seq) and seq[i] == 0:
        i += 1
    return np.all(seq[i:] < 0)


def check_bi_monocity(seq: np.ndarray) -> bool:
    """
    Check if the sequence is bi-monotonic ie the sequence is
    increasing and then decreasing (pssibly with an empty increasing or decreasing part).

    Parameters
    ----------
    seq : np.ndarray
        Sequence to check.
    
    Returns
    -------
    bool
        True if the sequence is bi-monotonic, False otherwise.
    """
    return check_unimodality(np.diff(seq))


if __name__ == "__main__":
    m_max = 26
    for m in range(m_max, 0, -1):
        u = compute_u(m)

        xs = np.linspace(0.5, 1, 10_000)
        for x in range(m):
            for mu in range(m):
                f = lambda t: np.polynomial.Polynomial(u[x, mu])((1 - t)/t) * t ** (m - 1)

                bi_monotonic = check_bi_monocity(f(xs))
                assert bi_monotonic, f'Failed for {m=} {x=}, mu={mu}'
    print('All tests passed.')
