import numpy as np
from numba import njit


@njit
def _mul_add_polynomials(h: int,
                           a_1: float,
                           a_0: float,
                           p: np.ndarray,
                           s: np.ndarray
                           ) -> np.ndarray:
    """
    Compute s(X) = s(X) + (a_1 * x + a_0) * p(X)
    where:

    Args:
        a_1, a_0: coefficients of the polynomial to add
        p, np.ndarray of shape m:
            polynomial of degree < h: : p(X) = sum_{i=0}^{h - 2} p[i] * X^{m - 1 - i}
        s, np.ndarray of shape m:
            polynomial of degree <= h: s(X) = sum_{i=0}^{h - 1} s[i] * X^{h - 1 - i}
    """
    for i in range(s.shape[0] - 1, s.shape[0] - h - 1, -1):
        s[i] += a_0 * p[i]
        s[i - 1] += a_1 * p[i]
    # assert p.shape[0] - (p != 0).argmax() < h or p.max() == 0, f"p = {p}, h = {h}, {(p != 0).argmax()=}"
    # s += a_0 * p
    # s[:-1] += a_1 * p[1:]


@njit
def compute_polynomials(m: int) -> np.ndarray:
    """
    Compute the polynomials coefficients u
    P(x | mu, pi) = sum_{d=0}^{m-1} u[mu, x, d] pi^(m - 1 - d)

    Parameters
    ----------
    m : int
        Number of categories
    
    Returns
    -------
    np.ndarray of shape (m, m, m)
        Polynomials coefficients
    """
    u = np.zeros((m, m, m, m)) # h, mu, x, d
    # is_computed = np.zeros((m, m, m), dtype=np.bool8) # h, mu, x
    # is_computed[0, :, :] = True
    u[0, :, :, -1] = 1. 

    for h in range(2, m + 1):
        for mu in range(h):
            for x in range(h):
                    # if x > h // 2:
                    #     assert is_computed[h - 1, h - 1 - x, h - 1 - mu], f"u[{h - 1}, {h - 1 - x}, {h - 1 - mu}] is not computed (symmetry) (h={h}, x={x}, mu={mu})"
                    #     u[h - 1, x, mu] = u[h - 1, h - 1 - x, h - 1 - mu]
                    #     is_computed[h - 1, x, mu] = True
                    # else:
                    s = u[h - 1, mu, x]
                    for y in range(x):
                        # assert is_computed[h - y - 1 - 1, max(mu - y - 1, 0), x - y - 1], f"u[{h - y - 1}, {x - y - 1}, {max(mu - y - 1, 0)}] is not computed (e_+) (h={h}, x={x}, mu={mu})"
                        p = u[h - y - 1 - 1, max(mu - y - 1, 0), x - y - 1]
                        prop = (h - y - 1) / h
                        _mul_add_polynomials(h=h - y, a_1=(mu > y) - prop, a_0=prop, p=p, s=s)
                    good_choice = mu == x or (0 == x and mu <= x) or (h - 1 == x and  mu >= x)
                    #                           e_- == e_=                  e_+ == e_=
                    s[-2] += good_choice - 1 / h
                    s[-1] += 1 / h
                    for y in range(x + 1, h):
                        # assert is_computed[y - 1, min(mu, y - 1), x], f"u[{y}, {x}, {min(mu, y - 1)}] is not computed (e_-) (h={h}, x={x}, mu={mu})"
                        p = u[y - 1, min(mu, y - 1), x]
                        prop = y / h
                        _mul_add_polynomials(h=y + 1, a_1=(mu < y) - prop, a_0=prop, p=p, s=s)
                    
                    s /= h
                    # is_computed[h - 1, mu, x] = True
    return u[-1]


# Recursively compute the probabilities
def compute_p_list(
    x: int,
    mu: int,
    pi: float,
    m: int,
) -> list[tuple[int, float]]:
    """
    Compute the probabilities of the (C_is, x) over all possible trajectories.
    Starting with the entire set of categories (probability 1), check every
    possible trajectory and update the probabilities accordingly.
    :param x: a single feature
    :param mu: position parameter
    :param pi: precision parameter
    :param m: number of categories
    :return: a list of probabilities :
        Every element is a trajectory and its probability (assuming x is attained)
          (list of objects [(y, z, e), p])
        [(sum(z_i), P(c, x | mu, pi)) for all possible trajectories c]
    """
    def recursive_compute_p_list(
        cur_e_min: int,
        cur_e_max: int,
        cur_zi_sum: int,
        cur_prob: float = 1.0,
        it: int = 0,
    ) -> list[tuple[int, float]]:
        """
        Auxiliary function to compute_p_list

        Parameters
        ----------
        cur_e_min : int
            Minimum value of the current interval
        cur_e_max : int
            Maximum value of the current interval (excluded)
        cur_zi_sum : int
            Current sum of z_i
        cur_prob : float
            Current probability
        it : int
            Current iteration

        Returns
        -------
        list[tuple[int, float]]
            List of sum of z_i and probability for each trajectory
        """
        if it == m - 1:
            # We have reached the end of the trajectory because only one element remains
            # If the element is x, then the probability is 1 (normalized with the trajectory probability)
            # Otherwise, the probability is 0
            # print("    " * it, cur_values, cur_prob)

            # reconstruct the list of e
            return [(cur_zi_sum, (cur_e_min == x) * cur_prob)]

        p_list = []
        for y in range(cur_e_min, cur_e_max):
            y: int  # pivot
            len_cur_e = cur_e_max - cur_e_min

            len_e_minus = y - cur_e_min
            len_e_plus = cur_e_max - (y + 1)

            # z = 0
            if y != cur_e_min:
                p_list.extend(
                    recursive_compute_p_list(
                        cur_e_min,
                        y,
                        cur_zi_sum,
                        cur_prob * len_e_minus / len_cur_e**2 * (1 - pi),
                        # probability to pick y then to pick z and finally to pick ejp1
                        it=it + 1,
                    )
                )

            if y + 1 != cur_e_max:
                p_list.extend(
                    recursive_compute_p_list(
                        y + 1,
                        cur_e_max,
                        cur_zi_sum,
                        cur_prob * len_e_plus / len_cur_e**2 * (1 - pi),
                        it=it + 1,
                    )
                )

            p_list.extend(
                recursive_compute_p_list(
                    y,
                    y + 1,
                    cur_zi_sum,
                    cur_prob * 1 / len_cur_e**2 * (1 - pi),
                    it=it + 1,
                )
            )

            # z = 1
            min_e = (y, y + 1)
            min_dist = abs(mu - y)
            if cur_e_min != y:
                d_e_minus = min(abs(mu - cur_e_min), abs(mu - (y - 1)))
                if d_e_minus < min_dist:
                    min_e = (cur_e_min, y)
                    min_dist = d_e_minus
            if y + 1 != cur_e_max:
                d_e_plus = min(abs(mu - (y + 1)), abs(mu - (cur_e_max - 1)))
                if d_e_plus < min_dist:
                    min_e = (y + 1, cur_e_max)
                    min_dist = d_e_plus
            p_list.extend(
                recursive_compute_p_list(
                    min_e[0],
                    min_e[1],
                    cur_zi_sum + 1,
                    cur_prob * pi / len_cur_e,
                    it=it + 1,
                )
            )
        return p_list

    return recursive_compute_p_list(1, m + 1, 0, 1, 0)


def sum_probability_zi(m: int, x: int, mu: int, pi: float) -> float:
    """
    Compute sum_{j = 1}^{m - 1} P(z_j = 1 | x, mu, pi)
    """ 
    s = 0
    p_list = compute_p_list(x=x, mu=mu, pi=pi, m=m)
    p_tot = 0  # P(x | mu, pi)
    for sum_zi, p in p_list:
        sum_zi: int  # sum_{j = 1}^{m - 1} z_j
        p: float  # p(x_i, ci | mu, pi)
        p_tot += p
        s += sum_zi * p  # z_j * p(c)
    return s / p_tot


if __name__ == "__main__":
    print("Version Iterative")
    print(compute_polynomials(3))
