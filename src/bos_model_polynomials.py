from functools import cache
import numpy as np
from numba import njit


@njit
def mul_add_polynomials_rec(a_1: float,
                            a_0: float,
                            p: np.ndarray,
                            s: np.ndarray
                            ) -> np.ndarray:
    """
    Compute s(X) = s(X) + (a_1 * x + a_0) * p(X)
    where:

    Args:
        a_1, a_0: coefficients of the polynomial to add
        p, np.ndarray of shape < h:
            polynomial : p(X) = sum_{i=0}^{deg(p) - 1} p[i] * X^{deg(p) - 1 - i}
        s, np.ndarray of shape h:
            polynomial : s(X) = sum_{i=0}^{h - 1} s[i] * X^{h - 1 - i}
    """
    # assert p.shape[0] < s.shape[0], f"p.shape[0] = {p.shape[0]} >= s.shape[0] = {s.shape[0]}"
    k = p.shape[0]
    s[-k:] += a_0 * p
    s[-k - 1: -1] += a_1 * p


def compute_polynomials_recursive(m: int) -> np.ndarray:
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
    @cache
    def aux_compute_bos_polynomials(h: int,
                                    x: int,
                                    mu: int
                                    ) -> np.ndarray:
        """
        Compute [ u_i, i in [[0, h[[ ] where
        P(x | x in [[0, h[[, mu, pi) = sum_{i=0}^{h-1} u_i * pi^{h - 1 - i}

        Args:
            h, int:  
                number of categories
            x, int in [[0, h[[: 
                observed value
            mu, int in [[0, h[[:
                true value
        """
        if h == 1:
            return np.array([1])
        elif x > h // 2:
            return aux_compute_bos_polynomials(h=h, x=h - 1 - x, mu=h - 1 - mu)
        else:
            s = np.zeros(h)
            for y in range(x):
                p = aux_compute_bos_polynomials(h=h - y - 1, x=x - y - 1, mu=max(mu - y - 1, 0))
                prop = (h - y - 1) / h
                mul_add_polynomials_rec(a_1=(mu > y) - prop, a_0=prop, p=p, s=s)
            good_choice = mu == x or (0 == x and mu <= x) or (h - 1 == x and  mu >= x)
            #                           e_- == e_=                  e_+ == e_=
            s[-2] += good_choice - 1 / h
            s[-1] += 1 / h
            for y in range(x + 1, h):
                p = aux_compute_bos_polynomials(h=y, x=x, mu=min(mu, y - 1))
                prop = y / h
                mul_add_polynomials_rec(a_1=(mu < y) - prop, a_0=prop, p=p, s=s)
            return s / h

    u = np.zeros((m, m, m))
    for mu in range(m):
        for x in range(m):
            u[mu, x, :] = aux_compute_bos_polynomials(h=m, x=x, mu=mu)
    return u


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


if __name__ == "__main__":
    print("Version Recursive")
    print(compute_polynomials_recursive(3))
    print("Version Iterative")
    print(compute_polynomials(3))
