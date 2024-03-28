import numpy as np
from bos_model_polynomials import compute_polynomials


def check_formal_negativity(p: np.polynomial.Polynomial) -> bool:
    """
    Check if the polynomial represented by the coefficients in p is negative (<= 0) for all x in [0, 1].
    Use a **sufficient** condition for negativity of a polynomial.

    Proof:
    For $x \in [0, 1]$
    $\sum a_i x^i \leq 0 \iff \sum_{i=1}^n a_i x^i \leq -a_0$
    $\iff \sum_{i=1}^n p_i x^i + \sum_{i=1}^n n_i x^i \leq -a_0$  (where $p_i$ and $n_i$ are the positive and negative parts of $a_i$)
    $\Leftarrow \sum_{i=1}^n p_i x^i \leq -a_0$ 
    $\Leftarrow \sum_{i=1}^n p_i \leq -a_0$

    Parameters
    ----------
    p : np.ndarray
        Coefficients of the polynomial.
    
    Returns
    -------
    bool
        True if all coefficients are negative (<= 0), False otherwise.
    """
    if np.all(p.coef <= 0):
        return True
    else:
        if p.coef[0] > 0:
            return False
        else:
            # if not p.coef[p.coef > 0].sum() <= -p.coef[0]:
            #     print(p.coef[p.coef > 0].sum(),  -p.coef[0])
            return p.coef[p.coef > 0].sum() <= -p.coef[0] 


def check_empirical_negativity(p: np.polynomial.Polynomial, nb_sample: int = 10_000) -> bool:
    """
    Check if the polynomial represented by the coefficients in p is negative (<= 0) for
    x in [0, 1] inter NN / nb_sample. Use an **empirical** condition for negativity of a polynomial.

    Parameters
    ----------
    p : np.ndarray
        Coefficients of the polynomial.
    nb_sample : int
        Number of random x to sample.
    
    Returns
    -------
    bool
        True if the polynomial is positive for all x, False otherwise.
    """
    y = p(np.linspace(0, 1, nb_sample))
    return np.all(y <= 0)


def compute_max_derivative(p: np.polynomial.Polynomial) -> float:
    """Compute a majorant of the maximum value of the derivative of the polynomial p on [0, 1].

    Use P' <= a_1 + sum_{i=2}^{n} p_i i where p_i are the positive coefficients of P.
    """
    dp = p.deriv()
    maj = dp.coef[0] + np.sum(dp.coef[1:][dp.coef[1:] > 0])
    # check if the majorant is correct
    # assert maj >= np.max(dp(np.linspace(0, 1, 10_000))), \
    #     f"Majorant {maj} is smaller than the maximum value of the derivative {np.max(dp(np.linspace(0, 1, 10_000)))} of the polynomial {str(p)}"
    return maj


def check_fomralcaly_negativity(p: np.polynomial.Polynomial) -> bool:
    """
    Check if the polynomial represented by the coefficients in p is negative (<= 0) for all x in [0, 1].
    Use a mix of formal and numerical conditions for negativity of a polynomial.
    It is a CNS.

    Use: with M >= max p'
    f(x + h) <= f(x) + h M

    Parameters
    ----------
    p : np.ndarray
        Coefficients of the polynomial.
    
    Returns
    -------
    bool
        True if all coefficients are negative (<= 0), False otherwise.
    """
    max_dp = compute_max_derivative(p)
    if max_dp < 0:
        return True
    assert max_dp >= 1e-10, f"Max derivative is too small: {max_dp}"
    x = 0
    y = p(x)
    assert y <= 1e-13, f"Value of the polynomial is too close to 0: {y}"
    while x < 1 and y <= 0: 
        y = p(x)
        assert y <= 1e-13, f"Value of the polynomial is too close to 0: {y}"
        x += -y / max_dp
    return y <= 0


def get_second_log_derivative_numerator(p: np.ndarray) -> np.polynomial.Polynomial:
    """
    Compute the numerator of the second derivative of the log of the polynomials represented by the coefficients in p.

    P(x) = p[0] * x^(n-1) + p[1] * x^(n-2) + ... + p[n-1]

    Returns the coefficients of the numerator of the second derivative of the log of the polynomial.
    (log P)'' = P'' P - (P')^2

    Parameters
    ----------
    p : np.ndarray
        Coefficients of the polynomial.
    
    Returns
    -------
    np.ndarray
        Coefficients of the numerator of the second derivative of the log of the polynomial.
    """
    p = np.polynomial.Polynomial(p[::-1])
    dp = p.deriv()
    ddp = dp.deriv()
    return ddp * p - dp ** 2


def full_check_log_concavity(p: np.ndarray) -> bool:
    """
    Check if the polynomial represented by the coefficients in p is log-concave on [0, 1].
    P(x) = p[0] * x^(n-1) + p[1] * x^(n-2) + ... + p[n-1]

    First compute the numerator of the second derivative of the log of the polynomial.
    Then check if the numerator is negative for all x in [0, 1].

    Parameters
    ----------
    p : np.ndarray
        Coefficients of the polynomial.
    
    Returns
    -------
    bool
        True if the polynomial is log-concave, False otherwise.
    """
    num = get_second_log_derivative_numerator(p)
    return check_formal_negativity(num) or check_fomralcaly_negativity(num)


def check_all_polynomials(m: int, log_file: str) -> bool:
    """
    Check if all polynomials for m of the BOS model are log-concave.

    Parameters
    ----------
    m : int
        Number of polynomials to consider.

    ...

    """
    u = compute_polynomials(m)
    for x in range(m):
        for mu in range(m):
            try:
                ok = full_check_log_concavity(u[x, mu])
            except AssertionError as e:
                print(f"Error for x={x}, mu={mu}: {e}")
                assert False, f"Error for x={x}, mu={mu}: {e}"
            if not ok:
                print(f"Not concave for x={x}, mu={mu}")
                assert False, f"Not concave for x={x}, mu={mu}"

    return True


if __name__ == "__main__":
    from tqdm import tqdm
    for m in tqdm(range(1, 50)):
        check_all_polynomials(m, "log_concavity_check.log")
