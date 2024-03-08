from unittest import TestCase, main
import numpy as np
from data_generator import bos_model
from bos_model_polynomials import  compute_polynomials
from clustering import compute_p_list
from bos_model_estimator import univariate_em, estimate_mu_pi_bos as estimate_mu_pi, probability_x_given_mu_pi_using_u


def _probability_x_given_mu_pi(m: int, x: int, mu: int, pi: float) -> float:
    # use compute_p_list
    return sum(p for _, p in compute_p_list(x=x, mu=mu, pi=pi, m=m))


def bos_model_sample(m: int, n: int, seed: int = None) -> np.ndarray:
    if seed is not None:
        np.random.seed(seed)
    
    true_mu = np.random.randint(1, m + 1)
    true_pi = np.random.rand()

    print(f'true_mu: {true_mu}, true_pi: {true_pi}')

    np.random.seed(seed)
    data = np.array([bos_model(m, true_mu, true_pi) for _ in range(n)])
    return data, true_mu, true_pi


class TestBosModel(TestCase):
    #def setUp(self) -> None:
    #    self.us = [None] + [compute_u(m) for m in range(1, 11)]

    #def test_compute_u(self):
    #    for m in range(1, 11):
    #        u = self.us[m]
    #        self.assertEqual(u.shape[0], m)
    #        self.assertEqual(u.shape[1], m)
    #        self.assertTrue((u >= 0).all())
    #        self.assertTrue(np.array_equal(u[:, :, 0], np.eye(m)))
    #        comb_values = np.array([comb(m - 1, j, exact=True) for j in range(m)])
    #        self.assertTrue(np.abs(u.sum(axis=1) - comb_values).max() < 1e-10)
    
    def test_estimate_mu_pi(self):
        for m in range(2, 5):
            for seed in range(3):
                print(f'm: {m}, seed: {seed}')
                data, _, _ = bos_model_sample(m, 1_000, seed)
                mu_hat_em, pi_hat_em, ll_em, _ = univariate_em(m=m, data=data, eps=1e-12)
                mu_hat_tri, pi_hat_tri, ll_tri, _ = estimate_mu_pi(m=m, data=data, epsilon=1e-12)
                self.assertEqual(mu_hat_tri, mu_hat_em)
                self.assertAlmostEqual(pi_hat_tri, pi_hat_em, places=3)
                self.assertAlmostEqual(ll_em, ll_tri, places=3)

    def test_probability_x_given_mu_pi(self):
        for m in range(1, 7):
            u = compute_polynomials(m)
            for pi in [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1]:
                for x in range(1, m + 1):
                    for mu in range(1, m + 1):
                        print(f"m={m}, pi={pi}, x={x}, mu={mu}")
                        prob_old = _probability_x_given_mu_pi(m, x, mu, pi)
                        prob = probability_x_given_mu_pi_using_u(m, x, mu, pi, u)
                        self.assertAlmostEqual(prob, prob_old)


if __name__ == "__main__":
    main()
