from unittest import TestCase, main
import numpy as np
from compute_u import compute_u
from god_model_estimator import estimate_mu_pi, probability_xi_given_mu_pi, probability_distribution_x_given_pi


class TestGodModel(TestCase):
    def setUp(self) -> None:
        self.us = [None] + [compute_u(m) for m in range(1, 11)]
    
    def test_compute_u(self):
        for m in range(1, 11):
            u = self.us[m]
            self.assertEqual(u.shape[0], m)
            self.assertEqual(u.shape[1], m)
            self.assertTrue((u >= 0).all())
    
    def test_probability_xi_given_mu_pi(self):
        for m in range(1, 11):
            u = self.us[m]
            for pi in [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1]:
                for x in range(1, m + 1):
                    probs_old = probability_distribution_x_given_pi(m, x, pi)
                    for mu in range(1, m + 1):
                        print(f"m={m}, pi={pi}, x={x}, mu={mu}")
                        prob = probability_xi_given_mu_pi(m, x, mu, pi, u)
                        self.assertAlmostEqual(prob, probs_old[mu - 1])
    
    def test_estimate_mu_pi_1(self):
        xs = np.array([3,4,3,3,3,2,3,2,3,3,4,9,3,3,4,3,1,3,2,3,2,5,1,3,3,1,1,3,2,7,3,5,4,3,5,3,1,10,3,3,2,2,2,1,2,3,3,3,2,3,3,3,3,3,3,10,1,3,1,4,3,3,5,1,2,3,2,3,3,3,3,4,3,9,2,1,3,1,1,3,3,3,3,3,7,3,4,1,5,3,4,3,2,2,1,2,5,3,3,2])
        mu, pi, ll = estimate_mu_pi(10, xs, u=self.us[10])
        self.assertEqual(mu, 3)
        self.assertAlmostEqual(pi, 0.7972585618117456)
        self.assertAlmostEqual(ll, -161.13938629955825)

    def test_estimate_mu_pi_2(self):
        # m = 6, mu = 1, pi = 0.9
        xs = np.array([1,1,1,1,1,1,3,1,1,1,1,1,1,1,1,1,1,1,1,1])
        mu, pi, ll = estimate_mu_pi(6, xs, u=self.us[6])
        self.assertEqual(1, mu)
        self.assertAlmostEqual(0.9679518926215107, pi)
        self.assertAlmostEqual(-5.095497835660907, ll)
    
    def test_estimate_mu_pi_3(self):
        # m = 9, mu = 6, pi = 0.95
        xs = np.array([6,6,6,6,6,8,6,7,4,6,6,6,6,6,6,7,6,5,6,6])
        mu, pi, ll = estimate_mu_pi(9, xs, u=self.us[9])
        self.assertEqual(6, mu)
        self.assertAlmostEqual(0.9201239523702662, pi)
        self.assertAlmostEqual(-18.670558582229166, ll)
    
    def test_estimate_mu_pi_4(self):
        # m = 6, mu = 1, pi = 0.9
        xs = np.array([1,1,1,1,1,1,3,1,1,1,1,1,1,1,1,1,1,1,1,1])
        mu, pi, ll = estimate_mu_pi(6, xs, u=self.us[6])
        self.assertEqual(1, mu)
        self.assertAlmostEqual(0.9679518926215107, pi)
        self.assertAlmostEqual(-5.095497835660907, ll)
    
    def test_estimate_mu_pi_5(self):
        # m = 9, mu = 6, pi = 0.95
        xs = np.array([6,6,6,6,6,8,6,7,4,6,6,6,6,6,6,7,6,5,6,6])
        mu, pi, ll = estimate_mu_pi(9, xs, u=self.us[9])
        self.assertEqual(6, mu)
        self.assertAlmostEqual(0.9201239523702662, pi)
        self.assertAlmostEqual(-18.670558582229166, ll)


if __name__ == "__main__":
    main()
