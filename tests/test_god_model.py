"""
 Copyright (c) 2024 Théo Rudkiewicz, Thomas Michel, Ali Ramlaoui

This program is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License along with this program. If not, see <http://www.gnu.org/licenses/>. 
"""
from unittest import TestCase, main
import numpy as np
from scipy.special import comb
from compute_u import compute_u as compute_u
from god_model_estimator import probability_x_given_mu_pi, probability_distribution_x_given_pi, estimate_mu_pi_god as estimate_mu_pi


class TestGodModel(TestCase):
    def setUp(self) -> None:
        self.us = [None] + [compute_u(m) for m in range(1, 11)]
    
    def test_compute_u(self):
        for m in range(1, 11):
            u = self.us[m]
            self.assertEqual(u.shape[0], m)
            self.assertEqual(u.shape[1], m)
            self.assertTrue((u >= 0).all())
            self.assertTrue(np.array_equal(u[:, :, 0], np.eye(m)))
            comb_values = np.array([comb(m - 1, j, exact=True) for j in range(m)])
            self.assertTrue(np.abs(u.sum(axis=1) - comb_values).max() < 1e-10)
    
    def test_probability_xi_given_mu_pi(self):
        for m in range(1, 11):
            u = self.us[m]
            for pi in [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1]:
                for x in range(1, m + 1):
                    probs_old = probability_distribution_x_given_pi(m, x, pi)
                    for mu in range(1, m + 1):
                        print(f"m={m}, pi={pi}, x={x}, mu={mu}")
                        prob = probability_x_given_mu_pi(m, x, mu, pi, u)
                        self.assertAlmostEqual(prob, probs_old[mu - 1])
    
    def test_estimate_mu_pi_1(self):
        xs = np.array([3,4,3,3,3,2,3,2,3,3,4,9,3,3,4,3,1,3,2,3,2,5,1,3,3,1,1,3,2,7,3,5,4,3,5,3,1,10,3,3,2,2,2,1,2,3,3,3,2,3,3,3,3,3,3,10,1,3,1,4,3,3,5,1,2,3,2,3,3,3,3,4,3,9,2,1,3,1,1,3,3,3,3,3,7,3,4,1,5,3,4,3,2,2,1,2,5,3,3,2])
        mu, pi, ll, _ = estimate_mu_pi(10, xs, u=self.us[10], epsilon=1e-4)
        self.assertEqual(mu, 3)
        self.assertAlmostEqual(pi, 0.7972585618117456)
        self.assertAlmostEqual(ll, -161.13938629955825)

    def test_estimate_mu_pi_2(self):
        # m = 6, mu = 1, pi = 0.9
        xs = np.array([1,1,1,1,1,1,3,1,1,1,1,1,1,1,1,1,1,1,1,1])
        mu, pi, ll, _ = estimate_mu_pi(6, xs, u=self.us[6], epsilon=1e-4)
        self.assertEqual(1, mu)
        self.assertAlmostEqual(0.9679518926215107, pi)
        self.assertAlmostEqual(-5.095497835660907, ll)
    
    def test_estimate_mu_pi_3(self):
        # m = 9, mu = 6, pi = 0.95
        xs = np.array([6,6,6,6,6,8,6,7,4,6,6,6,6,6,6,7,6,5,6,6])
        mu, pi, ll, _ = estimate_mu_pi(9, xs, u=self.us[9], epsilon=1e-4)
        self.assertEqual(6, mu)
        self.assertAlmostEqual(0.9201239523702662, pi)
        self.assertAlmostEqual(-18.670558582229166, ll)
    
    def test_estimate_mu_pi_4(self):
        # m = 6, mu = 1, pi = 0.9
        xs = np.array([1,1,1,1,1,1,3,1,1,1,1,1,1,1,1,1,1,1,1,1])
        mu, pi, ll, _ = estimate_mu_pi(6, xs, u=self.us[6], epsilon=1e-4)
        self.assertEqual(1, mu)
        self.assertAlmostEqual(0.9679518926215107, pi)
        self.assertAlmostEqual(-5.095497835660907, ll)
    
    def test_estimate_mu_pi_5(self):
        # m = 9, mu = 6, pi = 0.95
        xs = np.array([6,6,6,6,6,8,6,7,4,6,6,6,6,6,6,7,6,5,6,6])
        mu, pi, ll, _ = estimate_mu_pi(9, xs, u=self.us[9], epsilon=1e-4)
        self.assertEqual(6, mu)
        self.assertAlmostEqual(0.9201239523702662, pi)
        self.assertAlmostEqual(-18.670558582229166, ll)


if __name__ == "__main__":
    main()
