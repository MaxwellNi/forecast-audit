import itertools
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/analysis"))
from peer_rank_products import peer_rank_products


def brute(x, y, mx, my):
    n = len(x)
    result = []
    for i in range(n):
        terms = []
        for j in range(n):
            for k in range(n):
                if i == j or i == k or j == k:
                    continue
                a = float(x[j] < x[i]) + .5 * float(x[j] == x[i])
                b = float(y[k] < y[i]) + .5 * float(y[k] == y[i])
                terms.append((a - mx[i]) * (b - my[i]))
        result.append(np.mean(terms))
    return np.array(result)


class PeerRankProductsTests(unittest.TestCase):
    def test_all_three_row_tie_patterns(self):
        mx, my = np.array([.1, .5, .9]), np.array([.8, .3, .6])
        vectors = list(itertools.product(range(3), repeat=3))
        for x in vectors:
            for y in vectors:
                got = peer_rank_products(x, y, mx, my)
                np.testing.assert_allclose(got['corrected_residual_product'], brute(x,y,mx,my), atol=3e-16)

    def test_random_continuous_and_tied_triple_enumeration(self):
        rng = np.random.default_rng(20260905971)
        for n in [4, 7, 11]:
            for tied in [False, True]:
                x, y = rng.normal(size=(2,n))
                if tied:
                    x, y = np.round(x), np.round(y)
                mx, my = rng.uniform(size=(2,n))
                got = peer_rank_products(x,y,mx,my)
                np.testing.assert_allclose(got['corrected_residual_product'], brute(x,y,mx,my), atol=3e-16)

    def test_permutation_and_monotone_invariance(self):
        rng = np.random.default_rng(20260905972)
        x,y = rng.integers(-3,4,size=(2,33));mx,my=rng.uniform(size=(2,33))
        reference=peer_rank_products(x,y,mx,my)
        transformed=peer_rank_products(x**3,np.exp(y),mx,my)
        order=rng.permutation(33);permuted=peer_rank_products(x[order],y[order],mx[order],my[order])
        for name in reference:
            np.testing.assert_array_equal(reference[name],transformed[name])
            np.testing.assert_array_equal(reference[name][order],permuted[name])

    def test_constant_channel_is_exactly_zero_at_oracle_mean(self):
        got=peer_rank_products(np.ones(9),np.arange(9),np.full(9,.5),np.linspace(0,1,9))
        np.testing.assert_array_equal(got['corrected_residual_product'],np.zeros(9))

    def test_invalid_arrays(self):
        for arrays in [([1,2],[2,1],[.5,.5],[.5,.5]),
                       ([1,2,3],[1,2,3],[.5,.5],[.5]*3),
                       ([1,2,float('nan')],[1,2,3],[.5]*3,[.5]*3),
                       ([1,2,3],[1,2,3],[-.1,.5,.5],[.5]*3)]:
            with self.assertRaises(ValueError): peer_rank_products(*arrays)


if __name__ == '__main__':
    unittest.main()
