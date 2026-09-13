import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts/analysis"))
import itertools
import unittest
import numpy as np
from category_reference_certificate import (
    category_bias_budget, weighted_category_bias_budget, kernel_range,
    finite_reference_decision, distinct_scores)
from scipy.optimize import linprog
from scipy.stats import binom


class CategoryCertificateTests(unittest.TestCase):
    def test_ordered_triples_match_fast_score_with_ties(self):
        v=np.array([[0.,1.,1.,0.]])
        w=np.array([[1.,0.,1.,1.]])
        f=np.array([[.2,.4,.2,.7]])
        g=np.array([[.7,.4,.5,.3]])
        direct=[]
        for i,j,k in itertools.permutations(range(4),3):
            a=float(v[0,j]<v[0,i])+.5*float(v[0,j]==v[0,i])
            b=float(w[0,k]<w[0,i])+.5*float(w[0,k]==w[0,i])
            direct.append((a-f[0,i])*(b-g[0,i]))
        self.assertAlmostEqual(float(distinct_scores(v,w,f,g)[0]),np.mean(direct),14)

    def test_empty_category_retained(self):
        b=category_bias_budget([.35,.5],[.65,.5],[1000,0],[.35,np.nan],[.65,np.nan])
        self.assertEqual(b['lower_v'][1],0)
        self.assertEqual(b['upper_w'][1],1)
        self.assertGreaterEqual(b['bias_upper'],.25)

    def test_bias_bound_covers_all_rectangle_corners(self):
        f,g=np.array([.1,.6]),np.array([.8,.3])
        b=category_bias_budget(f,g,[600,900],[.4,.5],[.5,.6])
        for p in np.linspace(0,1,11):
            for rv,rw in itertools.product(itertools.product(*zip(b['lower_v'],b['upper_v'])),
                                           itertools.product(*zip(b['lower_w'],b['upper_w']))):
                bias=np.dot([p,1-p],(np.array(rv)-f)*(np.array(rw)-g))
                self.assertLessEqual(abs(bias),b['bias_upper']+1e-15)

    def test_range_covers_kernel(self):
        f,g=[.2,.8],[.6,.3]
        lo,hi=kernel_range(f,g)
        for c,a,b in itertools.product(range(2),np.linspace(0,1,11),np.linspace(0,1,11)):
            h=(a-f[c])*(b-g[c])
            self.assertLessEqual(lo,h+1e-15)
            self.assertLessEqual(h,hi+1e-15)

    def test_p_value_and_lower_bound_equivalence(self):
        for mean in (0,.01,.05,.1):
            d=finite_reference_decision(np.full(400,mean),64,[.35,.65],[.35,.65],.002)
            self.assertEqual(d['reject'],d['lower_bound']>0)
            self.assertIsNone(d['bias_to_se'])
            self.assertGreaterEqual(d['p'],.0001)

    def test_validation_rejects_bad_inputs(self):
        with self.assertRaises(ValueError):
            category_bias_budget([.5],[.5],[-1],[.5],[.5])
        with self.assertRaises(ValueError):
            category_bias_budget([1.1],[.5],[2],[.5],[.5])
        with self.assertRaises(ValueError):
            finite_reference_decision([0],2,[.5],[.5],0)
        with self.assertRaises(ValueError):
            finite_reference_decision([.8],3,[.5],[.5],0)

    def test_weighted_budget_matches_independent_linear_program(self):
        b=weighted_category_bias_budget([.35,.5,.7],[.65,.5,.3],
                                        [40,300,2000],[.36,.52,.67],[.66,.54,.31])
        q=b['error_v']*b['error_w']
        fit=linprog(-q,A_eq=np.ones((1,3)),b_eq=[1],
                    bounds=list(zip(np.zeros(3),b['category_mass_upper'])),method='highs')
        self.assertTrue(fit.success)
        self.assertAlmostEqual(b['bias_upper'],-fit.fun,12)
        self.assertLessEqual(b['bias_upper'],b['worst_cell_upper']+1e-15)

    def test_unseen_rare_category_does_not_take_all_mass(self):
        b=weighted_category_bias_budget([.35,.5],[.65,.5],
                                        [8192,0],[.35,np.nan],[.65,np.nan])
        self.assertLess(b['category_mass_upper'][1],.002)
        self.assertLess(b['bias_upper'],.01)

    def test_exact_binomial_upper_coverage(self):
        n,delta=20,.1
        upper=[]
        for k in range(n+1):
            b=weighted_category_bias_budget([.5,.5],[.5,.5],
                                            [k,n-k],[.5,.5],[.5,.5],delta)
            upper.append(b['category_mass_upper'][0])
        for p in np.linspace(.001,.999,101):
            failure=binom.pmf(np.arange(n+1),n,p)[np.asarray(upper)<p].sum()
            self.assertLessEqual(failure,delta/4+1e-14)

    def test_all_empty_validation_remains_safe(self):
        b=weighted_category_bias_budget([.5,.5],[.5,.5],[0,0],[np.nan,np.nan],[np.nan,np.nan])
        self.assertEqual(b['bias_upper'],.25)


if __name__=='__main__':
    unittest.main()
