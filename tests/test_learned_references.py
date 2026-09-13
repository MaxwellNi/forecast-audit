"""Independent finite-state checks for learned-mean reference correction."""
import itertools
from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/analysis'))
from learned_reference_study import binary_scores, fit_means

class LearnedReferenceTests(unittest.TestCase):
    def test_scores_match_literal_distinct_reference_average(self):
        rng=np.random.default_rng(47)
        for n in (3,4,9):
            v=rng.integers(0,2,size=(20,n));w=rng.integers(0,2,size=(20,n))
            f=rng.uniform(size=(20,n));g=rng.uniform(size=(20,n))
            shared,distinct=binary_scores(v,w,f,g)
            expected=[]
            for group in range(20):
                products=[]
                for i in range(n):
                    peers=[j for j in range(n) if j!=i]
                    products.append(np.mean([(float(v[group,j]<v[group,i])+.5*(v[group,j]==v[group,i])-f[group,i])*
                        (float(w[group,k]<w[group,i])+.5*(w[group,k]==w[group,i])-g[group,i])
                        for j in peers for k in peers if j!=k]))
                expected.append(np.mean(products))
            np.testing.assert_allclose(distinct,expected,atol=3e-16,rtol=0)

    def test_training_uses_empirical_leave_one_out_midranks(self):
        z=np.array([0,0,0,1,1,1]);v=np.array([0,1,0,1,1,0])
        ranks=np.array([np.mean([float(v[j]<v[i])+.5*(v[j]==v[i]) for j in range(len(v)) if j!=i]) for i in range(len(v))])
        np.testing.assert_allclose(fit_means(z,v),[ranks[z==c].mean() for c in (0,1)],atol=1e-15)

    def test_exact_null_expectations_with_imperfect_means(self):
        atoms=list(itertools.product((0,1),repeat=3))
        probabilities=np.array([.5*(.2+.6*z if v else .8-.6*z)*(.2+.6*z if w else .8-.6*z) for z,v,w in atoms])
        configurations=np.array(list(itertools.product(range(8),repeat=3)))
        values=np.array(atoms)[configurations]
        z,v,w=values[:,:,0],values[:,:,1],values[:,:,2]
        f=np.array([.25,.8]);g=np.array([.45,.55]);true=np.array([.35,.65])
        shared,distinct=binary_scores(v,w,f[z],g[z])
        weights=probabilities[configurations].prod(axis=1)
        bias=np.mean((true-f)*(true-g))
        self.assertAlmostEqual(float(weights@shared),bias+.0225/2,places=14)
        self.assertAlmostEqual(float(weights@distinct),bias,places=14)

    def test_empty_training_cell_has_bounded_fallback(self):
        fitted=fit_means(np.zeros(4,int),np.array([0,0,1,1]))
        self.assertEqual(fitted[1],.5)
        self.assertAlmostEqual(fitted[0],.5,places=14)

if __name__=='__main__':unittest.main()
