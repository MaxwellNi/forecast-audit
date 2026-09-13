import sys,unittest,itertools
from pathlib import Path
import numpy as np
from scipy.optimize import linprog
HERE=Path(__file__).resolve().parent
PUBLIC=HERE.parent/'scripts/analysis'
sys.path.insert(0,str(PUBLIC))
from reference_certificate_efficiency import *

class EfficiencyTests(unittest.TestCase):
 def test_signed_lp_independent_solver(self):
  rng=np.random.default_rng(82734)
  for c in (1,2,8):
   for _ in range(30):
    counts=rng.multinomial(200,np.full(c,1/c));f,g=rng.random((2,c));mv,mw=rng.random((2,c));b=signed_category_budget(f,g,counts,mv,mw)
    for sign,field,values in [(-1,'bias_upper',b['cell_signed_upper']),(1,'bias_lower',b['cell_signed_lower'])]:
     sol=linprog(sign*values,A_eq=np.ones((1,c)),b_eq=[1.],bounds=list(zip(np.zeros(c),b['category_mass_upper'])),method='highs');self.assertTrue(sol.success);self.assertAlmostEqual(b[field],sign*sol.fun,places=11)
    self.assertLessEqual(b['bias_upper'],b['absolute_bias_upper']+1e-13)
 def test_literal_triples_with_ties(self):
  v=np.array([[1,2,2,8,3,2,6],[5,5,1,0,6,6,7]],float);w=v[:,::-1].copy();f=np.full(v.shape,.3);g=np.full(v.shape,.6)
  h=triple_scores(v,w,f,g);expected=[]
  for row in range(2):
   for start in (0,3):
    cells=[]
    for i,j,k in itertools.permutations(range(start,start+3)):
     a=float(v[row,i]>v[row,j])+.5*(v[row,i]==v[row,j]);b=float(w[row,i]>w[row,k])+.5*(w[row,i]==w[row,k]);cells.append((a-.3)*(b-.6))
    expected.append(np.mean(cells))
  np.testing.assert_allclose(h,expected,atol=1e-15)
 def test_pvalue_inversion(self):
  rng=np.random.default_rng(813)
  for J in (2,30,3000):
   h=rng.uniform(-.15,.1,J)
   for kind in ('range','variance','independent_bernstein'):
    for mean in (-.1,0,.01,.1):
     for alpha in (.0002,.01,.05,.4):
      stat=float(h.mean()) if kind=='independent_bernstein' else mean
      d=one_sided_certificate(stat,h,J,[.5],[.5],-.02,alpha=alpha,kind=kind)
      if abs(d['lower_bound'])>1e-12:self.assertEqual(d['p']<alpha,d['lower_bound']>0)
 def test_persistent_negative_fit_error(self):
  b=signed_category_budget([.65],[.35],[1000000],[.5],[.5]);self.assertLess(b['bias_upper'],0)
  lo=one_sided_certificate(-.0125,np.array([-.02,.02]),1000000,[.65],[.35],b['bias_upper'],kind='range')
  old=one_sided_certificate(-.0125,None,1000000,[.65],[.35],b['absolute_bias_upper'],kind='range')
  self.assertGreater(lo['lower_bound'],0);self.assertLess(old['lower_bound'],0)
 def test_zero_variance_and_invalid_inputs(self):
  d=one_sided_certificate(0,np.zeros(10),10,[.5],[.5],0);self.assertEqual(d['p'],1)
  with self.assertRaises(ValueError):one_sided_certificate(0,[0],1,[.5],[.5],0)
  with self.assertRaises(ValueError):one_sided_certificate(0,[0,0],2,[.5],[.5],np.nan)
  with self.assertRaises(ValueError):one_sided_certificate(.1,[0,0],2,[.5],[.5],0,kind='independent_bernstein')
  with self.assertRaises(ValueError):planned_triples(.01,.02,[.5],[.5])
 def test_planned_cost_sufficient(self):
  J=planned_triples(.02,.002,[.4],[.6]);R=kernel_range([.4],[.6])[1]-kernel_range([.4],[.6])[0]
  threshold=.002+R*(np.sqrt(np.log(1/(.05-.0001)))+np.sqrt(np.log(1/.2)))/np.sqrt(2*J)
  self.assertLess(threshold,.02)

if __name__=='__main__':unittest.main()
