import itertools
import unittest
import numpy as np
import pandas as pd
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/analysis'))
from categorical_reference_audit import audit


class CategoricalInputTests(unittest.TestCase):
    def setUp(self):
        self.fits=pd.DataFrame({'category':['a','b'],'forecast_mean':[.4,.6],'outcome_mean':[.6,.4]})
        self.validation=pd.DataFrame([
            [0,'focal','a',1.,4.],[0,'reference','b',3.,3.],
            [1,'focal','b',4.,2.],[1,'reference','a',1.,5.]],
            columns=['pair','role','category','forecast','outcome'])
        self.evaluation=pd.DataFrame([[g,c,x,y] for g in ('g1','g2')
            for c,x,y in [('a',1.,4.),('b',3.,3.),('b',3.,5.),('a',2.,2.)]],
            columns=['group','category','forecast','outcome'])

    def test_generic_values_match_literal_triples(self):
        actual=audit(self.fits,self.validation,self.evaluation)
        group=self.evaluation[self.evaluation.group=='g1'].reset_index(drop=True)
        means=self.fits.set_index('category')
        vals=[]
        for i,j,k in itertools.permutations(range(4),3):
            a=float(group.forecast[j]<group.forecast[i])+.5*float(group.forecast[j]==group.forecast[i])
            b=float(group.outcome[k]<group.outcome[i])+.5*float(group.outcome[k]==group.outcome[i])
            m=means.loc[group.category[i]]
            vals.append((a-m.forecast_mean)*(b-m.outcome_mean))
        self.assertAlmostEqual(actual['result']['mean'],np.mean(vals),14)
        self.assertEqual(actual['validation_observations'],4)
        self.assertFalse(actual['sampling_assumptions_verified'])

    def test_missing_pair_member_rejected(self):
        with self.assertRaises(ValueError):
            audit(self.fits,self.validation.iloc[:-1],self.evaluation)

    def test_mixed_group_sizes_rejected(self):
        with self.assertRaises(ValueError):
            audit(self.fits,self.validation,self.evaluation.iloc[:-1])

    def test_undeclared_category_rejected(self):
        d=self.evaluation.copy();d.loc[0,'category']='c'
        with self.assertRaises(ValueError):audit(self.fits,self.validation,d)

    def test_duplicate_fit_and_nonfinite_data_rejected(self):
        with self.assertRaises(ValueError):
            audit(pd.concat([self.fits,self.fits]),self.validation,self.evaluation)
        d=self.validation.copy();d.loc[0,'forecast']=np.inf
        with self.assertRaises(ValueError):audit(self.fits,d,self.evaluation)


if __name__=='__main__':unittest.main()
