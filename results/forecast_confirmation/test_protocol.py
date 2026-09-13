"""Regression checks for exact nulls, target definitions and information time."""
from pathlib import Path
import io
import tempfile
import unittest
import zipfile
import numpy as np
import pandas as pd
from scipy.stats import rankdata
import experiment
from gates import classical,by_values,exact_forecast_residual


class ProtocolChecks(unittest.TestCase):
    def test_exact_category_copy_residual(self):
        rng=np.random.default_rng(667)
        for n in [13,336,4362,8713]:
            c=rng.integers(0,32,n)
            x=2477.1*(c+.5)/32
            a=exact_forecast_residual(x,c)
            np.testing.assert_array_equal(a,np.zeros(n))
            for y in [rng.integers(0,5,n),x.copy()]:
                pair=.25*(a[:2*(n//2):2]-a[1:2*(n//2):2])*np.sign(y[:2*(n//2):2]-y[1:2*(n//2):2])
                result=classical(0.,pair,0.,full=True)
                self.assertEqual(result['p'],1.)

    def test_exact_centering_against_rational(self):
        from fractions import Fraction
        rng=np.random.default_rng(967)
        for n in [7,19,77]:
            c=rng.integers(0,5,n);x=rng.integers(-2,4,n)
            q=(2*rankdata(x)-1).astype(int)
            a=exact_forecast_residual(x,c)
            for i in range(n):
                count=int((c==c[i]).sum());total=sum(int(v) for v in q[c==c[i]])
                exact=Fraction(int(q[i])*count-total,2*n*count)
                self.assertEqual(a[i],float(exact))

    def test_current_future_outcome_invariance(self):
        index=pd.date_range('2000-01-01',periods=600,freq='h')
        series=pd.Series(np.arange(600,dtype=float),index=index)
        series.iloc[40:44]=np.nan
        before=experiment.features(series,24)
        for j in [20,190,590]:
            changed=series.copy();changed.iloc[j:]+=1e8
            after=experiment.features(changed,24)
            np.testing.assert_equal(before.iloc[:j+1].to_numpy(),after.iloc[:j+1].to_numpy())
        self.assertTrue(np.isnan(before.loc[index[44],'lag_1']))

    def test_order_two_formula_with_ties(self):
        rng=np.random.default_rng(405)
        for n in [2,3,7,18]:
            a=rng.normal(size=n);y=rng.integers(0,4,n)
            matrix=.25*(a[:,None]-a[None,:])*np.sign(y[:,None]-y[None,:])
            direct=matrix.sum()/(n*(n-1))
            fast=np.sum(a*(rankdata(y)-(n+1)/2))/(n*(n-1))
            self.assertAlmostEqual(direct,fast,places=14)

    def test_complete_hour_aggregation_and_lag(self):
        stamps=pd.date_range('2000-01-01 00:00',periods=17,freq='10min')
        frame=pd.DataFrame({'date':stamps,'Appliances':np.arange(17)})
        for name in ['T'+str(i) for i in range(1,10)]+['RH_'+str(i) for i in range(1,10)]+['lights']:
            frame[name]=np.arange(17)
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'test.zip'
            with zipfile.ZipFile(path,'w') as z:z.writestr('energydata_complete.csv',frame.to_csv(index=False))
            series,extra,meta=experiment.source_series(dict(name='appliances',time_column='date',outcome_column='Appliances'),path)
            self.assertEqual(series.iloc[0],15)
            self.assertEqual(series.iloc[1],51)
            self.assertTrue(np.isnan(series.iloc[2]))
            self.assertTrue(np.isnan(extra.previous_hour_T1.iloc[0]))
            self.assertEqual(extra.previous_hour_T1.iloc[1],2.5)

    def test_bound_rejects_wrong_independent_center(self):
        with self.assertRaises(AssertionError):classical(1.01,np.ones(20),2.)
        z=classical(0.,np.zeros(20),0.,full=True)
        self.assertEqual(z['p'],1.)

    def test_by_fixed_family_keeps_nulls(self):
        q=by_values([.0001,1.,1.,1.,1.,1.,1.,1.])
        self.assertLess(q[0],.05)
        np.testing.assert_equal(q[1:],1.)


if __name__=='__main__':unittest.main()
