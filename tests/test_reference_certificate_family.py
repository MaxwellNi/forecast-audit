"""Family-resolution and generic CSV certificate checks."""
import itertools
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts/analysis'))
from categorical_reference_audit import audit
from reference_certificate_efficiency import family_validation_delta, one_sided_certificate


class FamilyResolutionTests(unittest.TestCase):
    def test_prespecified_family_budget_allows_isolated_by_signal(self):
        k,q,rho=100,.05,.1
        first=q/(k*sum(1/j for j in range(1,k+1)))
        delta=family_validation_delta(k,q,rho)
        self.assertAlmostEqual(delta,rho*first,places=17)
        old=one_sided_certificate(.1,None,100000,[.5],[.5],0,kind='range')
        new=one_sided_certificate(.1,None,100000,[.5],[.5],0,delta=delta,kind='range')
        self.assertGreater(old['p'],first)
        self.assertLess(new['p'],first)
        self.assertLess(old['p'],2*first)  # The old floor does not block every BY rank.

    def test_family_budget_validation_and_size_one(self):
        self.assertAlmostEqual(family_validation_delta(1,.05,.1),.005)
        for k in (0,-1,True,2.5):
            with self.assertRaises(ValueError):family_validation_delta(k)
        for q,rho in [(0,.1),(1,.1),(.05,0),(.05,1),(np.nan,.1),(.05,np.inf)]:
            with self.assertRaises(ValueError):family_validation_delta(3,q,rho)


class GenericSignedCliTests(unittest.TestCase):
    def setUp(self):
        self.fits=pd.DataFrame({'category':['a','b'],'forecast_mean':[.4,.6],'outcome_mean':[.6,.4]})
        self.validation=pd.DataFrame([
            [0,'focal','a',1.,4.],[0,'reference','b',3.,3.],
            [1,'focal','b',4.,2.],[1,'reference','a',1.,5.]],
            columns=['pair','role','category','forecast','outcome'])
        observations=[('a',1.,4.),('b',3.,3.),('b',3.,5.),('a',2.,2.),
                      ('a',7.,1.),('b',8.,4.),('a',-2.,8.)]
        self.evaluation=pd.DataFrame([[g,c,x+(g==1)*i,y+(g==1)*(i%3)]
                                     for g in range(2) for i,(c,x,y) in enumerate(observations)],
                                    columns=['group','category','forecast','outcome'])

    def literal(self, group, indices):
        means=self.fits.set_index('category')
        values=[]
        for i,j,k in itertools.permutations(indices,3):
            av=float(group.forecast[i]>group.forecast[j])+.5*(group.forecast[i]==group.forecast[j])
            bw=float(group.outcome[i]>group.outcome[k])+.5*(group.outcome[i]==group.outcome[k])
            f,g=means.loc[group.category[i],['forecast_mean','outcome_mean']]
            values.append((av-f)*(bw-g))
        return np.mean(values)

    def test_full_generic_mean_and_six_role_variance(self):
        expected_means=[];expected_triples=[]
        for _,group in self.evaluation.groupby('group',sort=False):
            group=group.reset_index(drop=True)
            expected_means.append(self.literal(group,range(7)))
            expected_triples.extend(self.literal(group,range(start,start+3)) for start in (0,3))
        result=audit(self.fits,self.validation,self.evaluation,certificate='signed_variance')
        self.assertAlmostEqual(result['result']['mean'],np.mean(expected_means),places=14)
        self.assertAlmostEqual(result['result']['kernel_sample_variance'],np.var(expected_triples,ddof=1),places=14)
        self.assertEqual(result['result']['effective_triples'],4)
        self.assertEqual(result['evaluation_observations'],14)
        self.assertEqual(result['variance_partition_rows'],12)
        changed=self.evaluation.copy()
        changed.loc[[6,13],['forecast','outcome']]=[100.,100.]
        other=audit(self.fits,self.validation,changed,certificate='signed_variance')
        self.assertAlmostEqual(result['result']['kernel_sample_variance'],other['result']['kernel_sample_variance'],places=14)
        self.assertNotAlmostEqual(result['result']['mean'],other['result']['mean'],places=8)

    def test_backward_compatibility_and_fixed_method(self):
        legacy=audit(self.fits,self.validation,self.evaluation)
        explicit=audit(self.fits,self.validation,self.evaluation,delta=.0001,certificate='absolute_range')
        self.assertEqual(legacy,explicit)
        signed=audit(self.fits,self.validation,self.evaluation,certificate='signed_range')
        self.assertAlmostEqual(legacy['result']['mean'],signed['result']['mean'],places=14)
        self.assertLessEqual(signed['result']['bias_upper'],legacy['result']['bias_upper']+1e-13)
        self.assertGreaterEqual(signed['result']['lower_bound'],legacy['result']['lower_bound']-1e-13)
        with self.assertRaises(ValueError):
            audit(self.fits,self.validation,self.evaluation,certificate='best')
        with self.assertRaises(ValueError):
            audit(self.fits,self.validation,self.evaluation,delta=.0001,family_size=100)
        with self.assertRaises(ValueError):
            audit(self.fits,self.validation,self.evaluation.iloc[:3],certificate='signed_variance')

    def test_cli_options_and_family_metadata(self):
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)
            for name,frame in [('fits',self.fits),('validation',self.validation),('evaluation',self.evaluation)]:
                frame.to_csv(path/(name+'.csv'),index=False)
            for mode in ('absolute_range','signed_range','signed_variance'):
                target=path/(mode+'.json')
                command=[sys.executable,str(ROOT/'scripts/analysis/categorical_reference_audit.py')]
                for name in ('fits','validation','evaluation'):
                    command.extend(['--'+name,str(path/(name+'.csv'))])
                command.extend(['--output',str(target),'--certificate',mode,'--family-size','100'])
                subprocess.run(command,check=True,capture_output=True,text=True)
                result=json.loads(target.read_text())
                self.assertEqual(result['certificate'],mode)
                self.assertEqual(result['family_size'],100)
                self.assertAlmostEqual(result['delta'],family_validation_delta(100),places=17)
                self.assertFalse(result['family_adjustment_applied'])
                self.assertFalse(result['sampling_assumptions_verified'])
                self.assertEqual(set(result['input_sha256']),{'fits','validation','evaluation'})
            conflict=subprocess.run(command+['--delta','.0001'],capture_output=True,text=True)
            self.assertNotEqual(conflict.returncode,0)


if __name__=='__main__':unittest.main()
