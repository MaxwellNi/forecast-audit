"""Independent checks of the raw equal-entity benchmark's target and support."""
from pathlib import Path
import importlib.util
import unittest
import tempfile

import numpy as np

PATH=Path(__file__).resolve().parents[1]/'scripts/analysis/equal_entity_directional.py'
SPEC=importlib.util.spec_from_file_location('equal_entity_directional',PATH)
audit=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(audit)


class DirectionalChecks(unittest.TestCase):
    def test_refuses_existing_output(self):
        with tempfile.TemporaryDirectory() as directory:
            marker=Path(directory)/"unchanged.txt"
            marker.write_text("preserve")
            with self.assertRaises(FileExistsError):
                audit.run(directory,repetitions=2)
            self.assertEqual(marker.read_text(),"preserve")
            self.assertEqual(list(Path(directory).iterdir()),[marker])

    def test_exact_bias_and_directional_expectation(self):
        # Closed form applies when the lookback is shorter than each balanced
        # fold. The exact matrix expression also covers longer lookbacks.
        for length,past in ((20,1),(40,1),(40,5)):
            folds=5;block=length//folds;loading=.2
            total=loading*folds*past*(block-past-1)/((folds-1)*block)
            self.assertAlmostEqual(audit.exact_mean(length,past,loading,'standard',middle_only=False)*length,total,places=12)
            for signal in (0.,.1,.3):
                self.assertAlmostEqual(audit.exact_mean(length,past,loading,'directional',signal),signal,places=12)
            self.assertGreater(audit.exact_mean(length,past,loading,'gapped_complementary'),0.)

    def test_aligned_eligibility_and_equal_entity_weights(self):
        rng=np.random.default_rng(743)
        x,y=rng.normal(size=(2,3,20));folds=audit.make_folds(20)
        eligible=np.ones((3,20),bool);eligible[1,[0,5,9,13]]=False;eligible[2,[3,6,11,16]]=False
        scores,used=audit.entity_scores(x,y,folds,eligible)
        for g in range(3):
            manual={'standard':[],'directional':[]};indices=[]
            for t in range(20):
                f=folds[t]
                if not eligible[g,t] or f in (0,4):continue
                earlier=np.flatnonzero(eligible[g]&(folds<f));later=np.flatnonzero(eligible[g]&(folds>f))
                other=np.flatnonzero(eligible[g]&(folds!=f))
                if not len(earlier) or not len(later):continue
                indices.append(t)
                manual['standard'].append((x[g,t]-np.mean(x[g,other]))*(y[g,t]-np.mean(y[g,other])))
                manual['directional'].append((x[g,t]-np.mean(x[g,earlier]))*(y[g,t]-np.mean(y[g,later])))
            np.testing.assert_array_equal(np.flatnonzero(used[g]),indices)
            for method in manual:self.assertAlmostEqual(scores[method][g],np.mean(manual[method]),places=12)
        self.assertAlmostEqual(audit.studentize(scores['directional'])[0],np.mean(scores['directional']),places=12)
        with self.assertRaises(ValueError):audit.entity_scores(x,y,folds,folds==2)
        bad=folds.copy();bad[4],bad[8]=bad[8],bad[4]
        with self.assertRaises(ValueError):audit.entity_scores(x,y,bad)
        with self.assertRaises(ValueError):audit.entity_scores(x,y,bad.astype(np.uint64))
        bad=folds.astype(float);bad[4]=1.5
        with self.assertRaises(ValueError):audit.entity_scores(x,y,bad)
        bad=folds.astype(float);bad[4]=np.nan
        with self.assertRaises(ValueError):audit.entity_scores(x,y,bad)
        with self.assertRaises(ValueError):audit.entity_scores(x,y,folds+1)

    def test_independent_entity_null_centres_at_exact_target(self):
        rng=np.random.default_rng(487)
        groups,length,past=10000,20,5
        innovations=rng.normal(size=(groups,length+past))
        y=innovations[:,past:]
        x=.2*sum(innovations[:,past-h:past+length-h] for h in range(1,past+1))+rng.normal(size=(groups,length))
        scores,_=audit.entity_scores(x,y,audit.make_folds(length))
        for method in scores:
            mean,se,_=audit.studentize(scores[method])
            exact=audit.exact_mean(length,past,.2,method)
            self.assertLess(abs(mean-exact),6*se)


if __name__=='__main__':unittest.main()
