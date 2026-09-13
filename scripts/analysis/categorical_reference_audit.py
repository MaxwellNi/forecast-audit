"""Audit one forecast using independent categorical validation and peer groups.

Fits CSV: category, forecast_mean, outcome_mean.
Validation CSV: pair, role (focal/reference), category, forecast, outcome.
Evaluation CSV: group, category, forecast, outcome.
All categories must be declared in the fits. Every pair has exactly two rows,
and evaluation groups have a common size >=3. These checks do not establish
the required independence or the common sampling law.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from category_reference_certificate import weighted_category_bias_budget, finite_reference_decision
from peer_rank_products import peer_rank_products
from reference_certificate_efficiency import (
    family_validation_delta, signed_category_budget, triple_scores,
    one_sided_certificate,
)


def _required(frame, columns, label):
    if not set(columns).issubset(frame.columns) or frame.empty:
        raise ValueError(f'{label} needs nonempty columns {columns}')
    if frame[list(columns)].isna().any().any():
        raise ValueError(f'{label} contains missing values')


def audit(fits, validation, evaluation, delta=None, alpha=.05,
          certificate='absolute_range', family_size=None, family_rho=.1):
    """Use one prespecified certificate and validation failure allocation.

    The default preserves the probability-weighted absolute range bound.
    Variance scores use consecutive input rows within each group: their order
    must be fixed independently of values. No sorting or best-bound selection
    is performed. A family size chooses delta, but does not apply BY itself.
    """
    if certificate not in ('absolute_range', 'signed_range', 'signed_variance'):
        raise ValueError('Unknown prespecified certificate')
    if family_size is not None:
        if delta is not None:
            raise ValueError('Declare either delta or family_size, not both')
        delta = family_validation_delta(family_size, q=alpha, rho=family_rho)
    elif delta is None:
        delta = .0001
    _required(fits,['category','forecast_mean','outcome_mean'],'fits')
    _required(validation,['pair','role','category','forecast','outcome'],'validation')
    _required(evaluation,['group','category','forecast','outcome'],'evaluation')
    fits,validation,evaluation=[d.copy() for d in (fits,validation,evaluation)]
    for d in (fits,validation,evaluation):
        d['category']=d.category.astype(str)
    if fits.category.duplicated().any():
        raise ValueError('Each declared category needs one pair of fitted means')
    categories={c:i for i,c in enumerate(fits.category)}
    for d in (validation,evaluation):
        if not d.category.isin(categories).all():
            raise ValueError('Undeclared category encountered')
        if not np.isfinite(d[['forecast','outcome']].to_numpy(float)).all():
            raise ValueError('Scores and outcomes must be finite')
    if set(validation.role)!={'focal','reference'}:
        raise ValueError('Validation roles must be focal and reference')
    if validation.duplicated(['pair','role']).any() or not validation.groupby('pair').size().eq(2).all():
        raise ValueError('Each validation pair must have one focal and one reference')
    focal=validation[validation.role=='focal'].set_index('pair').sort_index()
    reference=validation[validation.role=='reference'].set_index('pair').reindex(focal.index)
    z=focal.category.map(categories).to_numpy(int)
    count=np.bincount(z,minlength=len(fits))
    means=[]
    for name in ('forecast','outcome'):
        left=focal[name].to_numpy(float);right=reference[name].to_numpy(float)
        comparisons=(left>right).astype(float)+.5*(left==right)
        means.append(np.divide(np.bincount(z,weights=comparisons,minlength=len(fits)),
                               count,out=np.zeros(len(fits)),where=count>0))
    f,g=fits.forecast_mean.to_numpy(float),fits.outcome_mean.to_numpy(float)
    budget_function = weighted_category_bias_budget if certificate == 'absolute_range' else signed_category_budget
    budget=budget_function(f,g,count,*means,delta=delta)
    sizes=evaluation.groupby('group',sort=False).size()
    if sizes.nunique()!=1 or sizes.iloc[0]<3:
        raise ValueError('Evaluation groups need the same size, at least three')
    scores=[]
    independent_triples=[]
    for _,group in evaluation.groupby('group',sort=False):
        c=group.category.map(categories).to_numpy(int)
        v,w=group.forecast.to_numpy(float),group.outcome.to_numpy(float)
        values=peer_rank_products(v,w,f[c],g[c])
        scores.append(float(values['corrected_residual_product'].mean()))
        if certificate == 'signed_variance':
            independent_triples.append(triple_scores(v[None,:],w[None,:],f[c][None,:],g[c][None,:]))
    if certificate == 'absolute_range':
        result=finite_reference_decision(scores,int(sizes.iloc[0]),f,g,budget['bias_upper'],delta,alpha)
    else:
        effective=len(sizes)*(int(sizes.iloc[0])//3)
        h=np.concatenate(independent_triples) if independent_triples else None
        result=one_sided_certificate(float(np.mean(scores)),h,effective,f,g,budget['bias_upper'],
                                     delta=delta,alpha=alpha,
                                     kind='variance' if certificate == 'signed_variance' else 'range')
        result['group_se']=float(np.std(scores,ddof=1)/np.sqrt(len(scores))) if len(scores)>1 else float('nan')
        result['bias_to_se']=float(budget['bias_upper']/result['group_se']) if result['group_se']>0 else None
        result['bias_lower']=budget['bias_lower']
        result['absolute_bias_upper']=budget['absolute_bias_upper']
    if not np.isfinite(result['group_se']):
        result['group_se']=None
    result['certificate']=certificate
    return dict(result=result,scope='Population marginal-midrank residual covariance under the declared iid common-law design',
                decision='POSITIVE_LOWER_BOUND' if result['reject'] else 'NO_POSITIVE_LOWER_BOUND',
                sampling_assumptions_verified=False,groups=len(sizes),peers_per_group=int(sizes.iloc[0]),
                declared_categories=len(fits),validation_pairs=len(focal),validation_observations=len(validation),
                evaluation_observations=len(evaluation),delta=delta,alpha=alpha,
                certificate=certificate,
                family_size=int(family_size) if family_size is not None else None,
                family_fdr_level=alpha if family_size is not None else None,
                family_validation_rho=family_rho if family_size is not None else None,
                family_first_by_threshold=delta/family_rho if family_size is not None else None,
                family_adjustment_applied=False,
                variance_partition='Consecutive disjoint triples in supplied within-group row order; six roles averaged into each single score.' if certificate == 'signed_variance' else None,
                variance_partition_rows=len(sizes)*3*(int(sizes.iloc[0])//3) if certificate == 'signed_variance' else 0,
                category_validation_counts=count.tolist(),category_probability_upper=budget['category_mass_upper'].tolist(),
                required_design='Training, validation and evaluation are independent; validation pairs are disjoint; evaluation peers and groups are iid from the same law. The certificate, budget and input row ordering are fixed independently of evaluation values.',
                interpretation='A positive bound concerns this declared rank-scale covariance. It does not establish raw-unit forecasting gain or validate repeated fixed-peer panels.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('fits','validation','evaluation','output'):
        p.add_argument('--'+name,type=Path,required=True)
    allocation=p.add_mutually_exclusive_group()
    allocation.add_argument('--delta',type=float,default=None,
                            help='Prespecified validation failure probability (default: 0.0001)')
    allocation.add_argument('--family-size',type=int,
                            help='Prespecified BY family size K; uses delta=rho*alpha/(K*H_K)')
    p.add_argument('--family-rho',type=float,default=.1,
                   help='Validation share of first BY threshold when --family-size is given (default: 0.1)')
    p.add_argument('--alpha',type=float,default=.05)
    p.add_argument('--certificate',choices=['absolute_range','signed_range','signed_variance'],
                   default='absolute_range',help='Rule fixed before evaluation; default uses the absolute range bound')
    args=p.parse_args()
    if args.output.exists():p.error('Use a new output path')
    files=[args.fits,args.validation,args.evaluation]
    result=audit(*(pd.read_csv(x,dtype={'category':str}) for x in files),delta=args.delta,alpha=args.alpha,
                 certificate=args.certificate,family_size=args.family_size,family_rho=args.family_rho)
    result['input_sha256']={name:hashlib.sha256(path.read_bytes()).hexdigest()
                            for name,path in zip(('fits','validation','evaluation'),files)}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'decision':result['decision'],'lower_bound':result['result']['lower_bound']}))
