"""Independent exact certificates and fixed-archive reconstruction; no producer imports.
No optimization solver is imported or called. Validates arbitrary rational duals
and real finite tied-rank laws separately from archived solver proposals.
"""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
from fractions import Fraction as F
import argparse,hashlib,json,math,platform,itertools,time
from integrity import check_manifest
import numpy as np
import pandas as pd
from scipy.stats import beta
from scipy.special import logsumexp
HERE=Path(__file__).resolve().parent
SOURCE=HERE
STUDY=HERE/'inputs'
NAMES=['f','g','lower_x','upper_x','lower_y','upper_y','mass_lower','mass_upper']
Q=lambda x:F.from_float(float(x))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def ceilfloat(x):
    f=float(x)
    if Q(f)<x:f=math.nextafter(f,math.inf)
    assert Q(f)>=x
    assert Q(math.nextafter(f,-math.inf))<x
    return f

def build(data,equalities=True):
    z={k:list(map(Q,data[k])) for k in NAMES};m=len(z['f']);n=4*m
    c=[F(0)]*n;A=[];b=[]
    # Deliberately use grouped coordinates p[0:m],u[m:2m],v[2m:3m],w[3m:4m].
    def row(items,rhs=0):
        v=[F(0)]*n
        for j,val in items:v[j]+=val
        A.append(v);b.append(F(rhs))
    for k in range(m):
        pi,ui,vi,wi=k,m+k,2*m+k,3*m+k
        lx,hx,ly,hy=(z[s][k] for s in ['lower_x','upper_x','lower_y','upper_y'])
        f,g=z['f'][k],z['g'][k]
        for j,val in [(pi,f*g),(ui,-g),(vi,-f),(wi,F(1))]:c[j]=val
        row([(pi,1)],z['mass_upper'][k]);row([(pi,-1)],-z['mass_lower'][k])
        row([(pi,-hx),(ui,1)]);row([(pi,lx),(ui,-1)])
        row([(pi,-hy),(vi,1)]);row([(pi,ly),(vi,-1)])
        row([(pi,-lx*ly),(ui,ly),(vi,lx),(wi,-1)])
        row([(pi,-hx*hy),(ui,hy),(vi,hx),(wi,-1)])
        row([(pi,hx*ly),(ui,-ly),(vi,-hx),(wi,1)])
        row([(pi,lx*hy),(ui,-hy),(vi,-lx),(wi,1)])
    E=[[F(int(i<m)) for i in range(n)]];d=[F(1)]
    if equalities:
        E += [[F(int(m<=i<2*m)) for i in range(n)],[F(int(2*m<=i<3*m)) for i in range(n)]];d += [F(1,2)]*2
    return c,A,b,E,d

def dot(a,b):return sum((x*y for x,y in zip(a,b)),F(0))

def dual(data,y,z,equalities=True):
    c,A,b,E,d=build(data,equalities);y=list(map(Q,y));z=list(map(Q,z))
    assert len(A)==len(y) and len(E)==len(z) and min(y)>=0
    r=[c[j]-sum(A[i][j]*y[i] for i in range(len(A)))-sum(E[i][j]*z[i] for i in range(len(E))) for j in range(len(c))]
    correction=sum((max(t,F(0)) for t in r),F(0))
    return dot(b,y)+dot(d,z)+correction,correction,r

def separate(data):
    d={k:list(map(Q,v)) for k,v in data.items()};l=d['mass_lower'];u=d['mass_upper'];m=len(l)
    if sum(l)>1 or sum(u)<1:return None
    q=[max((x-d['f'][k])*(y-d['g'][k]) for x,y in itertools.product([d['lower_x'][k],d['upper_x'][k]],[d['lower_y'][k],d['upper_y'][k]])) for k in range(m)]
    p=list(l);remaining=1-sum(p)
    for k in sorted(range(m),key=lambda k:q[k],reverse=True):
        add=min(remaining,u[k]-p[k]);p[k]+=add;remaining-=add
    assert remaining==0
    return dot(p,q)

def check_receipt(r):
    sep=separate(r['inputs'])
    if sep is None:
        assert r['status']=='fallback_empty_probability_box'
        assert r['bias_upper']==r['separable_upper']==1;return
    assert r['separable_upper']==ceilfloat(sep)
    if r['status']!='certified':assert r['bias_upper']==ceilfloat(sep);return
    a=r['audit'];upper,corr,res=dual(r['inputs'],a['inequality_multipliers'],a['equality_multipliers'],r['mean_constraints'])
    assert str(upper)==a['certified_upper_fraction'] and str(corr)==a['positive_residual_fraction']
    assert str(min(upper,sep))==a['chosen_upper_fraction']
    assert r['bias_upper']==ceilfloat(min(upper,sep)) and r['certified_dual_upper']==ceilfloat(upper)
    assert r['bias_upper']<=r['separable_upper']
    assert float(max(map(abs,res)))==a['max_absolute_residual']

def ranks(x):
    # Independent sort/equal-group average rank, not scipy rankdata.
    ix=np.argsort(x,kind='stable');out=np.empty(len(x));start=0
    while start<len(x):
        stop=start+1
        while stop<len(x) and x[ix[stop]]==x[ix[start]]:stop+=1
        out[ix[start:stop]]=(start+stop+1)/2;start=stop
    return out

def category(cat,value):
    out=np.full(32,.5)
    for k in range(32):
        if np.any(cat==k):out[k]=np.mean(value[cat==k])
    return out

def numeric(a,b):return np.where(a>b,1.,np.where(a==b,.5,0.))

def triple_values(x,y,f,g):
    x,y,f,g=[v.reshape(-1,3) for v in [x,y,f,g]]
    terms=[]
    for i,j,k in itertools.permutations(range(3)):
        terms.append((numeric(x[:,i],x[:,j])-f[:,i])*(numeric(y[:,i],y[:,k])-g[:,i]))
    return np.sum(terms,axis=0)/6

def coherent_random_checks():
    rng=np.random.default_rng(202609122237);tests=0
    for m in [1,2,4,8,16]:
        for repetition in range(40):
            n=64;cats=np.tile(np.arange(m),n//m);rng.shuffle(cats)
            xx=rng.integers(0,11,n);yy=rng.integers(0,7,n)
            rx=[F(int(sum(2*int(v>u)+int(v==u) for u in xx)),2*n) for v in xx]
            ry=[F(int(sum(2*int(v>u)+int(v==u) for u in yy)),2*n) for v in yy]
            p=[F(1,m)]*m;mx=[sum((rx[i] for i in range(n) if cats[i]==k),F(0))/(n//m) for k in range(m)]
            my=[sum((ry[i] for i in range(n) if cats[i]==k),F(0))/(n//m) for k in range(m)]
            assert dot(p,mx)==dot(p,my)==F(1,2)
            fx=[F(int(t),64) for t in rng.integers(0,65,m)];gy=[F(int(t),64) for t in rng.integers(0,65,m)]
            data=dict(f=fx,g=gy,lower_x=[max(F(0),v-F(1,16)) for v in mx],upper_x=[min(F(1),v+F(1,16)) for v in mx],lower_y=[max(F(0),v-F(1,8)) for v in my],upper_y=[min(F(1),v+F(1,8)) for v in my],mass_lower=[F(0)]*m,mass_upper=[min(F(1),v+F(1,16)) for v in p])
            data={k:list(map(float,v)) for k,v in data.items()}
            x=p+[p[k]*mx[k] for k in range(m)]+[p[k]*my[k] for k in range(m)]+[p[k]*mx[k]*my[k] for k in range(m)]
            c,A,b,E,d=build(data);assert all(dot(a,x)<=bb for a,bb in zip(A,b));assert all(dot(e,x)==dd for e,dd in zip(E,d))
            true=dot(c,x);assert true<=separate(data)
            # Any nonnegative inequality multipliers work; negative residuals do not need correction.
            y=rng.integers(0,33,len(A))/32;z=rng.integers(-32,33,len(E))/32
            upper,correction,_=dual(data,y,z);assert true<=upper;assert ceilfloat(upper)>=upper
            tests+=1
    return tests

def corner_equivalence_checks():
    rng=np.random.default_rng(202609122247)
    tests=0
    for rep in range(240):
        lx,hx=sorted(F(int(t),32) for t in rng.integers(0,33,2))
        ly,hy=sorted(F(int(t),32) for t in rng.integers(0,33,2))
        if rep%5==0:hx=lx
        if rep%7==0:hy=ly
        raw=[int(t) for t in rng.integers(0,33,4)];total=max(1,sum(raw))
        lam=[F(t,total) for t in raw]
        if rep%11==0:lam=[F(0)]*4
        corners=list(itertools.product([lx,hx],[ly,hy]));p=sum(lam)
        u=sum(v*a for v,(a,b) in zip(lam,corners));v=sum(z*b for z,(a,b) in zip(lam,corners));w=sum(z*a*b for z,(a,b) in zip(lam,corners))
        assert w>=ly*u+lx*v-lx*ly*p and w>=hy*u+hx*v-hx*hy*p
        assert w<=ly*u+hx*v-hx*ly*p and w<=hy*u+lx*v-lx*hy*p
        if hx>lx and hy>ly:
            hh=(w-ly*u-lx*v+lx*ly*p)/((hx-lx)*(hy-ly));hl=(u-lx*p)/(hx-lx)-hh;lh=(v-ly*p)/(hy-ly)-hh
            recovered=[p-hh-hl-lh,lh,hl,hh]
        elif hx==lx and hy>ly:recovered=[p-(v-ly*p)/(hy-ly),(v-ly*p)/(hy-ly),F(0),F(0)]
        elif hy==ly and hx>lx:recovered=[p-(u-lx*p)/(hx-lx),F(0),(u-lx*p)/(hx-lx),F(0)]
        else:recovered=[p,F(0),F(0),F(0)]
        assert min(recovered)>=0 and sum(recovered)==p
        assert sum(z*a for z,(a,b) in zip(recovered,corners))==u
        assert sum(z*b for z,(a,b) in zip(recovered,corners))==v
        assert sum(z*a*b for z,(a,b) in zip(recovered,corners))==w
        tests+=1
    return tests

def examples():
    # Exact affine majorant xy <= (x+y)/2 - 3/16 at every corner
    # of both strict-improvement rectangles. Weighted global means total one.
    for lo,hi in [(F(1,4),F(3,8)),(F(5,8),F(3,4))]:
        assert all(x*y<=(x+y)/2-F(3,16) for x,y in itertools.product([lo,hi],repeat=2))
    assert F(1,2)-F(3,16)==F(5,16)
    assert (F(1,4)**2+F(3,4)**2)/2==F(5,16)
    assert (F(3,8)**2+F(3,4)**2)/2==F(45,128)
    # The other relaxed maxima equal the largest corner products, achieved
    # by half mass at the equal-low and equal-high corners of every cell.
    for lo,hi,expected in [(F(0),F(1),F(1,4)),(F(3,8),F(5,8),F(1,64))]:
        assert (lo+hi)/2==F(1,2)
        assert max((x-F(1,2))*(y-F(1,2)) for x,y in itertools.product([lo,hi],repeat=2))==expected
        assert ((lo-F(1,2))**2+(hi-F(1,2))**2)/2==expected
    # Coherent three-cell maximum via explicit sliced-box vertices (no optimizer).
    p=[F(1,4),F(3,8),F(3,8)];vertices=[]
    for free in range(3):
        fixed=[i for i in range(3) if i!=free]
        for signs in itertools.product([-1,1],repeat=2):
            t=[F(0)]*3
            for k,z in zip(fixed,signs):t[k]=F(z)
            t[free]=-sum(p[k]*t[k] for k in fixed)/p[free]
            if abs(t[free])<=1:vertices.append(dot(p,[v*v for v in t]))
    assert max(vertices)==F(3,4)
    # p-weighted conditional rank means from a valid linear category propensity law.
    means=[F(1,2),F(1,2)+F(9,16)*F(1,12)/F(3,8),F(1,2)-F(9,16)*F(1,12)/F(3,8)]
    assert means==[F(1,2),F(5,8),F(3,8)]
    for z in [F(0),F(1)]:
        prob=[F(1,4),F(3,8)+F(9,16)*(z-F(1,2)),F(3,8)-F(9,16)*(z-F(1,2))]
        assert min(prob)>=0 and sum(prob)==1
    return {'three_category_coherent_optimum':'3/256','three_category_relaxed_optimum':'1/64','three_category_gap':'1/256','single_category_true_bias':'0','single_category_relaxed_optimum':'1/4','strict_improvement_separable':'45/128','strict_improvement_joint':'5/16'}

def run(results=None,inputs=None,output=None):
    global STUDY
    STUDY=Path(inputs).resolve() if inputs is not None else HERE/'inputs'
    results=Path(results).resolve() if results is not None else HERE/'recorded'
    check_manifest()
    watched=[STUDY/'selection_certificates.csv']+[STUDY/task/name for task in ['appliances','metro'] for name in ['forecast_archive.npz','sampling_indices.npz']]
    before={str(p):sha(p) for p in watched}
    if output is not None and Path(output).exists():raise FileExistsError('Output directory must not exist')
    begin=time.perf_counter();frame=pd.read_csv(results/'all_candidates.csv',float_precision='round_trip')
    old=pd.read_csv(STUDY/'selection_certificates.csv',float_precision='round_trip');delta=.1*.05/(8*sum(1/k for k in range(1,9)))
    maximum={'input':0.,'source_bias':0.,'mean':0.,'variance':0.,'target':0.,'bias_truth':0.,'p':0.};rows=[]
    def compare(name,a,b,tol=2e-12):
        err=float(np.max(np.abs(np.asarray(a)-np.asarray(b))));maximum[name]=max(maximum[name],err);assert err<tol,(name,err)
    for task in ['appliances','metro']:
        ar=np.load(STUDY/task/'forecast_archive.npz');dr=np.load(STUDY/task/'sampling_indices.npz');sel=ar['selection'];y=ar['y'][sel]
        tr,va,ev=[dr[k] for k in ['training','validation','evaluation']]
        for base in ['seasonal_day','ridge']:
            cat=ar[base+'__category'][sel];vcat=cat[va[:,0]];counts=np.bincount(vcat,minlength=32)
            for _,row in frame[(frame.task==task)&(frame.baseline==base)].iterrows():
                receipt=json.loads((results/'certificates'/row.certificate).read_text());check_receipt(receipt)
                x=ar[base+'__'+row.candidate][sel]
                f=category(cat[tr],(ranks(x[tr])-1)/(len(tr)-1));g=category(cat[tr],(ranks(y[tr])-1)/(len(tr)-1))
                mx=category(vcat,numeric(x[va[:,0]],x[va[:,1]]));my=category(vcat,numeric(y[va[:,0]],y[va[:,1]]))
                r=np.array([math.sqrt(math.log(8*32/delta)/(2*n)) if n else math.inf for n in counts])
                caps=np.array([beta.isf(delta/64,int(n)+1,len(va)-int(n)) if n<len(va) else 1. for n in counts])
                expected=dict(f=f,g=g,lower_x=np.maximum(0,mx-r),upper_x=np.minimum(1,mx+r),lower_y=np.maximum(0,my-r),upper_y=np.minimum(1,my+r),mass_lower=np.zeros(32),mass_upper=caps)
                for name,values in expected.items():compare('input',values,receipt['inputs'][name])
                source=old[(old.task==task)&(old.baseline==base)&(old.candidate==row.candidate)]
                ru=source[source.method=='reference_u'].iloc[0];rb=source[source.method=='reference_betting'].iloc[0]
                compare('source_bias',row.old_bias,ru.bias)
                assert row.joint_bias==receipt['bias_upper'] and row.old_bias==receipt['separable_upper']
                h=triple_values(x[ev],y[ev],f[cat[ev]],g[cat[ev]])
                compare('mean',h.mean(),rb['mean']);compare('variance',h.var(ddof=1),ru.variance)
                support=[(u-ff)*(v-gg) for ff,gg in zip(f,g) for u,v in itertools.product([0,1],repeat=2)]
                lo,hi=min(support),max(support);width=hi-lo;j=len(h)
                compare('mean',row.full_u_mean,ru['mean']);compare('mean',row.radius,ru.radius)
                margin=max(0,row.full_u_mean-row.joint_bias);A=math.sqrt(2*h.var(ddof=1)/j);C=2*width/math.sqrt(j*(j-1))+width/(3*j)
                root=(math.sqrt(A*A+4*C*margin)-A)/(2*C)
                pu=min(1,delta+2*math.exp(-max(root*root,2*j*margin*margin/width**2)))
                if row.joint_bias>=hi:pb=1.
                else:
                    z=(h-lo)/width;t=(row.joint_bias-lo)/width
                    capitals=np.array([sum(np.log(1+q*(z/t-1))) for q in np.geomspace(1e-4,.99,64)])
                    logE=logsumexp(capitals)-math.log(64);pb=min(1,delta+math.exp(-logE)) if logE>0 else 1.
                compare('p',pu,row.joint_variance_p);compare('p',pb,row.joint_betting_p)
                rx,ry=(ranks(x)-.5)/len(x),(ranks(y)-.5)/len(y);tx,ty=category(cat,rx),category(cat,ry)
                prob=np.bincount(cat,minlength=32)/len(cat)
                theta=np.mean((rx-tx[cat])*(ry-ty[cat]));bias=np.dot(prob,(tx-f)*(ty-g))
                compare('target',theta,row.exact_target_diagnostic);compare('bias_truth',bias,row.true_bias_diagnostic)
                assert row.joint_bias>=bias and row.joint_bias<=row.old_bias
                assert row.validation_event_diagnostic and np.all(prob<=caps) and np.all((tx>=expected['lower_x'])&(tx<=expected['upper_x'])) and np.all((ty>=expected['lower_y'])&(ty<=expected['upper_y']))
                assert row.old_lower==row.full_u_mean-row.old_bias-row.radius
                assert row.joint_lower==row.full_u_mean-row.joint_bias-row.radius
                rows.append(dict(task=task,baseline=base,candidate=row.candidate,dual_verified=True,joint_bias=row.joint_bias,old_bias=row.old_bias,p_variance=pu,p_betting=pb))
    assert len(rows)==32 and frame[['joint_variance_by','joint_betting_by']].eq(1).all().all()
    # All raw values are one, so BY must also be exactly one in every family.
    assert frame[['old_variance_p','joint_variance_p','old_betting_p','joint_betting_p']].eq(1).all().all()
    negative=json.loads((SOURCE/'negative_signed_check.json').read_text());check_receipt(negative['certificate'])
    assert negative['certificate']['bias_upper']==-15/1024 and negative['certificate']['separable_upper']==-9/1024
    result=dict(status='PASS',scope='Independent no-solver rational certificate verification; all 32 existing archive candidates; no producer imports or new domain observations.',certificate_rows=len(rows),max_errors=maximum,random_finite_tied_rank_law_checks=coherent_random_checks(),exact_corner_equivalence_checks=corner_equivalence_checks(),negative_signed_certificate_verified=True,exact_examples=examples(),strict_allowance_improvements=int((frame.old_bias>frame.joint_bias).sum()),median_relative_improvement=float(np.median((frame.old_bias-frame.joint_bias)/frame.old_bias)),maximum_relative_improvement=float(np.max((frame.old_bias-frame.joint_bias)/frame.old_bias)),variance_rejections=0,betting_rejections=0,environment=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__),elapsed_seconds=time.perf_counter()-begin,source_sha256={'joint_bias.py':sha(HERE/'joint_bias.py'),'all_candidates.csv':sha(results/'all_candidates.csv'),'MANIFEST.json':sha(HERE/'MANIFEST.json')},verifier_sha256=sha(__file__))
    assert all(sha(Path(p))==expected for p,expected in before.items()),'An input changed during verification'
    check_manifest()
    if output is not None:
        output=Path(output);output.mkdir(parents=True,exist_ok=False)
        pd.DataFrame(rows).to_csv(output/'verified_candidates.csv',index=False)
        (output/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
    return result

if __name__=='__main__':
    parser=argparse.ArgumentParser(description='Read-only independent exact certificate and archive verification')
    parser.add_argument('--results',type=Path,default=HERE/'recorded')
    parser.add_argument('--inputs',type=Path,default=HERE/'inputs')
    parser.add_argument('--output',type=Path,help='Optional new directory for verification receipts')
    args=parser.parse_args();run(args.results,args.inputs,args.output)
