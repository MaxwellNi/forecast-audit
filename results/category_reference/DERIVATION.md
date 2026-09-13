# Computable categorical reference certificate

The paper combines an observable allowance for conditional-mean errors with
classical concentration for bounded U-statistics. This document explains both
the original maximum-cell allowance and the probability-weighted version.
The latter is Proposition 2 in the main paper. Neither construction gives a
guarantee for arbitrary observed rank panels or HAC inference.

Let the raw triples `(V,W,Z)` be iid from an unknown fixed distribution, with
`Z` in a declared finite set of `C` categories. Ties are allowed. The marginal
midrank transforms are `R_V=F_V^m(V)`, `R_W=F_W^m(W)`. The estimand is
`theta=E[(R_V-m_V(Z))(R_W-m_W(Z))]`. No conditional-independence assumption
is needed. The fits `f_c,g_c` lie in `[0,1]` and are fixed using an independent
training set. Evaluation consists of `M` independent groups of `N>=3` iid
observations from this same law.

Validation uses `L` independent disjoint pairs, each containing two new raw
observations from this law, independent of training and evaluation. The first
observation supplies `Z`, `V`, `W`; the second supplies reference values
`V'`, `W'`. Compute `A=c(V,V')`, `B=c(W,W')`, where
`c(v,v')=1{v'<v}+.5*1{v'=v}`. Conditional on the first member's category,
`E[A|Z=c]=m_V(c)` and `E[B|Z=c]=m_W(c)`. Independence between the two
channels is unnecessary. Independence between pairs is essential.

Let `n_c` be the number of validation focal rows in category `c`, and
`a_c,b_c` the two average comparisons. For `n_c>0`, set

`r_c = sqrt(log(4*C/delta)/(2*n_c))`,

`I_Vc = [a_c-r_c,a_c+r_c] intersect [0,1]`, and similarly `I_Wc`.
For an empty category use `[0,1]`. Conditional on all focal categories, each
occupied-cell Hoeffding bound fails with probability at most `delta/(2*C)`;
the union across both channels and all categories is at most `delta`.
Integrating over the random counts gives the same unconditional guarantee.

For endpoints `l,u`, let `e_Vc=max(|l-f_c|,|u-f_c|)` and similarly `e_Wc`.
Then the fully observable budget

`B_H=max_c e_Vc*e_Wc`

satisfies `P(|b_H|<=B_H)>=1-delta`, where
`b_H=E[(m_V-f)(m_W-g)]`. Unknown category probabilities are not inserted or
estimated: the weighted expectation is bounded by its largest cell term.
This protects rare or missing categories at a potentially substantial power
cost, made visible by the study. Random continuous bins certify the represented
category target only; they do not bound omitted continuous-control structure.

Conditional on training, the ordered-triple kernel
`h=(c(V_i,V_j)-f(Z_i))*(c(W_i,W_k)-g(Z_i))`, `i,j,k` all different,
has a computable range `[a,b]`: take the minimum and maximum of
`f_c*g_c`, `-f_c*(1-g_c)`, `-(1-f_c)*g_c`, `(1-f_c)*(1-g_c)` over categories.
Symmetrization preserves this range. Let `R=b-a` and
`J=M*floor(N/3)`. Averaging permutations of disjoint triples and exponential
convexity gives

`P(Dbar-E[Dbar|H] > t | H) <= exp(-2*J*t^2/R^2)`.

This is Hoeffding's classical bounded U-statistic argument. Since
`E[Dbar|H]=theta+b_H`, the lower bound

`L_alpha = Dbar-B_H-R*sqrt(log(1/(alpha-delta))/(2*J))`

has `P(L_alpha>theta)<=alpha` for `0<delta<alpha<1`. Conditioning also on
validation preserves the evaluation law. A union bound spends `delta` on
failure of the nuisance envelope and `alpha-delta` on the upper evaluation
tail. It is finite-sample and requires no positive asymptotic variance.

Equivalently, with fixed `delta`,

`p_cert=min(1, delta+exp(-2*J*max(Dbar-B_H,0)^2/R^2))`

is super-uniform for the entire null `theta<=0`. For thresholds `u<delta`
it never rejects; for `u>=delta`, the two failure budgets sum to `u`.
Consequently a fixed full family can apply ordinary BY to these p-values,
with arbitrary between-model dependence. Shared training and validation are
allowed if each marginal certificate has its declared validity. At large
family sizes `delta` itself limits resolution and must be fixed accordingly.
Choosing fits, groups, category definitions, validation budget or hypothesis
family after reading evaluation data requires a different argument.

The study fixes `delta=0.0001`, compares against the existing normal
studentization, reports all training and validation costs, and retains cases
where rare categories or poor fits make the certificate uninformative. It
does not establish that the bound is sharp, that the learning cost is minimax,
or that its power beats calibrated classical methods. The benefit demonstrated
is an actual computable guarantee under stated sampling conditions without
oracle nuisance functions or a supplied null law.

Primary concentration source: Wassily Hoeffding (1963),
[Probability Inequalities for Sums of Bounded Random Variables](https://doi.org/10.1080/01621459.1963.10500830).


## Frequency-weighted refinement

The preserved maximum-cell run was examined before this refinement. The new
run reuses its primitive seeds and is an exploratory paired follow-up.
Allocate half of delta to the conditional-mean intervals, replacing the
radius by sqrt(log(8*C/delta)/(2*n_c)). Allocate the other half to all
category probabilities: U_c is the one-sided Clopper-Pearson upper limit
with failure delta/(2*C), based on n_c out of L focal validation rows.
Explicitly, U_c=Beta^{-1}(1-delta/(2*C);n_c+1,L-n_c) if n_c<L,
and U_c=1 otherwise. At L=0 every U_c is one.

With probability at least1-delta the mean intervals and all p_c<=U_c hold
simultaneously. No independence between these events is required. Put
q_c=e_Vc*e_Wc, and maximize sum p_c*q_c over p>=0, sum p=1, p_c<=U_c.
Greedy allocation to decreasing q_c solves this linear program. The true
category distribution is feasible on the simultaneous event, which proves
the same bias envelope and certificate. Each U_c is at least n_c/L, so the
reported optimization remains feasible even off the confidence event.
Unseen cells are protected by their probability upper limits instead of
being assigned all the population mass.

Both runs have108 configurations and1,000 paired replications each. Training,
validation and evaluation cost50,176 observations at the largest setting.
Across the complete grids no nuisance-envelope or lower-bound coverage
failure occurs; this is empirical corroboration, not proof of zero risk.
At eight categories with rare probability0.005 and signal0.5 (target0.02731),
M400,N64,8,192 training observations and8,192 validation pairs, the mean
budget changes0.166423→0.008923 and detection0/1,000→1,000/1,000.
The moderate eight-category signals remain undetected at this sample size.

Because the refinement divides delta differently, it is not pointwise tighter
than the original maximum-cell bound. Its budget is larger on14,296 paired
rows;30 decisions favor only maximum-cell and3,182 favor only weighting.
These outcomes are retained in paired_budget_comparison.csv and the two full
replication tables. Bounds decrease relative to the maximum-cell construction
using the SAME new mean intervals, but that is a different comparator.

Primary probability-limit source: [Clopper and Pearson (1934)](https://doi.org/10.1093/biomet/26.4.404).
