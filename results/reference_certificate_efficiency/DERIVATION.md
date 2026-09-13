# Signed bias and variance-sensitive reference certificates

This derivation gives the signed learning allowance, finite variance bound,
probability inversion and sufficient sample-cost conditions used by the code.

## Statistical scope

The signed categorical bound replaces the absolute bias allowance. It is
never larger on the **same** confidence rectangles and probability caps. For the evaluation term, a valid, inexpensive comparator
retains the full order-three U-statistic and measures kernel variability on a
fixed set of disjoint symmetrized triples. Its proof combines a confidence
bound on the kernel standard deviation with the classical permutation proof
of U-statistic concentration. It does not substitute the variance of
dependent row scores into an iid theorem.

The useful increment is an observable, one-sided composition for this target,
with explicit costs and a cheap variance estimate. It is neither a new general empirical Bernstein inequality nor a minimax
detection boundary, optimal allocation rule or certificate of raw forecasting
gain. Full-U concentration and variance-sensitive finite bounds have close
classical predecessors.

## 1. Setup and validation event

Condition on independent training; the category set of size C and clipped
fits f_c,g_c are then fixed. The validation sample has m independent disjoint
focal/reference pairs. In each occupied focal category, the two comparison
means estimate m_V(c),m_W(c), using half credit for ties. For a fixed
0 < delta < 1 use

    r_c = sqrt(log(8 C / delta)/(2 n_c)),
    I_Vc = [max(0,av_c-r_c), min(1,av_c+r_c)],
    I_Wc = [max(0,aw_c-r_c), min(1,aw_c+r_c)].

An empty category receives both intervals [0,1]. Conditional on the focal
categories, two two-sided Hoeffding events per category cost at most
delta/(2C). Their simultaneous failure probability is at most delta/2.
The comparison coordinates within a pair need not be independent.

For each category, let u_c be the one-sided Clopper–Pearson upper limit with
individual error delta/(2C): beta.isf(delta/(2C), n_c+1, m-n_c) when n_c<m,
and 1 when n_c=m. When m=0, all caps equal 1. Define

    P = {p: 0 <= p_c <= u_c, sum_c p_c = 1}.

The simultaneous probability-cap event fails with probability at most
delta/2. The event E consisting of both rectangle coverage and cap coverage
therefore has probability at least 1-delta; no independence between these
events is assumed. It is the absolute-bound event without any extra allowance.
Zero-probability categories can have any chosen conditional means in [0,1],
since their true contributions vanish. Counts still follow the fixed
multinomial law. The empirical proportions are feasible because CP upper
limits exceed the empirical proportions, so P is nonempty even off E.

## 2. Signed bias LP, dominance, and edge cases

For a rectangle [l_V,u_V] x [l_W,u_W], form all four products

    (l_V-f)(l_W-g), (l_V-f)(u_W-g),
    (u_V-f)(l_W-g), (u_V-f)(u_W-g).

Let qplus_c and qminus_c be their maximum and minimum. A bilinear function
is affine in either coordinate when the other is held fixed. Taking an
endpoint in each coordinate proves that these four products give its exact
extrema on the rectangle, including degenerate intervals and endpoints.

Define

    Bplus  = max_{p in P} sum_c p_c qplus_c,
    Bminus = min_{p in P} sum_c p_c qminus_c.

On E the true probability vector is feasible and each true product is in
[qminus_c,qplus_c]. Therefore

    Bminus <= b_H = sum_c p_c (m_V(c)-f_c)(m_W(c)-g_c) <= Bplus.

Both programs are solved by assigning probability mass to sorted cell
coefficients until total mass is exactly one: descending for the maximum,
ascending for the minimum. This is the fractional-knapsack exchange proof:
if a lower-valued cell has positive mass while a higher-valued cell has
unused capacity, transferring mass increases the objective. **Even negative
coefficients must receive mass when needed to achieve sum p=1.** An optimizer
that stops when coefficients become negative is wrong.

Let e_Vc=max(|l_Vc-f_c|,|u_Vc-f_c|), with e_Wc analogous. For every cell,
qplus_c <= e_Vc e_Wc. Maximizing over the identical P proves pointwise

    Bplus <= Babs := max_{p in P} sum_c p_c e_Vc e_Wc.

Thus replacing Babs by Bplus increases the lower bound by exactly
Babs-Bplus, while preserving its failure probability. This dominance holds
with the same data, delta, rectangles, and caps; changing those quantities
does not support an unconditional numerical dominance comparison.

Bplus may be negative. For one cell with f=.8, g=.2 and both intervals
[.4,.6], the four products are [-.08,-.16,-.04,-.08]; Bplus=-.04,
Bminus=-.16, and Babs=.16. The lower bound legitimately adds .04 instead of
subtracting .16. Clamping Bplus at zero stays valid but discards this gain.
The quantity is a signed upper bias limit, not an absolute error norm.

### A permanent detection loss from discarding the sign

The sign improvement can persist even with unlimited validation and
evaluation. For one declared category take Bernoulli V,W with
P(V=W=1)=P(V=W=0)=.29 and P(V=1,W=0)=P(V=0,W=1)=.21. Both marginal
success probabilities are .5, so their marginal midranks are .25 and .75,
both conditional means are .5, and

    theta = .25 Cov(V,W) = .25(.29-.25)=.01.

Use fixed allowed fits f=.65 and g=.35. Their bias is
b=(.5-.65)(.5-.35)=-.0225 and the corrected full U mean converges to
theta+b=-.0125. As validation size grows, Babs converges to .0225 but
Bplus converges to -.0225. Thus as evaluation also grows,

    absolute-certificate lower bound -> theta+b-|b|=-.035,
    signed-certificate lower bound   -> theta+b-b=.01.

More generally, for fixed opposite-sign errors and b<0 the absolute
certificate fails to detect every 0<theta<2|b| asymptotically, whereas
the signed certificate detects any fixed theta>0 when its validation
limits shrink. This statement conditions on persistent fixed fits; it
does not apply to a sequence of consistent fits with b tending to zero.
It isolates unnecessary sign loss, not a limitation of classical
inference supplied with the same signed bias information. A classical
signed bias correction can exploit that information as well.

The numerical receipt also gives a finite arithmetic example using
one million validation pairs and one million effective triples, evaluated
at the population evaluation mean. This is a worked certificate
calculation, not a reported random draw or a finite power estimate.

Unobserved cells, exact zero effects, zero observed variance, f/g equal to
0 or 1, ties, C=1, and validation size zero all retain valid definitions.
An all-empty validation sample generally gives a weak certificate; it
cannot justify dropping declared cells. Numerical code should reject
nonfinite inputs and infeasible arbitrary caps. For a nonempty clipped fit
vector, the rectangle kernel width R is positive: each category already
has width at least 1/2. A generic zero-range branch therefore is unnecessary
for this specific kernel, though a general bounded-kernel API may need it.

## 3. Full U-statistic and the independent triple score

For three independent observations define the symmetric kernel

    h(o1,o2,o3) = (1/6) sum_{(i,j,k) a permutation of (1,2,3)}
                    (a(V_i,V_j)-f(Z_i))(a(W_i,W_k)-g(Z_i)).

Every term uses different forecast and outcome reference observations.
The group mean of the corrected reference row scores is exactly the order-three
U-statistic with kernel h. Conditional on training,

    E h = mu = theta+b_H,   Var(h)=sigma^2.

Let [ell,u] be the fitted rectangle kernel range, R=u-ell. Averaging does not
leave that range, so h lies in [ell,u]. For M independent common-law groups
of N rows, write J=M floor(N/3). In each group take the first 3 floor(N/3)
rows in **a fixed ordering independent of their values**, split them into
triples, and compute all six permutations for every triple. The resulting
X_1,...,X_J are iid conditional on training, with law h. Let

    s^2 = sum_j (X_j-Xbar)^2/(J-1),  J>=2.

This is the variance of the kernel, not the group mean and not the row
product. The six terms in one triple are dependent and yield **one**
observation for the variance computation. Remainder rows may remain in the
full U-statistic, even though the variance estimate omits them. Fixed
input order is valid under the declared iid design; sorting by observed
scores before forming triples is not.

## 4. Finite variance-sensitive bound retaining the full U-statistic

Fix a test level a with delta<a<1 and write

    x = log(2/(a-delta)),
    A = sqrt(2 s^2/J),
    C_J = 2 R/sqrt(J(J-1)) + R/(3J),
    r_B(a) = A sqrt(x) + C_J x,
    r_H,half(a) = R sqrt(x/(2J)),
    r_U(a) = min(r_B(a), r_H,half(a)).

Then

    L_U,a = Ubar - Bplus - r_U(a)
    satisfies P(L_U,a > theta) <= a.

Here Ubar is the original full within-group U-statistic average. The bound
does not need additional evaluation observations. The fixed triple scores
can be calculated in linear time after the full scores are available.

### Proof of the evaluation bound

Condition on training and validation, so mu and sigma are fixed. Theorem 10
of Maurer–Pontil, applied after scaling X_j to [0,1], gives

    P(sigma > s + R sqrt(2x/(J-1))) <= exp(-x).                 (1)

For the full U-statistic, take a uniform random permutation in each group,
form disjoint blocks of three, and let T_pi be the mean of the resulting
J kernel values. Every unordered triple appears with the same probability,
so Ubar=E_pi[T_pi | evaluation rows]. Exponential convexity gives

    E exp(lambda(Ubar-mu)) <= E exp(lambda(T_pi-mu)).

For each fixed permutation T_pi is a mean of J independent bounded kernel
variables with variance sigma^2. A centered variable Y of width R first
satisfies the Bennett mgf bound, for every lambda>0,

    log E exp(lambda Y)
      <= (sigma^2/R^2) [exp(lambda R)-1-lambda R].

Indeed |E Y^k|<=sigma^2 R^(k-2) for k>=2, so expand the exponential
and apply log(1+v)<=v. Independence and exponential convexity transfer
this bound to the full U-statistic:

    log E exp(lambda(Ubar-mu))
      <= (J sigma^2/R^2) [exp(lambda R/J)-1-lambda R/J].

The inequality k!>=2*3^(k-2) implies
exp(u)-1-u<=u^2/[2(1-u/3)] for 0<=u<3. Thus the centered variable
also satisfies, for 0<lambda<3/R,

    log E exp(lambda Y) <= sigma^2 lambda^2/[2(1-R lambda/3)].

Scaling the mean and applying exponential convexity gives

    log E exp(lambda(Ubar-mu))
      <= sigma^2 lambda^2/[2J(1-R lambda/(3J))].

Here is the exact Chernoff algebra, to disambiguate the linear constant.
For v=sigma^2/J>0 and c=R/(3J), put t=sqrt(2x/v) and
lambda=t/(1+ct), which belongs to (0,1/c). If
b=sqrt(2vx)+cx=vt+cvt^2/2, then

    lambda b - v lambda^2/[2(1-c lambda)]
      = [v t^2(1+ct/2)-v t^2/2]/(1+ct)
      = v t^2/2 = x.

Markov's inequality therefore gives

    P(Ubar-mu > sigma sqrt(2x/J)+R x/(3J)) <= exp(-x).         (2)

The rational **mgf** already implies the linear term cx, not 2cx.
The looser rational Bernstein **tail** exp[-b^2/{2(v+cb)}] is also
a consequence, but inverting that already-weakened tail can instead
produce sqrt(2vx)+2cx. The present proof uses the mgf directly and does
not make that intermediate loss. No theorem constant changes are needed.

For a second analytic verification, let h(z)=(1+z)log(1+z)-z for z>=0.
Its convex dual is exp(lambda)-1-lambda. Set z=sqrt(2u)+u/3 and
lambda=sqrt(2u)/[1+sqrt(2u)/3]. The exponential-series inequality above
and the exact algebra with v=1,c=1/3 give

    h(z) = sup_{lambda>=0} {lambda z-[exp(lambda)-1-lambda]}
         >= lambda z-lambda^2/[2(1-lambda/3)] = u.

Since h is increasing, this proves h^{-1}(u)<=sqrt(2u)+u/3. Thus the
exact Bennett mgf gives the same claimed threshold by inversion as well.

If sigma=0 the kernel and U-statistic are almost surely constant and (2)
holds directly. The same argument applies to the lower tail by negation.
Hoeffding's bounded-variable mgf also transfers through the same convexity
step and gives

    P(Ubar-mu > R sqrt(x/(2J))) <= exp(-x).                   (3)

The smaller of the two *true-parameter* thresholds in (2) and (3) is fixed
conditional on training/validation. Its failure probability is therefore
at most exp(-x), without a union bound between (2) and (3). On event (1),
r_B(a) is at least the threshold in (2), hence r_U(a) is at least their
minimum. The probability that Ubar-mu exceeds r_U(a) is consequently at
most 2 exp(-x)=a-delta. This union step permits the same rows to determine
both s and Ubar. Union with E^c costs delta, proving the claim.

An equivalent intermediate expression uses

    sigma_up = min(R/2, s + R sqrt(2x/(J-1)))

and minimizes its Bernstein radius with the Hoeffding radius. The clipping
by R/2 has no effect after this minimum because that clipped Bernstein
branch is already above the Hoeffding branch. The simple formula above is
therefore exactly this valid construction, not an approximation.

### Inverted finite p-value

Let d=Ubar-Bplus for testing theta<=0. If d<=0 return p=1. Otherwise set

    x_H = 2J(d/R)^2,
    z_B = 2d/[A+sqrt(A^2+4 C_J d)],
    x_B = z_B^2,
    p_U = min(1, delta+2 exp(-max(x_H,x_B))).

This is the inversion of the nested rejection events d>r_U(a). The rational
root formula avoids cancellation. At every a in (delta,1), rejection implies
the corresponding finite confidence failure under theta<=0, proving
super-uniformity. Thresholds at or below delta never reject. The same
formulas test theta<=theta0 by replacing d with Ubar-Bplus-theta0.

The original signed-Hoeffding p-value remains

    p_H = min(1, delta+exp(-2J(max(d,0)/R)^2)).

The new p-value can be larger when the variance branch does not help,
because its validation/variance/tail decomposition uses a factor 2. Do not
select min(p_H,p_U) after looking at the evaluation data without correction.
Valid options are a fixed method, separate reporting, or a prespecified
Bonferroni combination min(1,2 min(p_H,p_U)).

## 5. Classical independent-triple comparator and attribution

The immediate independent-triple comparator uses Xbar, the same Bplus, and

    r_MP(a)=sqrt(2 s^2 log(2/(a-delta))/J)
            +7R log(2/(a-delta))/(3(J-1)).

It follows directly from the one-sided empirical Bernstein inequality in
[Maurer and Pontil (2009), Theorem 4, p. 2](https://www.cs.mcgill.ca/~colt2009/papers/012.pdf),
using the reflected variable for the upper deviation. The standard
deviation confidence statement used above is their Theorem 10, p. 4,
and the known-variance mean inequality with linear coefficient 1/3 is their
Theorem 3, p. 2. Our unsimplified C_J is no larger than 7R/[3(J-1)].
The full-U mean is the average of all randomized block means and has no
larger conditional variance than a single fixed block mean, but neither
realized lower bound uniformly dominates the other because their centers
differ. The triple partition can affect realized variance estimates.

[Peel, Anthoine and Ralaivola (2010), Theorem 3](https://papers.neurips.cc/paper_files/paper/2010/file/d6ef5f7fa914c19931a55bb262ec879c-Paper.pdf)
is the nearest direct prior art: empirical Bernstein inequalities for
bounded symmetric U-statistics, with variance estimators that themselves
are higher-order U-statistics. Their full-U method and our inexpensive
fixed-block variance estimate are different implementations of existing
variance-sensitive concentration machinery. Do not claim that their
theorem is implemented by an ordinary variance of corrected row scores.
No claim of superiority to their full variance estimators is supported
without a matched implementation and experiment. The present construction
avoids computing an order-six variance U-statistic, at the cost of a
partition-dependent, potentially less efficient variance estimate.

## 6. Family-aware failure allowance fixed before data

For a fixed K-hypothesis BY family and desired FDR level q, let

    H_K = sum_{k=1}^K 1/k, t_1=q/(K H_K).

The certificate p-values have infimum delta, so an isolated signal cannot
pass BY rank one when delta>=t_1. With delta=.0001, K=100, q=.05,
t_1 is about 0.0000963878; rank one is inaccessible, but rank two is not
blocked by that floor alone. It is incorrect to claim that no rejection
at any rank is possible.

A simple prespecified rule is

    delta = rho q/(K H_K),  0<rho<1, e.g. rho=.1.

This reserves fraction rho of the first-rank threshold for validation and
leaves (1-rho)t_1 for evaluation. The choice is principled as a resolution
constraint, not a proof of optimal power. Reducing rho tightens the floor
but widens the validation rectangles and probability caps. K, q, rho,
categories, method, and fixed partition rule should be frozen before the
validation/evaluation data. K may be a conservative declared upper bound
if fewer hypotheses are eventually tested, with the testing protocol
declared in advance. Do not select delta to maximize the realized lower
bound. A finite prespecified tuning grid is possible only with its own
simultaneous error correction or independent selection data.

BY uses each hypothesis's unconditional super-uniform p-value; there is no
extra requirement that the K validation events jointly have probability
1-delta. Arbitrary dependence between models is allowed by BY. The common
validation/evaluation design must still establish each marginal guarantee.

## 7. Detectable effects and observation cost

These are **sufficient** conditions for this certificate, not lower bounds
on what any test can do. On E define the observable bias uncertainty width

    W_B=Bplus-Bminus, so Bplus-b_H <= W_B.

After validation, for an independent evaluation sample and any beta in
(0,1), Hoeffding's lower-tail event holds with probability at least 1-beta.
Thus the original signed-Hoeffding certificate is positive with conditional
probability at least 1-beta whenever

    theta > W_B + R/sqrt(2J) *
                 [sqrt(log(1/(a-delta)))+sqrt(log(1/beta))].  (4)

For a declared minimum effect Delta>W_B, it suffices to choose

    J > R^2 [sqrt(log(1/(a-delta)))+sqrt(log(1/beta))]^2
             /[2(Delta-W_B)^2].                             (5)

This condition is computable from validation and a declared Delta. Selecting
the evaluation sample size from validation is permitted if evaluation is
fresh and its size is fixed before observing its values. If Delta<=W_B,
this particular sufficient condition cannot certify adequate power at any
evaluation size. That does not prove impossibility: the actual bias slack
Bplus-b_H may be substantially smaller. Unconditional statements must also
account for E^c, with probability at most delta.

For the full-U variance route let sigma be the true conditional kernel
standard deviation, x=log(2/(a-delta)), and y=log(2/beta). The upper SD
inequality and lower Bernstein tail, each with error beta/2, imply power
at least 1-beta conditional on E if

    theta > W_B
       + sigma sqrt(2/J) (sqrt(x)+sqrt(y))
       + R {2[sqrt(x y)+x]/sqrt(J(J-1)) + (x+y)/(3J)}.        (6)

To see this, bound s above by sigma+R sqrt(2y/(J-1)), insert it in r_B,
and add sigma sqrt(2y/J)+Ry/(3J) for the lower evaluation fluctuation.
The minimum with Hoeffding can only improve the actual radius.
Equation (6) isolates O(sigma/sqrt(J)+R/J) evaluation cost at fixed error
levels. In particular zero true kernel variance leaves only an O(R/J)
evaluation term; zero *observed* variance still leaves a nonzero linear
allowance. A comparison of actual effect sizes, validation uncertainty,
and constants is necessary before claiming practical power improvements.

If a known or independently certified sigma0>=sigma is available, write
g=Delta-W_B>0,

    A0=sigma0 sqrt(2)(sqrt(x)+sqrt(y)),
    D0=R {2 sqrt(2)[sqrt(xy)+x]+(x+y)/3}.

Because sqrt(J(J-1))>=J/sqrt(2) for J>=2, a simple sufficient choice is

    J > max(2, (2A0/g)^2, 2D0/g).                           (7)

The universal choice sigma0=R/2 is valid but may erase variance gains.
Plugging an uncorrected pilot or evaluation standard deviation into (7)
does not yield its stated prospective guarantee. An independently
certified pilot upper limit carries its own confidence failure and sample
cost. Formula (6) may be shown with true sigma as a diagnostic in a
simulation, labeled as a theoretical power condition rather than an
observable input to the audit.

Total observation cost is m_train+2m_val+MN. For a fixed group size N>=3,
J_req blocks can be realized with M=ceil(J_req/floor(N/3)); all MN rows
must be counted. This yields an evaluation cost close to 3J_req plus
rounding/remainder overhead. These formulas offer no law-free prescription
for m_train: without a learning-rate assumption, no training budget can
guarantee small fitted-mean error. Rare-category caps and the measured W_B
make that limitation observable rather than hiding it in a rate claim.

## 8. Executable verification and comparison

From the package root, `python reproduce_artifacts.py --output /tmp/forecast-audit-check --skip-figures` includes the independent certificate verifier. It recomputes all 500,000 inference rows and 500 cell summaries, and reconstructs eight complete primitive replications with generic rank comparisons and independent linear programs. The public unit tests additionally check ties, negative allowances, probability inversion, family resolution and misuse of the independent-triple mean. Numerical checks support the implementation; the proofs above supply validity for the stated class of laws.

The paired study compares absolute range, signed range, signed full-U variance, classical independent-triple empirical Bernstein and pooled full-U variance at identical training, validation and evaluation costs. Its tables expose signed versus absolute allowances, sampling radii, coverage, power and remaining validation uncertainty. The classical variance-sensitive comparators largely share the power gains over the original range-only construction.
