# Direct aggregate learning-bias validation

This document proves the bound used by the aggregate validator. The two
forecast tasks were already inspected when this construction was developed.

## Information and cost contract

Let the declared categories have **known exact** probabilities p_c. Independent
training fixes f_c and g_c in [0,1]. A validation unit is an independent pair of
raw observations (O,O'), and supplies A = a(V,V') - f_C and
B = a(W,W') - g_C, where C is the focal category and a is the midrank comparison.
Then E[A|C=c]=a_c=m_V(c)-f_c and E[B|C=c]=b_c=m_W(c)-g_c. A and B within the same
unit may be dependent. The inferential target is the single aggregate

    b = sum_c p_c a_c b_c.

Before inspecting values, split validation units into two independent streams.
Use A from the first and B from the second. Conditional on all focal category
assignments, the category averages a_hat_c and b_hat_c are independent across
streams and categories, with counts n_Ac,n_Bc. This does not require observing
population ranks or learning their unknown conditional means correctly.
The reference observations O' and their category labels are not conditioned
on: every reference retains its independent common marginal law.

With n validation pairs the raw cost is 2n rows, identical to the original
pair-based validator: splitting discards some channel information, not rows.
Training and independent evaluation remain separate costs. Known category
probabilities require a declared design or complete control metadata; replacing
them by sample frequencies without another uncertainty argument is invalid.
For a fixed archive whose entire control column is available, exact p_c is
available. This information advantage must also be offered to strong baselines.

## Finite-sample upper bound

First suppose all positive-mass categories occur in both streams. Let

    b_hat = sum_c p_c a_hat_c b_hat_c,
    x = log(3/delta),
    r_A = sqrt{ (x/2) sum_c p_c^2 b_hat_c^2 / n_Ac },
    r_B = sqrt{ (x/2) sum_c p_c^2 a_hat_c^2 / n_Bc },
    d_c = p_c / (4 sqrt(n_Ac n_Bc)),
    r_2 = sqrt{2x sum_c d_c^2} + x max_c d_c.

Then, conditional on independent training and on all focal validation category
assignments,

    P{ b > b_hat + r_A + r_B + r_2 } <= delta.

No simultaneous confidence rectangle for 2C means is constructed. The radius
is fully observable; it does not substitute an estimated mean into an unknown
variance while pretending that mean is exact.

For a category missing in either stream, omit it from the sums and add
p_c times a deterministic upper bound on a_c b_c. The maximum of the four
products at the rectangle [-f_c,1-f_c] times [-g_c,1-g_c] is valid. The set of
omitted categories is fixed after conditioning on the focal assignments, so no extra
failure probability is spent. A global bound can always be intersected with
any other deterministic bound. Intersecting with another random bound needs
the appropriate joint error budget.

## Proof

Write e_Ac=a_hat_c-a_c and e_Bc=b_hat_c-b_c. Each e_Ac is centered and
sub-Gaussian with variance proxy 1/(4n_Ac), by Hoeffding's lemma for a mean of
independent range-one variables; similarly for e_Bc. The identity is

    b - b_hat = -sum p_c b_hat_c e_Ac
                -sum p_c a_hat_c e_Bc + sum p_c e_Ac e_Bc.

Conditional on the B stream, the first sum exceeds r_A with probability at
most exp(-x). Conditional on the A stream, the second exceeds r_B with that
same bound. The conditional events need not be independent: a union bound
will be used.

For independent centered sub-Gaussian X,Y with proxies s_X^2,s_Y^2, first
conditioning on X and then integrating a standard normal auxiliary variable
gives

    E exp(tXY) <= (1 - t^2 s_X^2 s_Y^2)^(-1/2)

for |t|s_Xs_Y<1. Thus, for R=sum p_c e_Ac e_Bc, independence across categories
and -log(1-u^2) <= u^2/(1-|u|) imply

    log E exp(tR) <= t^2 sum d_c^2 / {2(1-t max d_c)}.

This is the standard sub-gamma MGF inequality. Chernoff optimization gives
P{R > sqrt(2x sum d_c^2)+x max d_c} <= exp(-x).
The three one-sided tail bounds prove the claim. Conditioning can then be
removed. Ties cause no change because the comparisons still lie in [0,1].

## Near-correct efficiency, with its precise scope

If both nuisance fits are exact, a_c=b_c=0. For fixed balanced categories
p_c=1/C and fixed balanced counts n_Ac=n_Bc=m/C, the expected total radius is
at most

    [3 sqrt(C x / 8) + x/4] / m.

Indeed E[a_hat_c^2] <= 1/(4n_Ac), likewise for b_hat_c, and Jensen's inequality
bounds each linear radius by sqrt(C x/8)/m. The product radius is exactly
sqrt(C x/8)/m + x/(4m).

This is O((sqrt(C log(1/delta))+log(1/delta))/m). It is an expected-radius
statement at the exact-fit point, not a high-probability width or power claim.
For small persistent errors, the linear terms additionally scale with their
weighted magnitudes. For unknown category masses, rare categories, temporal
dependence, or reused validation references, the preceding bound cannot be
silently transferred. The costs and query model remain essential.

## Connection to an unbiased second-order estimator

For iid validation units T=(C,A,B) and known masses, the symmetric kernel

    h(T_i,T_j) = 1{C_i=C_j}(A_i B_j + A_j B_i)/(2p_Ci)

has expectation b. Its projection is q(T)=(A b_C+B a_C)/2, and

    Var(U_n) = 4(n-2)Var(q)/[n(n-1)] + 2Var(h)/[n(n-1)].

Writing Q_Ac=E[A^2|c], Q_Bc=E[B^2|c], Q_ABc=E[AB|c],

    E[h^2] = (1/2) sum_c {Q_Ac Q_Bc + Q_ABc^2} <= C.

At exact fits Var(q)=0. Moreover each centered range-one variable has second
moment at most 1/4, so E[h^2]<=C/16 and Var(U_n)<=C/[8n(n-1)]. The coarser
bound 2C/[n(n-1)] follows without this last sharpening. A fully averaged U estimator
can use more information than the split estimator. Applying a generic bounded
kernel Hoeffding radius, however, loses this degenerate rate. The split bound
above is one elementary observable finite-sample solution, not a proof of
optimality of either estimator.

## Novelty assessment

The projection kernel 1{C_i=C_j}/p_Ci is a finite-category higher-order
influence-function kernel. Direct second-order bias correction, degeneracy
when nuisance errors vanish, and variance terms C/n^2 already have clear
prior ownership. Relevant primary sources include:

- Robins et al., *Higher order estimating equations for high-dimensional
  models*, Annals of Statistics (2017),
  https://pmc.ncbi.nlm.nih.gov/articles/PMC6453538/ .
- Liu, Mukherjee and Robins, *Assumption-lean falsification tests of rate
  double-robustness of double-machine-learning estimators* (2024),
  https://arxiv.org/abs/2306.10590 .
- Kim et al., *Semi-Supervised U-statistics* (2025),
  https://arxiv.org/abs/2402.18921 .

The possible paper increment is a carefully attributed, explicit finite-rank
validation bound under an exact-metadata contract. It must be compared against
known-forecast and direct-covariance routes under the same metadata access.
Neither the algebra nor changing the tail inequality warrants a claim of a
general new inference paradigm. No claim of first publication has been verified.
