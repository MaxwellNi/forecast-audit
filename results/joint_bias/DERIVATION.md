# Joint rank-mean constraints for a signed categorical bias allowance

This is a bounded, post-exposure research increment. McCormick relaxation, perspective reformulation, linear programming, weak duality, concentration inequalities and multiple testing are classical. The additional information used here is the exact identity that **each population marginal midrank has mean one half**, including ties. We do not claim a sharp attainable bias bound, a new general optimization method, or a new concentration inequality.

## Statistical contract

Condition on independent training information H. There is one common law P of (X,Y,C), where C takes values in a declared finite set of m categories. The marginal rank transforms, conditional on H when necessary, are

R_X(x)=P(X'<x | H)+P(X'=x | H)/2,

and similarly R_Y. The independent reference (X',Y',C') has this same law; it is not restricted to the focal category. Let p_c=P(C=c | H), µ_Xc=E[R_X(X) | C=c,H], µ_Yc analogously, and training fits f_c,g_c in [0,1]. For zero-mass categories the means can be assigned arbitrarily in [0,1]. The target is

θ=E[(R_X−µ_XC)(R_Y−µ_YC) | H].

The order-three distinct-reference kernel has expectation θ+b, where

b=Σ_c p_c(µ_Xc−f_c)(µ_Yc−g_c).

This identity does not require conditional independence of X and Y. The present result changes only the upper allowance for b. It does not turn a time series into independent observations, change the categorical target into a richer-control target, or establish forecasting loss improvement.

Suppose an independent validation sample gives a simultaneous confidence event E with conditional probability at least 1−δ:

ℓ_c≤p_c≤h_c; L_Xc≤µ_Xc≤U_Xc; L_Yc≤µ_Yc≤U_Yc for every c.

Endpoints must belong to [0,1] and lower endpoints must not exceed upper endpoints. Their construction can be the existing split-δ Hoeffding/Clopper–Pearson construction. No independence between its count and mean confidence events is needed: a uniform union bound suffices. The following result is conditional on valid confidence inputs; certification of the LP arithmetic does not itself certify a floating-point implementation of a binomial quantile.

## Exact rank identities and the relaxation

For independent copies, a(x,x')+a(x',x)=1 pointwise, including ties. Exchangeability therefore gives E R_X=E R_Y=1/2. Consequently

Σp_c=1,  Σp_cµ_Xc=Σp_cµ_Yc=1/2.

These are deterministic properties of the target law, so using them spends no additional validation error probability.

Introduce p_c,u_c,v_c,w_c, intending u_c=p_cµ_Xc, v_c=p_cµ_Yc, w_c=p_cµ_Xcµ_Yc. Impose the following **linear outer relaxation**, in addition to 0≤p_c,u_c,v_c,w_c≤1:

ℓ_c≤p_c≤h_c,

L_Xc p_c≤u_c≤U_Xc p_c,   L_Yc p_c≤v_c≤U_Yc p_c,

w_c≥L_Yc u_c+L_Xc v_c−L_Xc L_Yc p_c,

w_c≥U_Yc u_c+U_Xc v_c−U_Xc U_Yc p_c,

w_c≤L_Yc u_c+U_Xc v_c−U_Xc L_Yc p_c,

w_c≤U_Yc u_c+L_Xc v_c−L_Xc U_Yc p_c,

Σp_c=1,  Σu_c=Σv_c=1/2.

Let B_MC be the maximum of Σ(w_c−g_c u_c−f_c v_c+f_c g_c p_c) over this set. The probability bounds may include lower bounds, although the archived comparison uses the old one-sided upper bounds and lower bounds zero. Empty categories are retained.

**Proposition (joint upper allowance).** On E the relaxation is nonempty and

b ≤ B_MC ≤ B_sep,

where B_sep=max{Σp_c q_c : ℓ≤p≤h, Σp=1} and

q_c=max_{x∈{L_Xc,U_Xc}, y∈{L_Yc,U_Yc}}(x−f_c)(y−g_c).

The right comparison uses exactly the same rectangles and probability box. Thus it isolates the additional rank identities rather than a different allocation of δ or different validation observations.

**Proof.** On E the intended values (p,pµ_X,pµ_Y,pµ_Xµ_Y) satisfy every displayed constraint; the four inequalities follow by expanding nonnegative products (µ_X−L_X)(µ_Y−L_Y), (U_X−µ_X)(U_Y−µ_Y), (U_X−µ_X)(µ_Y−L_Y), and (µ_X−L_X)(U_Y−µ_Y). This proves b≤B_MC. For p_c>0, divide the four inequalities and mean bounds by p_c. They describe the classical convex hull of (x,y,xy) over the corresponding rectangle. A linear objective on that hull is at most its maximum over the four corners, q_c. Multiply by p_c and sum. When p_c=0 the constraints force u_c=v_c=w_c=0, so this term is zero. Maximizing over the same probability box proves B_MC≤B_sep. Degenerate rectangles follow by the same inequalities, or by continuity. □

Without the two rank-mean equalities the LP equals B_sep: choose a maximizing rectangle corner in each category and a maximizing probability vector. The new constraints can produce a strict improvement, but they need not do so.

## Finite probability implication

Let U be the same fixed order-three distinct-reference statistic already used in the manuscript and let r_ε be any justified one-sided sampling radius satisfying

P{U−(θ+b)>r_ε | H,validation}≤ε

for every realized training/validation sample, under its declared evaluation design. This statement includes a correctly proved data-dependent variance radius; it does not follow by inserting an observed variance into Hoeffding's inequality.

For δ<α and any measurable returned allowance B satisfying b≤B on E, the lower bound

L_α=U−B−r_(α−δ)

satisfies P{L_α>θ | H}≤α. The proof is the union bound E^c plus the conditional sampling event. Hence the joint bound can replace B_sep in the existing finite theorem without changing its sampling assumptions or constants. Conditional null θ≤0 is sufficient; conditional independence is not required. A correctly constructed fixed-mixture betting p-value can use the same allowance by conditioning on training and validation, with its existing additive δ correction. There is no extra error charge for the exact mean identities.

The module below returns an upward-rounded, independently checkable upper allowance at most its own exactly computed, same-input B_sep. Thus the same finite guarantee applies when its input confidence event is valid. An observed infeasible confidence set is **not** a statistical rejection rule: fallback rules are specified below.

## Numerical upper certificate and conservative fallbacks

Write the relaxation as max cᵀx subject to Ax≤b₀, Ex=d, 0≤x≤1. A floating-point solver only proposes multipliers y≥0,z. Reconstruct all coefficients, input endpoints and multipliers as their exact binary rational values. Set r=c−Aᵀy−Eᵀz. For every feasible x,

cᵀx ≤ b₀ᵀy+dᵀz+Σ_i max(r_i,0) = U_dual.

This is exact weak duality with a box-residual correction. It does not require the proposed multipliers to satisfy a floating-point dual tolerance. The primal solver objective is a diagnostic only and is never used as an upper guarantee. The module computes min(U_dual,B_sep) in exact rational arithmetic, then rounds upward to binary64. Monotonicity of upward rounding gives a returned bound no greater than the upward-rounded B_sep. Each result stores its inputs, multipliers, exact residual correction and exact fraction strings, so a verifier needs no optimization call.

If the solver fails, returns nonfinite multipliers or reports that the joint system is infeasible, return the exact separable upper bound with a visible fallback status. If the probability box is itself empty, return the universal b≤1. On E neither genuine infeasibility can occur; the universal fallback makes an empty box harmless even when observed. Numerical infeasibility is never taken to prove that E failed. Malformed input (NaN, reversed endpoints, fits outside [0,1]) raises an error rather than silently constructing a guarantee. Near-empty sets do not receive a claimed primal optimum; the exact dual residual remains a valid upper bound or the algorithm falls back.

## Strict improvement and strict relaxation: exact examples

1. **Attainable strict improvement.** Let p=(1/2,1/2), f=g=(0,0), and identical X/Y mean rectangles [1/4,3/8] and [5/8,3/4]. Then B_sep=45/128, whereas B_MC=5/16. The true rank law X=Y=Z with Z uniform on [0,1] and C indicating the two half intervals has means (1/4,3/4), so it attains 5/16. To see the upper bound, set x₁+y₁=s, with x₂+y₂=2−s from the equalities. The two relevant upper McCormick faces, after cancellation, give total objective ≤5/16; the finite LP certificate also checks this exactly. This example establishes that the added information can matter, without asserting a generic power gain.

2. **Single-category strict relaxation.** Let p=1, f=g=1/2, and both mean intervals [0,1]. The true means are known to be exactly 1/2, so b=0. Yet the relaxed point u=v=w=1/2 is feasible and gives B_MC=1/4. The relaxation is therefore not the exact coherent-parameter optimization, even in this elementary case. Directly substituting known means would solve this special case, but we retain it as a limitation of the stated LP, rather than conceal it with a special rule.

3. **A nontrivial, rank-attainable three-category gap.** Let p=(1/4,3/8,3/8), f=g=1/2, and every mean interval [3/8,5/8]. The LP permits u=v=p/2 and w=p(1/4+1/64), so B_MC=1/64. In contrast, the maximum over coherent means satisfying the two weighted-mean equalities is 3/256. To prove this, write errors as (1/8)t with |t_c|≤1 and pᵀt=0. By Cauchy–Schwarz, the largest cross-product equals the largest weighted squared norm. That convex function reaches its maximum at a vertex of the sliced box. Enumerating which two coordinates are endpoints yields maximum 3/4 (the others are 2/3), achieved by t=(0,1,−1). Hence the maximum is (1/64)(3/4)=3/256. This coherent optimum is attained by a real rank law: let X=Y=Z uniform on [0,1] and

P(C=1|Z)=1/4,

P(C=2|Z)=3/8+(9/16)(Z−1/2),

P(C=3|Z)=3/8−(9/16)(Z−1/2).

All conditional probabilities are nonnegative and sum to one. They give the stated masses and means (1/2,5/8,3/8). The LP gap is 1/256.

There are additional constraints for means that arise from a *common rank law*: for example p_c/2≤µ_Xc≤1−p_c/2, and similarly Y. To prove this, split p_cµ_Xc into references in the same category (exact contribution p_c²/2 by exchangeability) and references outside it (between zero and p_c(1−p_c)). The proposed LP does not enforce these nonlinear constraints or all subset analogues. This is another reason not to call its optimum distributionally sharp. Adding further relaxations is outside this bounded construction.

## Classical sources and claim boundary

- G. P. McCormick (1976), *Computability of global solutions to factorable nonconvex programs: Part I: Convex underestimating problems*, Mathematical Programming 10, 147–175. [Publisher and DOI](https://link.springer.com/article/10.1007/BF01580665). The bilinear convexification principle is classical.
- H. Hijazi (2015), *Perspective Envelopes for Bilinear Functions*. [Author technical report](https://optimization-online.org/2015/03/4841/). Perspective formulations and stronger convexifications are existing optimization tools; this is not a priority claim about their use.
- N. Boland, S. Dey, T. Kalinowski, M. Molinaro and F. Rigterink, *Bounding the gap between the McCormick relaxation and the convex hull for bilinear functions*, Mathematical Programming 162 (2017), 523–535. [Primary manuscript](https://arxiv.org/abs/1507.08703). Linked bilinear relaxations can have gaps; the concrete examples above are derived here, not attributed to that paper.
- T. B. Armstrong and M. Kolesár (2018), *Optimal Inference in a Class of Regression Models*, Econometrica 86, 655–683. [Author manuscript](https://www.princeton.edu/~mkolesar/papers/optimal.pdf). Accounting for a worst-case bias in one-sided inference is established methodology. Its optimality results under specific function classes do not imply optimality here.

A defensible description is: “We strengthen the categorical learning allowance by coupling the two conditional rank-mean vectors through their exact marginal means. A perspective McCormick relaxation yields a computable conservative upper bound no larger than the same-input separable allowance.” The rank identity is elementary, and the optimization is classical. This is a precise additional inference constraint and an implementation result, not evidence of a major first theorem or broad minimax advantage.
