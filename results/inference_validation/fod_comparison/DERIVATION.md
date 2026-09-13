# Classical FOD moment: prespecified comparison

This follow-up compares a standard forward-orthogonal-deviation (FOD) outcome moment with the three recorded raw-score fits. It is **not** a full implementation of the panel GMM estimator of Arellano and Bover, and no novelty is claimed for the transformation or the predictable-instrument argument. The actual primary paper defines the future-mean transformation and its variance factor in equations (24)-(25), pp. 41-42: [Arellano and Bover (1995)](https://www.cemfi.es/~arellano/arellano-bover-1995.pdf).

The new specification, normalization, primary comparisons and tests were fixed before this follow-up was executed. Earlier raw-family results had already been reviewed. This is therefore a retrospective methodological follow-up, not a fresh confirmation of an independently discovered advantage. Existing scripts, results, the frozen reviews and the manuscript remain unchanged.

## Data and evaluation rows

Use exactly the recorded generator and RNG streams. For entity g, model k and time t=1,...,20,

\[
Y_{gt}=a_g+\epsilon_{gt},\qquad
X^0_{gkt}=b_{gk}+0.2\left(5a_g+\sum_{h=1}^5\epsilon_{g,t-h}\right)
+\frac{\xi_{gt}+\zeta_{gkt}}{\sqrt2}.
\]

All innovations and intercepts are independent standard Gaussian variables except that the shared \(\xi\) induces forecast-noise correlation 0.5 across models; outcomes are shared across the model family. Five preceding innovations are supplied before the first observed time. Entity trajectories are independent. The alternatives are

\[
X_{gkt}=X^0_{gkt}+\tau\,1\{k\le3\}\epsilon_{gt},
\qquad \tau\in\{0,0.03,0.06,0.10\}.
\]

The current-innovation injections deliberately violate the null and are positive controls, not prospective forecasts. Every method evaluates **exactly t=5,...,16**, the 12 rows in the middle three of five contiguous folds. Earlier and later observed rows may be used for nuisance/instrument construction, as in the original comparison. The new FOD method uses more granular chronological nuisance sets than the blocked directional method; that design difference is the comparator being evaluated.

## Moment and target normalization

Let

\[
c_t=\sqrt{\frac{20-t}{21-t}},\quad
Z_{gkt}=X_{gkt}-\frac1{t-1}\sum_{s<t}X_{gks},\quad
\widetilde Y_{gt}=c_t\left(Y_{gt}-\frac1{20-t}\sum_{s>t}Y_{gs}\right).
\]

Define the equal-entity score

\[
S_{gk}^{FOD}=\frac{\sum_{t=5}^{16}Z_{gkt}\widetilde Y_{gt}}{\sum_{t=5}^{16}c_t}.
\]

The denominator is a positive design-only scalar. It retains the relative classical FOD factors and makes response to constant contemporaneous innovation covariance one. Dividing by 12 instead would multiply every entity score by the same constant and leave its studentized statistic unchanged. Dividing each individual row by \(c_t\) would change relative weights; that different method is not used here.

Under the null, the instrument is measurable using past outcomes, intercepts and forecast noises available at t. Outcome innovations at t or later have conditional mean zero against that information. Entity outcome effects cancel under future-mean subtraction. Thus

\[
E[Z_{gkt}\widetilde Y_{gt}]=0,\qquad E[S_{gk}^{FOD}]=0.
\]

This centering argument requires an appropriate innovation condition; arbitrary serial dependence does not satisfy it. Gaussianity is not required for the moment identity, but is part of the common simulation law.

For an affected model under the injected alternative,

\[
Z_{gkt}=Z^0_{gkt}+\tau\left(\epsilon_{gt}-\frac1{t-1}\sum_{s<t}\epsilon_{gs}\right).
\]

Independence and unit innovation variance give

\[
E[Z_{gkt}\widetilde Y_{gt}]=c_t\tau,
\quad E[S_{gk}^{FOD}]=\tau.
\]

The same zero and constant-alternative means hold for the original directional moment. If effects vary over time, the FOD mean is \(\sum c_t\tau_t/\sum c_t\), whereas the original equal-row directional mean is \(12^{-1}\sum\tau_t\). **The target match is exact for the specified constant-effect DGP; it does not assert target equivalence for arbitrary heterogeneous alternatives.**

For a score written \(X^\top A Y\), the original complementary and gapped scores have exact means

\[
E[S]=\sum_{t,u}A_{tu}\,0.2\,1\{1\le t-u\le5\}+\tau\operatorname{tr}(A)
=b_A+\kappa_A\tau.
\]

We retain their original matrices and statistics for exact replay. The analytic table reports both \(b_A\) and \(\kappa_A\), as well as the mean after division by \(\kappa_A\). This positive scaling matches constant-effect response but cannot remove their null bias. We do not treat nominal-reference power of biased methods as a calibrated comparison.

## Inference and comparisons

For each model and method, use \(\bar S/(\operatorname{sd}_{G-1}(S)/\sqrt G)\), with equal entity weights and the original numerical guard. Under independent entities, mean zero, bounded fourth moments and positive limiting variance, this has the usual asymptotic normal null limit. The alternative mean \(\tau>0\) gives consistency as G grows. It is not a finite-sample normal or t theorem for these non-Gaussian products.

Retain the original independent 5,000-draw calibration stream and 1,000-family evaluation stream, batch size 20, three entity counts (25,100,400), family sizes (10,11), all four signals, and normal, t(G−1), and independent supplied-null references. Each method receives its own statistic values on the **same** calibration primitives. Plus-one upper-tail ranks include ties. Every model stays in BY at 0.05. The primary comparison is G=400, K=11; other combinations are retained as sensitivities.

Calibration needs the correct simulator law as extra information. Rank validity averages over calibration-bank randomness; reported evaluation intervals condition on the realized banks. Calibrated methods have the same nominal level, not forced equal realized FDR. This does not validate a calibration procedure for unknown observational panels.

Power differences are computed per evaluation family, then averaged over 1,000 families. Correlated models are not treated as independent replications. Retain pointwise Monte Carlo intervals and, for the 27 primary contrasts (three signals × three old methods × three references), Bonferroni normal Monte Carlo intervals. They describe this fixed experiment and conditional bank, not broad performance guarantees. All outcomes, including a loss for FOD or directional fitting, will be reported.

## Verification before execution

Six independent checks pass: rowwise FOD products versus the matrix implementation; orthonormality of the full classical FOD transformation; exact null and alternative means and intercept cancellation; independently generated raw primitives including injected alternatives; invariance of studentization to the target-normalizing scalar; and calibration ties/BY against an independent library.

The complete replay additionally checks all original calibration and evaluation statistics against their frozen arrays, requires identical original BY decisions, and verifies that every original result hash remains unchanged. These checks address implementation fidelity; they do not broaden the statistical assumptions.
