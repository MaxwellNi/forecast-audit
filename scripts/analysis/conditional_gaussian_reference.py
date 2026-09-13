"""Classical conditional Gaussian reference for prespecified linear scores.

This module is a scoped benchmark, not a distribution-free panel certificate.
It requires Y | X,Z ~ N(Phi gamma, sigma**2 V), with supplied known V,
and score directions that do not use evaluation outcomes. Neither assumption
can be established by this numerical routine.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import solve_triangular
from scipy.stats import t


def _finite_matrix(value, name):
    value = np.asarray(value, dtype=float)
    if value.ndim != 2 or not np.isfinite(value).all():
        raise ValueError(f"{name} must be a finite matrix")
    return value


def orthogonal_basis(design):
    design = _finite_matrix(design, "design")
    u, singular, _ = np.linalg.svd(design, full_matrices=False)
    tolerance = max(design.shape) * np.finfo(float).eps * singular.max(initial=0)
    return u[:, singular > tolerance]


class GaussianReference:
    def __init__(self, mean_design, covariance):
        self.design = _finite_matrix(mean_design, "mean_design")
        covariance = _finite_matrix(covariance, "covariance")
        n = len(self.design)
        if covariance.shape != (n, n) or not np.allclose(covariance, covariance.T, atol=1e-12, rtol=1e-12):
            raise ValueError("covariance must be symmetric and match the design")
        self.cholesky = np.linalg.cholesky(covariance)
        self.basis = orthogonal_basis(self.whiten(self.design))
        self.dimension = n - self.basis.shape[1]
        if self.dimension < 2:
            raise ValueError("at least two residual dimensions required")

    def whiten(self, values):
        return solve_triangular(self.cholesky, np.asarray(values, float), lower=True)

    def project(self, values):
        return values - self.basis @ (self.basis.T @ values)

    def evaluate(self, outcomes, raw_directions):
        """Columns pair outcomes and directions; zero directions abstain.

        Exact one-sided t reference holds under the module's stated null.
        Outcome-dependent direction selection is outside its guarantee.
        """
        y = _finite_matrix(outcomes, "outcomes")
        a = _finite_matrix(raw_directions, "raw_directions")
        if y.shape != a.shape or len(y) != len(self.design):
            raise ValueError("outcomes and directions must have the same shape")
        r = self.project(self.whiten(y))
        v = self.project(self.cholesky.T @ a)
        vn = np.linalg.norm(v, axis=0)
        rn = np.linalg.norm(r, axis=0)
        defined = (vn > 1e-11 * (1 + np.linalg.norm(self.cholesky.T @ a, axis=0))) & (rn > 1e-12)
        numerator = np.divide(np.sum(v * r, axis=0), vn, out=np.zeros_like(vn), where=defined)
        remaining = np.maximum(0., np.sum(r * r, axis=0) - numerator**2)
        statistic = np.divide(numerator, np.sqrt(remaining / (self.dimension - 1)), out=np.zeros_like(vn), where=defined & (remaining > 0))
        boundary = defined & (remaining == 0) & (numerator != 0)
        statistic[boundary] = np.copysign(np.inf, numerator[boundary])
        pvalue = np.where(defined, t.sf(statistic, self.dimension - 1), 1.)
        return {"statistic": statistic, "pvalue": pvalue, "defined": defined,
                "direction": v, "residual": r, "df": self.dimension - 1}


def leave_cluster_operator(design, clusters):
    """Block leave-out residual operator, normalized to target mean covariance.

    A Phi=0 and A_gg=I/n whenever deleting every cluster preserves design rank.
    This is a classical leave-out regression construction, not a new estimator.
    """
    phi = _finite_matrix(design, "design")
    groups = np.asarray(clusters)
    if groups.ndim != 1 or len(groups) != len(phi):
        raise ValueError("clusters must match design rows")
    n = len(phi)
    rank = orthogonal_basis(phi).shape[1]
    if rank != phi.shape[1]:
        raise ValueError("supply a full-column-rank mean design")
    result = np.eye(n) / n
    for group in np.unique(groups):
        held = np.flatnonzero(groups == group)
        train = np.flatnonzero(groups != group)
        if orthogonal_basis(phi[train]).shape[1] != rank:
            raise ValueError("deleting a cluster loses a mean-design direction")
        result[np.ix_(held, train)] = -phi[held] @ np.linalg.pinv(phi[train]) / n
    if np.max(np.abs(result @ phi), initial=0) > 1e-9:
        raise ArithmeticError("leave-out annihilation check failed")
    return result
