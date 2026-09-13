#!/usr/bin/env python3
"""Rebuild four figures from recorded aggregates and exact identities.

No observations are simulated, removed, or jittered on a measured axis.
Vertical offsets in the canonical comparison separate conditions only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

COLORS = ("#333333", "#0072B2", "#D55E00", "#009E73", "#882255")
MARKERS = ("o", "s", "^", "D", "x")
plt.rcParams.update({
    "font.family": "Liberation Serif", "font.size": 8.5,
    "mathtext.fontset": "stix", "axes.labelsize": 8.5,
    "axes.titlesize": 8.5, "xtick.labelsize": 8,
    "ytick.labelsize": 8, "legend.fontsize": 8,
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6,
    "ytick.major.width": 0.6, "svg.hashsalt": "icdm-candidate-figures-20260907",
})


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(fig, directory, name, record):
    # Keep a fixed physical canvas: final 3.5-inch placement preserves font sizes.
    fig.savefig(directory / (name + ".pdf"), metadata={
        "CreationDate": None, "ModDate": None, "Creator": "Matplotlib",
    })
    fig.savefig(directory / (name + ".svg"), metadata={"Date": None})
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    outside = []
    for artist in fig.findobj(match=matplotlib.text.Text):
        if not artist.get_visible() or not artist.get_text():
            continue
        box = artist.get_window_extent(renderer)
        if box.width and box.height and (
            box.x0 < -0.5 or box.y0 < -0.5 or
            box.x1 > fig.bbox.width + 0.5 or box.y1 > fig.bbox.height + 0.5
        ):
            outside.append(artist.get_text())
    if outside:
        raise RuntimeError(f"Text outside {name}: {outside}")
    record["canvas_points"] = [v * 72 for v in fig.get_size_inches()]
    record["pdf_sha256"] = digest(directory / (name + ".pdf"))
    record["svg_sha256"] = digest(directory / (name + ".svg"))
    record["visible_text_outside_canvas"] = outside
    plt.close(fig)
    return record


def canonical(directory, inputs):
    rows = inputs["canonical_rows"]
    methods = ["gcm_coarse", "gcm_fine", "gcm_spline", "extrapolated", "kci_gamma"]
    labels = ["GCM: 8 bins", "GCM: 32 bins", "GCM: splines", "Extrapolation", "KCI"]
    conditions = [("linear", "null", "Linear"), ("smooth", "null", "Smooth"),
                  ("nonlinear", "null", "Quadratic"), ("heavy_tail", "null", r"Heavy tail ($t_5$)"),
                  ("smooth", "zero_covariance_dependence", "Zero covariance")]
    fig, ax = plt.subplots(figsize=(3.5, 2.85))
    fig.subplots_adjust(left=.275, right=.98, bottom=.17, top=.80)
    plotted = []
    for j, (regime, alternative, label) in enumerate(conditions):
        for i, method in enumerate(methods):
            selected = [r for r in rows if r["regime"] == regime and
                        r["alternative"] == alternative and r["method"] == method]
            if len(selected) != 1:
                raise ValueError("ambiguous canonical result")
            row = selected[0]
            rate = row["rejections"] / row["replications"]
            if rate != row["rate"]:
                raise ValueError("recorded rate disagrees with rejection count")
            lo, hi = row["wilson95"]
            if lo > rate + 1e-14 or hi < rate - 1e-14:
                raise ValueError("Wilson interval does not contain its point estimate")
            ax.errorbar(rate, i + (j - 2) * .14,
                        xerr=[[max(0, rate-lo)], [max(0, hi-rate)]], fmt=MARKERS[j],
                        ms=3.0, color=COLORS[j], mew=.65, mfc="white",
                        elinewidth=.7, capsize=0, zorder=3,
                        label=label if i == 0 else None)
            plotted.append({key: row[key] for key in
                            ("regime", "alternative", "method", "rejections", "replications", "rate", "wilson95")})
    for y in np.arange(.5, 4.5):
        ax.axhline(y, color=".88", lw=.5, zorder=0)
    ax.axvline(.05, color=".25", lw=.7, ls=":", zorder=1)
    ax.set(xlim=(-.008, 1.03), ylim=(4.45, -.45), xlabel="Rejection rate")
    ax.set_yticks(range(5), labels)
    ax.set_xticks([0, .25, .5, .75, 1], ["0", "0.25", "0.50", "0.75", "1"])
    ax.tick_params(axis="y", length=0, pad=4)
    ax.legend(loc="lower center", bbox_to_anchor=(.50, .815),
              bbox_transform=fig.transFigure, ncol=2, frameon=False,
              handlelength=1.4, columnspacing=1.0, labelspacing=.25)
    fig.text(.53, .012, "Zero covariance is an alternative for KCI.",
             ha="center", fontsize=8)
    return save(fig, directory, "fig_canonical_comparison", {
        "plotted_cells": plotted, "stored_source_cells": len(rows),
        "uncertainty": "Recorded pointwise 95% Wilson intervals; no simultaneous-coverage claim.",
        "reference_line": .05,
    })


def controls(directory, inputs):
    domains = [("electricity", "seasonal_naive_7", 14, "Electricity"),
               ("retail", "Last week", 2, "Retail"),
               ("ratings", "Item mean", 0, "Ratings")]
    fig, axes = plt.subplots(3, 1, figsize=(3.5, 2.7))
    fig.subplots_adjust(left=.225, right=.96, bottom=.17, top=.83, hspace=.55)
    values = []
    for ax, (domain, model, lag, title) in zip(axes, domains):
        selected = [r for r in inputs["baseline_rows"] if r["domain"] == domain and
                    r["model"] == model and r["lag"] == lag]
        if len(selected) != 1:
            raise ValueError("ambiguous baseline-copy aggregate")
        row = selected[0]["statistics_10_bins_beta1_beta2"]
        if len(row) != 3 or not all(np.isfinite(row)):
            raise ValueError("invalid baseline-copy statistics")
        low, high = min(min(row), 0), max(max(row), 1.6448536269514722)
        span = high-low
        ax.set(xlim=(low-.13*span, high+.25*span), ylim=(-.6, 2.6))
        for j, value in enumerate(row):
            ax.plot(value, 2-j, MARKERS[j], color=COLORS[j], ms=4.3,
                    mfc="white", mew=.8, zorder=3)
            close_negative = value < 0 and abs(value) < .2*span
            ax.annotate(f"{value:.2f}", (value, 2-j), (-5 if close_negative else 5, 0),
                        textcoords="offset points", va="center", fontsize=8,
                        ha="right" if close_negative else "left")
        ax.axvline(0, color=".6", lw=.55, zorder=0)
        ax.axvline(1.6448536269514722, color=".2", lw=.75, ls=":", zorder=0)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.text(-.04, .5, title, transform=ax.transAxes, ha="right", va="center", fontsize=8.5)
        ax.set_xticks({"electricity": [-20, 0, 20], "retail": [0, 5, 10],
                       "ratings": [-4, 0, 4, 8]}[domain])
        ax.tick_params(axis="x", pad=1, length=2.5)
        values.append({"domain": domain, "model": model, "lag": lag,
                       "statistics_10_bins_beta1_beta2": row})
    axes[-1].set_xlabel(r"Residual statistic $T$ (scales differ)", labelpad=2)
    handles = [Line2D([0], [0], marker=MARKERS[j], color=COLORS[j], linestyle="none",
                      mfc="white", markersize=4, label=label)
               for j, label in enumerate(["10 bins", r"Extrap., $\beta=1$", r"Extrap., $\beta=2$"])]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .99),
               ncol=3, frameon=False, handletextpad=.2, columnspacing=.65)
    return save(fig, directory, "fig_baseline_controls", {
        "plotted_values": values, "reference_line": 1.6448536269514722,
        "uncertainty": "Observed statistics only; no confidence intervals are implied.",
    })


def smooth(directory):
    q = np.array([4, 6, 8, 12, 16, 24, 32, 48, 64], dtype=float)
    ladder = np.array([1, 2, 4], dtype=float)
    # Exact minimum-Euclidean-norm weights for sum(w)=1, sum(w*r^-2)=0.
    weights = np.array([-1/6, 1/2, 2/3])
    assert np.allclose(weights.sum(), 1) and np.allclose(weights @ ladder**-2, 0)
    before = {"Linear": 12/q**2,
              "Quadratic": 16/q**2 - (144/45)/q**4,
              "Sine": 18*(1-np.sinc(1/q)**2)}
    fine = q[:, None]*ladder
    after = {"Quadratic": -(144/45)*(fine**-4 @ weights),
             "Sine": (18*(1-np.sinc(1/fine)**2)) @ weights}
    fig, axes = plt.subplots(1, 2, figsize=(3.5, 2.15), sharex=True)
    fig.subplots_adjust(left=.19, right=.985, bottom=.23, top=.77, wspace=.55)
    handles = []
    for j, (name, values) in enumerate(before.items()):
        line, = axes[0].loglog(q, values, color=COLORS[j], marker=MARKERS[j],
                              mfc="white", ms=3, mew=.6, lw=.85, label=name)
        handles.append(line)
    for j, (name, values) in enumerate(after.items(), 1):
        axes[1].loglog(q, abs(values), color=COLORS[j], marker=MARKERS[j],
                      mfc="white", ms=3, mew=.6, lw=.85)
    axes[0].set_title("One resolution", pad=3)
    axes[1].set_title(r"Extrapolation ($\beta=2$)", pad=3)
    axes[0].set_ylabel("Absolute population bias", labelpad=1)
    axes[0].set_yticks([.001, .01, .1, 1, 10])
    axes[1].set_yticks([1e-7, 1e-5, 1e-3, .1])
    for ax in axes:
        ax.set_xticks([4, 16, 64], ["4", "16", "64"])
        ax.tick_params(axis="both", which="minor", length=0)
        ax.tick_params(axis="both", which="major", pad=1, length=2.5)
        ax.set_xlabel(r"Coarsest bin count $q$", labelpad=2)
    axes[1].text(.05, .03, "Linear: exactly zero", transform=axes[1].transAxes, fontsize=8)
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, 1.01),
               ncol=3, frameon=False, handlelength=1.4, columnspacing=1.0)
    return save(fig, directory, "fig_smooth_bin_bias", {
        "q": q.tolist(), "ladder": ladder.tolist(), "weights": weights.tolist(),
        "before": {k: v.tolist() for k, v in before.items()},
        "after_signed": {k: v.tolist() for k, v in after.items()},
        "linear_after_exact": 0, "source": "Exact population identities, not fitted simulation lines.",
    })


def public_comparison(directory, inputs):
    rows = inputs["public_comparison_rows"]
    if len(rows) != 41 or len({(r["domain"],r["model"]) for r in rows}) != 41:
        raise ValueError("expected all 41 distinct public forecasters")
    panels = [("electricity", "Electricity", (-7, 11), [-5, 0, 5, 10]),
              ("retail", "M5 retail", (-15, 12), [-10, 0, 10]),
              ("ratings", "MovieLens", (0, 16), [0, 5, 10, 15]),
              ("portfolios", "OSAP forecasts", (-3, 2.5), [-2, 0, 2])]
    fig, axes = plt.subplots(2, 2, figsize=(3.5, 3.2))
    fig.subplots_adjust(left=.16, right=.985, bottom=.20, top=.87, hspace=.48, wspace=.38)
    threshold = 1.6448536269514722
    multiplicities = []
    for ax, (domain, title, limits, ticks) in zip(axes.flat, panels):
        selected = [r for r in rows if r["domain"]==domain]
        finite = [r for r in selected if not r["undefined"]]
        grouped = {}
        for row in finite:
            grouped.setdefault((row["extrapolated_T"], row["spline_T"], row["copy_or_order_guard"]), []).append(row["model"])
        for (x,y,guarded), models in grouped.items():
            if not limits[0]<x<limits[1] or not limits[0]<y<limits[1]:
                raise ValueError("public coordinate outside fixed panel limits")
            ax.scatter(x,y,s=25 if guarded else 14,marker="^" if guarded else "o",
                       facecolor="white" if guarded else COLORS[1],
                       edgecolor=COLORS[2] if guarded else COLORS[1],linewidth=.8,zorder=3)
            if len(models)>1:
                ax.annotate(f"×{len(models)}",(x,y),(5,2),textcoords="offset points",fontsize=8)
                multiplicities.append(dict(domain=domain,models=models,x=x,y=y,count=len(models)))
        ax.plot(limits,limits,color=".65",linewidth=.6,zorder=0)
        ax.axhline(threshold,color=".25",linestyle=":",linewidth=.6,zorder=0)
        ax.axvline(threshold,color=".25",linestyle=":",linewidth=.6,zorder=0)
        ax.set(xlim=limits,ylim=limits)
        ax.set_xticks(ticks); ax.set_yticks(ticks)
        ax.set_title(f"{title} ({len(finite)}/{len(selected)})",pad=3)
        ax.tick_params(pad=1,length=2.5)
    fig.text(.55,.11,r"Extrapolated statistic $T$ ($\beta=2$)",ha="center",fontsize=8.5)
    fig.text(.018,.55,r"Spline-adjusted statistic $T$",va="center",rotation=90,fontsize=8.5)
    handles = [Line2D([0],[0],marker="o",color=COLORS[1],linestyle="none",markersize=4,label="Other forecasters"),
               Line2D([0],[0],marker="^",color=COLORS[2],mfc="white",linestyle="none",markersize=5,label="Copy/order guard")]
    fig.legend(handles=handles,loc="upper center",bbox_to_anchor=(.53,1.01),ncol=2,
               frameon=False,handletextpad=.3,columnspacing=1.0)
    fig.text(.53,.025,"Finite pairs / all models; panel scales differ.",ha="center",fontsize=8)
    return save(fig,directory,"fig_public_comparison",{
        "beta":2,"all_models":rows,"finite_pairs":sum(not r["undefined"] for r in rows),
        "undefined_pairs":sum(r["undefined"] for r in rows),"coincident_groups":multiplicities,
        "reference_line":threshold,"axis_scales":"Linear, limits stated separately for each domain",
        "interpretation":"Raw statistics before the copy/order guard; larger T is not evidence of better calibration or power.",
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", type=Path, default=Path("results/figure_evidence"))
    parser.add_argument("--output-dir", "--output", dest="output", type=Path, default=Path("figures"))
    parser.add_argument("--receipt", type=Path, default=None)
    parser.add_argument("--primary-beta", type=int, choices=(2,), default=2)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    args.receipt = args.receipt or args.output/"figure_receipt.json"
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    inputs_path = args.evidence_root/"plot_inputs.json"
    inputs = json.loads(inputs_path.read_text())
    records = {"fig_canonical_comparison": canonical(args.output, inputs),
               "fig_baseline_controls": controls(args.output, inputs),
               "fig_smooth_bin_bias": smooth(args.output),
               "fig_public_comparison": public_comparison(args.output, inputs)}
    receipt = {"generator_sha256": digest(Path(__file__)),
               "evidence_sha256": digest(inputs_path), "sources": inputs["source_sha256"],
               "figures": records, "font": "Liberation Serif; STIX mathematics",
               "minimum_declared_font_points": 8,
               "final_placement_width_inches": 3.5,
               "selection": "All 25 requested comparison cells, all 9 baseline statistics, all 41 public model pairs (38 finite, 3 undefined)."}
    args.receipt.write_text(json.dumps(receipt, indent=2)+"\n")
    print(json.dumps({"receipt": str(args.receipt), "figures": list(records)}))


if __name__ == "__main__":
    main()
