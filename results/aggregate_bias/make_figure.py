"""Render recorded detection and aggregate-bias results without new experiments."""
from pathlib import Path
import argparse
import hashlib
import json
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)

source_a = HERE.parent / "certificate_factorial" / "summary.csv"
source_b = HERE / "simulation" / "summary.csv"
factorial = pd.read_csv(source_a)
left = factorial[(factorial.design == "eight_rare") &
                 (factorial.signal == .25) & (factorial.fit == "estimated") &
                 (factorial.validation_pairs == 8192) &
                 factorial.groups.isin([100, 400])].copy()
assert len(left) == 8
expected = {"absolute_range": [0., 0.], "signed_range": [0., 0.],
            "absolute_variance": [.274, 1.], "signed_variance": [.435, 1.]}
for method, values in expected.items():
    actual = left[left.method == method].sort_values("groups")
    assert actual.total_observations.tolist() == [30976, 50176]
    assert actual.power.tolist() == values

summary = pd.read_csv(source_b)
keys = ["categories", "validation_pairs", "mass", "fit_error"]
assert not summary.duplicated(keys + ["method"]).any()
assert len(summary) == 144 and summary.repetitions.eq(2000).all()
points = summary.pivot(index=keys, columns="method", values="median_slack").reset_index()
points.columns.name = None
points = points.rename(columns={"rectangle": "rectangle_median_slack",
                                "aggregate": "aggregate_median_slack"})
points["ratio"] = points.rectangle_median_slack / points.aggregate_median_slack
assert len(points) == 72 and points.groupby("categories").size().to_dict() == {
    2: 18, 8: 18, 32: 18, 128: 18}
assert np.isfinite(points.ratio).all() and (points.ratio > 0).all()
assert int((points.ratio < 1).sum()) == 10
assert int((points.ratio > 1).sum()) == 62

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8,
    "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 8,
    "ytick.labelsize": 8, "legend.fontsize": 8, "pdf.fonttype": 42,
    "ps.fonttype": 42, "svg.fonttype": "none", "svg.hashsalt": "aggregate-bias-figure"})
fig, axes = plt.subplots(1, 2, figsize=(7, 2.05),
                         gridspec_kw={"width_ratios": [1, 1.25]})
fig.subplots_adjust(left=.070, right=.987, bottom=.235, top=.70, wspace=.31)
blue, orange = "#176b9d", "#b35a00"
styles = {
    "absolute_range": dict(color=orange, linestyle="--", marker="x", markersize=6),
    "signed_range": dict(color=blue, linestyle="--", marker="D", markersize=6,
                         markerfacecolor="none"),
    "absolute_variance": dict(color=orange, linestyle="-", marker="s", markersize=5.3,
                              markerfacecolor="none"),
    "signed_variance": dict(color=blue, linestyle="-", marker="o", markersize=3.2)}
labels = ["Absolute + range", "Signed + range", "Absolute + variance", "Signed + variance"]
handles = []
ax = axes[0]
for method, label in zip(styles, labels):
    rows = left[left.method == method].sort_values("groups")
    line, = ax.plot(rows.total_observations, rows.power, linewidth=1.15,
                    clip_on=False, label=label, **styles[method])
    handles.append(line)
ax.set_xlim(29200, 52300)
ax.set_ylim(-.055, 1.055)
ax.set_yticks([0, .5, 1], labels=["0", "0.5", "1"])
ax.set_xticks([30976, 50176], labels=["30,976", "50,176"])
ax.set_ylabel("Rejection rate", labelpad=2)
ax.set_xlabel("Total observations", labelpad=2)
ax.set_title(r"(a) Eight categories, $\vartheta=0.01366$", pad=5)
ax.annotate("0.435", (30976, .435), xytext=(5, 3), textcoords="offset points", color=blue)
ax.annotate("0.274", (30976, .274), xytext=(5, -10), textcoords="offset points", color=orange)
ax.legend(handles, labels, loc="lower center", bbox_to_anchor=(.47, 1.26),
          ncol=2, frameon=False, handlelength=1.5, handletextpad=.4,
          columnspacing=.85, labelspacing=.35, borderpad=0)

ax = axes[1]
ax.set_xlim(.5, 4.5)
ax.set_yscale("log", base=2)
ax.set_ylim(.45, 20)
ax.set_yticks([.5, 1, 2, 4, 8, 16], labels=["0.5", "1", "2", "4", "8", "16"])
ax.minorticks_off()
ax.set_xticks([1, 2, 3, 4], labels=["2", "8", "32", "128"])
ax.set_xlabel("Number of categories", labelpad=2)
ax.set_ylabel("Rectangle / aggregate slack", labelpad=2)
ax.set_title("(b) Aggregate learning-bias bound", pad=5)
ax.axhline(1, color=".35", linestyle=(0, (3, 2)), linewidth=.7, zorder=1)
for axis in axes:
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", color=".90", linewidth=.5)
    axis.set_axisbelow(True)

# Deterministic horizontal packing separates markers while keeping every y value
# equal to its recorded ratio. Horizontal offsets do not encode another variable.
fig.canvas.draw()
diameter = 4.1 * fig.dpi / 72
points["plot_x"] = np.nan
group_summaries = []
for category, center in zip([2, 8, 32, 128], [1., 2., 3., 4.]):
    rows = points[points.categories == category].sort_values(["ratio"] + keys[1:])
    placed = []
    x_center = ax.transData.transform((center, 1))[0]
    for idx, row in rows.iterrows():
        y_pixel = ax.transData.transform((center, row.ratio))[1]
        candidates = [0.]
        for old_x, old_y in placed:
            vertical = abs(old_y - y_pixel)
            if vertical < diameter:
                horizontal = math.sqrt(diameter * diameter - vertical * vertical)
                candidates.extend([old_x - x_center - horizontal,
                                   old_x - x_center + horizontal])
        valid = [offset for offset in candidates if all(
            (x_center + offset - old_x) ** 2 + (y_pixel - old_y) ** 2 >=
            diameter ** 2 - 1e-7 for old_x, old_y in placed)]
        offset = min(valid, key=lambda value: (abs(value), value))
        x = ax.transData.inverted().transform((x_center + offset, y_pixel))[0]
        assert abs(x - center) < .45
        points.loc[idx, "plot_x"] = x
        placed.append((x_center + offset, y_pixel))
    median = float(rows.ratio.median())
    group_summaries.append(dict(categories=category, count=18,
        minimum=float(rows.ratio.min()), median=median, maximum=float(rows.ratio.max()),
        aggregate_tighter=int((rows.ratio > 1).sum()),
        rectangle_tighter=int((rows.ratio < 1).sum())))
    ax.hlines(median, center - .25, center + .25, color=".15", linewidth=1.05, zorder=2)
for mask, color in [(points.ratio > 1, blue), (points.ratio < 1, orange)]:
    rows = points[mask]
    ax.scatter(rows.plot_x, rows.ratio, s=10, facecolors="white", edgecolors=color,
               linewidths=.75, zorder=3, clip_on=False)
ax.legend(handles=[Line2D([], [], marker="o", markersize=3.3, linestyle="none",
                        markerfacecolor="white", color=blue, label="Aggregate tighter"),
                   Line2D([], [], marker="o", markersize=3.3, linestyle="none",
                        markerfacecolor="white", color=orange, label="Rectangle tighter"),
                   Line2D([], [], color=".15", linewidth=1.05, label="Median across settings")],
          loc="lower center", bbox_to_anchor=(.52, 1.26), ncol=2, frameon=False,
          handlelength=1.15, handletextpad=.4, columnspacing=.85,
          labelspacing=.35, borderpad=0)

stem = args.output / "fig_aggregate_validation"
for extension in ["pdf", "svg", "png"]:
    metadata = {"CreationDate": None, "ModDate": None, "Creator": None,
                "Producer": None} if extension == "pdf" else (
                {"Date": None, "Creator": None} if extension == "svg" else None)
    fig.savefig(stem.with_suffix("." + extension), dpi=300, metadata=metadata)
plt.close(fig)
points.to_csv(args.output / "aggregate_validation_points.csv", index=False,
              float_format="%.17g")
left.to_csv(args.output / "factorial_detection_points.csv", index=False,
            float_format="%.17g")
record = {
    "size_inches": [7, 2.05], "minimum_font_points": 8,
    "left_panel": "Previously reported eight-category estimated-fit ablation; 1,000 repetitions per cell.",
    "right_panel": "All 72 known-category-probability validation settings; 2,000 repetitions per method and setting.",
    "ratio_definition": "Median(rectangle upper bound - true bias) / median(aggregate upper bound - true bias), computed separately within each setting.",
    "interpretation": "Ratios above one favor the aggregate bound. Ten ratios are below one. Horizontal black marks are descriptive medians across 18 settings, not confidence intervals.",
    "point_placement": "Deterministic horizontal packing only; no vertical jitter, trimming, or setting selection.",
    "scope": "Validation-bound comparison and previously reported ablation; not an independent forecast confirmation, discovery-power comparison for the aggregate method, or general superiority claim.",
    "counts": {"settings": 72, "aggregate_tighter": 62, "rectangle_tighter": 10},
    "group_summaries": group_summaries,
    "input_sha256": {"../certificate_factorial/summary.csv": hashlib.sha256(source_a.read_bytes()).hexdigest(),
                     "simulation/summary.csv": hashlib.sha256(source_b.read_bytes()).hexdigest()},
    "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
}
stem.with_suffix(".json").write_text(json.dumps(record, indent=2) + "\n")
print(json.dumps(record["counts"]))
