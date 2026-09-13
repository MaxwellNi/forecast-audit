"""Native vector figure from the complete frozen Gaussian-reference study."""
from pathlib import Path
import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "results/gaussian_reference/summary.json"
OUTPUT = ROOT / "results/gaussian_figure"


def main():
    global SOURCE, OUTPUT
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=SOURCE)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    SOURCE, OUTPUT = args.source, args.output_dir
    d = json.loads(SOURCE.read_text())
    OUTPUT.mkdir(parents=True, exist_ok=False)
    plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 8,
                         "axes.labelsize": 8, "axes.titlesize": 9,
                         "pdf.fonttype": 42, "ps.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.linewidth": .6, "xtick.major.size": 2.5,
                         "ytick.major.size": 2.5, "svg.hashsalt": "gaussian-reference-fixed"})
    fig, axes = plt.subplots(1, 2, figsize=(7., 2.65))
    fig.subplots_adjust(left=.085, right=.985, top=.84, bottom=.30, wspace=.35)
    metadata = []
    choices = [
        [("extrapolation_beta2_normal", "Normal reference", "#777777", "s", -.025),
         ("extrapolation_beta2_gaussian", "Gaussian reference", "#D55E00", "o", .025)],
        [("leave_cluster_raw_gaussian", "Uncentered", "#777777", "s", -.045),
         ("leave_cluster_learned_gaussian", "Learned mean", "#0072B2", "o", 0.),
         ("gls_gaussian", "GLS", "#111111", "^", .045)]]
    for panel, (axis, methods) in enumerate(zip(axes, choices)):
        effect = [0., .15][panel]
        for method, label, color, marker, offset in methods:
            rows = sorted([r for r in d["rows"] if r["specification"] == "correct" and
                           r["effect"] == effect and r["method"] == method], key=lambda r: r["design"])
            if len(rows) != 4:
                raise ValueError("Incomplete figure cells")
            x = np.arange(4) + offset
            y = np.array([r["rate"] for r in rows])
            intervals = np.array([r["wilson95"] for r in rows])
            axis.errorbar(x, y, yerr=np.array([y-intervals[:, 0], intervals[:, 1]-y]),
                          color=color, marker=marker, markersize=3.7, linewidth=.9,
                          capsize=2., elinewidth=.7, label=label)
            metadata.extend([{"panel": panel+1, "method": method, "design": r["design"],
                              "x": float(xx), "rate": r["rate"], "wilson95": r["wilson95"]}
                             for xx, r in zip(x, rows)])
        axis.set_xticks(range(4), ["12, 0", "12, 0.6", "36, 0", "36, 0.6"])
        axis.set_xlabel(r"Periods $M$, serial correlation $\rho$")
        axis.set_xlim(-.2, 3.2)
        axis.yaxis.set_major_formatter(PercentFormatter(1., decimals=0))
        axis.grid(axis="y", color=".9", linewidth=.5)
        axis.set_axisbelow(True)
        axis.legend(loc="upper center", bbox_to_anchor=(.5, -.36), ncol=len(methods),
                    frameon=False, fontsize=7, handlelength=1.4, columnspacing=.9)
    axes[0].set(title=r"(a) Null: extrapolation with $\beta=2$", ylabel="False-positive rate", ylim=(0, .65))
    axes[0].axhline(.05, color="#111111", linestyle=":", linewidth=.8)
    axes[1].set(title=r"(b) Weak signal: leave-cluster scores", ylabel="Detection rate", ylim=(0, .8))
    fig.savefig(OUTPUT / "gaussian_reference.pdf", metadata={"Creator": "Matplotlib", "CreationDate": None, "ModDate": None})
    fig.savefig(OUTPUT / "gaussian_reference.svg", metadata={"Date": None})
    fig.savefig(OUTPUT / "gaussian_reference.png", dpi=200)
    plt.close(fig)
    record = {"source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
              "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "scope": "Correct known-covariance Gaussian cells only; raw directions are scalar-bin probes, not the deployed ranked-panel pipeline. Error bars are marginal Wilson95, not simultaneous intervals.",
              "points": metadata, "size_inches": [7., 2.65], "raster_dpi": 200}
    (OUTPUT / "provenance.json").write_text(json.dumps(record, indent=2)+'\n')


if __name__ == "__main__":
    main()
