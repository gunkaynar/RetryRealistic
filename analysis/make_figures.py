#!/usr/bin/env python3
"""Draw the paper's two result figures from the labels.

  figs/fig_fabrication_by_fault.pdf   Figure 2: baseline outcome by fault type
  figs/fig_monitor_tradeoff.pdf       Figure 3: the Monitor's effect, pooled

Usage: python analysis/make_figures.py
"""
import os, sys, collections
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from common import REPO, LABELS, labels

rcParams.update({"font.size": 9, "font.family": "serif", "axes.spines.top": False,
                 "axes.spines.right": False, "figure.dpi": 150, "savefig.bbox": "tight"})
# Okabe-Ito: correct = bluish green, fabricated = vermillion, honest failure = grey
C = {"correct": "#009E73", "fabricated": "#D55E00", "honest_failure": "#999999"}
META = {"CreationDate": None}          # no timestamp, so a rerun gives the same bytes


def pool(rows):
    rows = [r for r in rows if r["fault"] != "none" and r["label"] in LABELS]
    c = collections.Counter(r["label"] for r in rows)
    return {o: 100.0 * c[o] / len(rows) for o in LABELS}


def main(argv):
    out = os.path.join(REPO, "figs")
    os.makedirs(out, exist_ok=True)
    base = [r for r in labels("baseline") if r["fault"] != "none"]

    byf = {f: pool([r for r in base if r["fault"] == f]) for f in {r["fault"] for r in base}}
    faults = sorted(byf, key=lambda f: (-byf[f]["fabricated"], f))
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    y = list(range(len(faults)))
    left = [0.0] * len(faults)
    for o in LABELS:
        vals = [byf[f][o] for f in faults]
        ax.barh(y, vals, left=left, color=C[o], height=0.72, label=o.replace("_", " "),
                edgecolor="white", linewidth=0.8)
        if o == "fabricated":
            for i in y:
                ax.text(left[i] + vals[i] / 2, i, f"{vals[i]:.0f}", ha="center", va="center",
                        color="white", fontsize=8, fontweight="bold")
        left = [a + b for a, b in zip(left, vals)]
    ax.set_yticks(y); ax.set_yticklabels([f.replace("_", " ") for f in faults], fontsize=8)
    ax.invert_yaxis(); ax.set_xlim(0, 100); ax.set_xlabel("% of fault-fired runs")
    ax.legend(ncol=3, fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, 1.13), frameon=False)
    ax.tick_params(length=0)
    fig.savefig(os.path.join(out, "fig_fabrication_by_fault.pdf"), metadata=META); plt.close(fig)

    pts = {"baseline": pool(base), "full cascade": pool(labels("monitor_full")),
           "code-only": pool(labels("monitor_code_only"))}
    mk = {"baseline": "o", "full cascade": "s", "code-only": "^"}
    col = {"baseline": "#999999", "full cascade": "#D55E00", "code-only": "#009E73"}
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    for name, p in pts.items():
        ax.scatter(p["fabricated"], p["correct"], s=90, marker=mk[name], color=col[name], zorder=3)
        ax.annotate(name, (p["fabricated"], p["correct"]), textcoords="offset points",
                    xytext=(6, 6), fontsize=8)
    ax.set_xlabel("% fabricated  (lower = better)")
    ax.set_ylabel("% correct  (higher = better)")
    ax.grid(True, color="#e6e6e6", linewidth=0.6, zorder=0)
    ax.set_axisbelow(True); ax.tick_params(length=0); ax.margins(0.18)
    fig.savefig(os.path.join(out, "fig_monitor_tradeoff.pdf"), metadata=META); plt.close(fig)
    print("figures written to figs/")


if __name__ == "__main__":
    main(sys.argv[1:])
