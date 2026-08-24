"""Four-method comparison: base / Parseval / Stiefel / Pion (performance profiles)."""

import os
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
AXIS = "#c3c2b7"

COLORS = {
    "Pion (spectrum-preserving)": "#2a78d6",
    "Parseval (soft)": "#1baf7a",
    "Stiefel (hard)": "#eda100",
    "base": "#eb6834",
}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "DejaVu Sans"],
    "font.size": 11,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
    "axes.labelcolor": INK_SECONDARY,
    "xtick.color": INK_MUTED, "ytick.color": INK_MUTED,
    "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.dpi": 200, "savefig.bbox": "tight",
})

PLOT_DIR = "plots"
SAVE, CHANGE = 5000, 40000
PTS = CHANGE // SAVE


def load_task_means(pat, seeds=range(6)):
    curves = []
    for s in seeds:
        with open(f"results/{pat.format(s=s)}", "rb") as f:
            d = pickle.load(f)
        curves.append(np.asarray(d["mean_eval_success"], dtype=float))
    L = min(len(c) for c in curves)
    curves = [c[:L] for c in curves]
    out = []
    nt = L // PTS
    for c in curves:
        for t in range(nt):
            seg = c[t * PTS:(t + 1) * PTS]
            out.append(seg[PTS // 2:].mean())
    return np.asarray(out), nt


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
    ax.grid(True, color=GRIDLINE, linewidth=0.7, alpha=0.9)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def main():
    os.makedirs(PLOT_DIR, exist_ok=True)
    methods = {
        "Pion (spectrum-preserving)": "data_gridworld_ninerooms_pion_{s}.pkl",
        "Parseval (soft)": "data_gridworld_ninerooms_parseval_{s}.pkl",
        "Stiefel (hard)": "data_gridworld_ninerooms_lie_group_{s}.pkl",
        "base": "data_gridworld_ninerooms_base_{s}.pkl",
    }

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for name, pat in methods.items():
        vals, nt = load_task_means(pat)
        x = np.sort(vals)
        y = 1.0 - np.arange(len(x)) / float(len(x))
        ax.plot(x, y, color=COLORS[name], linewidth=2.0, label=name, zorder=3)
    style_axes(ax)
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("Average success rate")
    ax.set_ylabel("Pr(Success rate > x)")
    ax.set_title("Gridworld — Lie group vs Parseval", loc="left",
                 color=INK_PRIMARY, fontweight="bold", pad=10)
    ax.legend(frameon=False, loc="lower left", fontsize=9)
    out = f"{PLOT_DIR}/gridworld_lie_comparison.png"
    fig.savefig(out)
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    main()
