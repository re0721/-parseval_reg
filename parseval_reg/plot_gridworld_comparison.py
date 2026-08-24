"""Gridworld comparison figures: main baselines + ablation (performance profiles)."""

import os
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scipy.stats

SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
AXIS = "#c3c2b7"

# categorical palette (reference), Parseval highlighted as blue
COLORS = {
    "Parseval": "#2a78d6",
    "base": "#eb6834",
    "LayerNorm": "#1baf7a",
    "SnP": "#eda100",
    "Regen": "#e87ba4",
    "W-Regen": "#008300",
    "angles-only": "#1baf7a",
    "norm-only": "#eda100",
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
PTS = CHANGE // SAVE  # 8 points per task, 20 tasks


def load_task_means(pattern, seeds=range(6)):
    out = []
    for s in seeds:
        with open(f"results/{pattern.format(s=s)}", "rb") as f:
            d = pickle.load(f)
        c = np.asarray(d["mean_eval_success"], dtype=float)
        for t in range(20):
            seg = c[t * PTS:(t + 1) * PTS]
            out.append(seg[PTS // 2:].mean())
    return np.asarray(out)


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
    ax.grid(True, color=GRIDLINE, linewidth=0.7, alpha=0.9)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def profile(ax, methods):
    for name, pattern in methods.items():
        vals = load_task_means(pattern)
        x = np.sort(vals)
        y = 1.0 - np.arange(len(x)) / float(len(x))
        ax.plot(x, y, color=COLORS[name], linewidth=2.0, label=name, zorder=3)
    style_axes(ax)
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("Average success rate")
    ax.set_ylabel("Pr(Success rate > x)")
    ax.legend(frameon=False, loc="upper right", fontsize=10)


def main():
    os.makedirs(PLOT_DIR, exist_ok=True)

    # Fig.4-style: Parseval vs baselines
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    profile(ax, {
        "Parseval": "data_gridworld_ninerooms_parseval_{s}.pkl",
        "base": "data_gridworld_ninerooms_base_{s}.pkl",
        "LayerNorm": "data_gridworld_ninerooms_layer_norm_{s}.pkl",
        "SnP": "data_gridworld_ninerooms_snp_{s}.pkl",
        "Regen": "data_gridworld_ninerooms_regen_{s}.pkl",
        "W-Regen": "data_gridworld_ninerooms_w-regen_{s}.pkl",
    })
    ax.set_title("Gridworld — algorithm comparison", loc="left",
                 color=INK_PRIMARY, fontweight="bold", pad=10)
    fig.savefig(f"{PLOT_DIR}/gridworld_method_comparison.png")
    plt.close(fig)
    print("wrote", f"{PLOT_DIR}/gridworld_method_comparison.png")

    # Fig.6-style: ablation
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    profile(ax, {
        "Parseval": "data_gridworld_ninerooms_parseval_{s}.pkl",
        "angles-only": "data_gridworld_ninerooms_parseval_{s}_angles.pkl",
        "norm-only": "data_gridworld_ninerooms_parseval_{s}_groups64.pkl",
    })
    ax.set_title("Gridworld — ablation (angles vs norm)", loc="left",
                 color=INK_PRIMARY, fontweight="bold", pad=10)
    fig.savefig(f"{PLOT_DIR}/gridworld_ablation.png")
    plt.close(fig)
    print("wrote", f"{PLOT_DIR}/gridworld_ablation.png")


if __name__ == "__main__":
    main()
