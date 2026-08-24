"""Gridworld — stable rank (spectral degradation) of the actor's hidden layers.

stable rank = (||W||_F / ||W||_2)^2 = sum(sigma_i^2) / sigma_max^2.
It measures how many effective directions the weight matrix has: ~64 = full rank,
~1 = collapsed to a single direction. Loss of plasticity is driven by this
collapse (the base agent's hidden-layer stable rank falls from ~21/34 to ~3/8).

Data comes from the pkl field `actor_matrix_stable_rank`, which logs one value per
linear layer with 'weight' in its name (agent.py get_log_quantities). Layer order:
  [0] input->hidden (225x64), [1] hidden->hidden (64x64), [2] hidden->output (64x4).

NOTE: OFT's hidden layers (OETLinear) name their frozen base as 'W0', not 'weight',
so their hidden-layer stable rank is NOT logged (only the output layer). OFT is
therefore omitted here; its spectrum is frozen by construction just like POET's.
"""
import os
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scipy.stats
from matplotlib.ticker import FuncFormatter

SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
AXIS = "#c3c2b7"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "font.size": 11,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
    "axes.labelcolor": INK_SECONDARY,
    "xtick.color": INK_MUTED, "ytick.color": INK_MUTED,
    "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.dpi": 200, "savefig.bbox": "tight",
})

PLOT_DIR = "plots"
SAVE, CHANGE, NUM_STEPS = 5000, 40000, 800000
SEEDS = range(6)

METHODS = [
    ("base",       "base (no reg.)",               "#8b8a85", 'results/data_gridworld_ninerooms_base_{s}.pkl'),
    ("parseval",   "Parseval (soft reg.)",         "#1baf7a", 'results/data_gridworld_ninerooms_parseval_{s}.pkl'),
    ("pion",       "Pion (spectrum-preserving)",   "#2a78d6", 'results/data_gridworld_ninerooms_pion_{s}_full.pkl'),
    ("poet_exact", "POET-exact",                   "#7b4fd6", 'results/data_gridworld_ninerooms_poet_exact_{s}_full_lr5e4.pkl'),
]
LAYERS = [(0, "Input -> hidden (225 x 64)"), (1, "Hidden -> hidden (64 x 64)")]


def load(pattern):
    arrs = []
    for s in SEEDS:
        with open(pattern.format(s=s), "rb") as f:
            d = pickle.load(f)
        arrs.append(np.asarray(d["actor_matrix_stable_rank"], dtype=float))  # (160, 3)
    return np.stack(arrs)  # (seeds, 160, 3)


def iqm(x, axis=0):
    return scipy.stats.trim_mean(x, 0.25, axis)


def moving_avg(x, w):
    if w <= 1:
        return np.asarray(x, float)
    x = np.asarray(x, float)
    out = np.empty_like(x)
    c = np.cumsum(np.insert(x, 0, 0))
    for i in range(len(x)):
        lo = max(0, i - w + 1)
        out[i] = (c[i + 1] - c[lo]) / (i + 1 - lo)
    return out


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
    ax.grid(True, color=GRIDLINE, linewidth=0.7, alpha=0.9)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def mark_tasks(ax):
    for x in np.arange(CHANGE, NUM_STEPS, CHANGE):
        ax.axvline(x, color=INK_MUTED, linewidth=0.7, linestyle="--", alpha=0.30, zorder=0)


def fmt_steps(x, _):
    return f"{x/1e6:.1f}M" if x >= 1e6 else f"{x/1e3:.0f}k"


def main():
    os.makedirs(PLOT_DIR, exist_ok=True)
    data = {k: load(p) for k, _, _, p in METHODS}
    xs = SAVE * np.arange(data["base"].shape[1])

    fig, axes = plt.subplots(2, 1, figsize=(8.0, 7.2), sharex=True)
    for layer_idx, title in LAYERS:
        ax = axes[layer_idx]
        for key, label, color, _ in METHODS:
            v = data[key][:, :, layer_idx]          # (seeds, 160)
            m = moving_avg(iqm(v, 0), 5)
            lo = np.percentile(v, 25, axis=0)
            hi = np.percentile(v, 75, axis=0)
            ax.plot(xs, m, color=color, linewidth=2.0, label=label, zorder=3)
            ax.fill_between(xs, moving_avg(lo, 5), moving_avg(hi, 5),
                            color=color, alpha=0.10, linewidth=0, zorder=2)
        mark_tasks(ax)
        style_axes(ax)
        ax.set_ylabel("Stable rank")
        ax.set_title(title, loc="left", color=INK_SECONDARY, fontsize=11, pad=6)
        ax.set_ylim(0, 64)
    axes[1].set_xlabel("Environment steps (dashed = task switch)")
    axes[1].xaxis.set_major_formatter(FuncFormatter(fmt_steps))
    axes[1].set_xlim(0, NUM_STEPS)
    fig.suptitle("Gridworld — actor stable rank (spectral degradation vs preservation)",
                 x=0.005, ha="left", color=INK_PRIMARY, fontweight="bold", fontsize=13, y=0.995)
    axes[0].legend(frameon=False, loc="upper right", ncol=2, fontsize=9.5)
    out = f"{PLOT_DIR}/gridworld_stable_rank.png"
    fig.savefig(out)
    plt.close(fig)
    print("wrote", out)

    # per-method summary for the record
    print("\nStable rank: first vs last task (mean over 6 seeds):")
    print(f"{'Method':<28} {'layer':<22} {'first':>7} {'last':>7} {'delta':>8}")
    print("-" * 74)
    for key, label, _, _ in METHODS:
        for layer_idx, lname in LAYERS:
            v = data[key][:, :, layer_idx]      # (seeds, 160)
            # first/last task means (drop the boundary eval at task end)
            a = v[:, :8][:, :-1].mean()
            b = v[:, -8:][:, :-1].mean()
            print(f"{label:<28} {lname:<22} {a:>7.2f} {b:>7.2f} {b-a:>+8.2f}")


if __name__ == "__main__":
    main()
