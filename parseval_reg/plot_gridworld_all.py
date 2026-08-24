"""Gridworld six-method comparison plots (English labels, smoothed).

The advisor asked for EMA/averaging to see the overall trend. But each run has
only 160 eval points (800k steps / save_freq 5000), so a 100-point window would
cover 62.5% of the run and erase the 20-task structure entirely (8 points per
task). Three figures, each at a data-appropriate granularity:

  fig1 raw curve (seed IQM, light smoothing w=3) -- keeps the sawtooth
  fig2 EMA smoothing at three spans -- shows the smoothing-strength tradeoff
  fig3 per-task trend (one point per task) -- plasticity loss at a glance

Boundary note: main.py evaluates on the same step the goal switches
((i_step+1) % freq == 0), so the last point of each task measures the brand-new,
zero-training task (almost always 0). fig3 drops it; fig1/fig2 keep it (it is a
real measurement of the new task at t=0, just offset by one eval interval).
"""
import os
import re
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
PTS_PER_TASK = CHANGE // SAVE          # 8
SEEDS = range(6)

# Legend order = by final performance, best first
METHODS = [
    ("pion",       "Pion (spectrum-preserving)",      "#2a78d6", 'logs/pion_full', 'pion'),
    ("poet_exact", "POET-exact",                       "#7b4fd6", 'logs/full',   'poet_exact_lr5e4'),
    ("parseval",   "Parseval (soft reg.)",             "#1baf7a", 'logs/ref',       'gw_parseval'),
    ("oft",        "OFT (one-sided)",                  "#e08a1e", 'logs/full',      'oft_lr3e3'),
    ("base",       "base (no reg.)",                   "#8b8a85", 'logs/ref',       'gw_base'),
]

PAT = re.compile(r"^(\d+)\s+success\s+([\d.]+)")


def load(d, prefix):
    """Return (steps, curves) -- curves is a (seeds, points) array of success."""
    curves = []
    for s in SEEDS:
        path = f"{d}/{prefix}_{s}.log"
        vals, steps = [], []
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                m = PAT.match(line.strip())
                if m and int(m.group(1)) < NUM_STEPS:
                    steps.append(int(m.group(1)))
                    vals.append(float(m.group(2)))
        curves.append(np.asarray(vals, dtype=float))
    L = min(len(c) for c in curves)
    return np.asarray(steps[:L]), np.stack([c[:L] for c in curves])


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


def ema(x, span):
    """Standard EMA, alpha = 2/(span+1), bias-corrected."""
    x = np.asarray(x, float)
    a = 2.0 / (span + 1.0)
    out = np.empty_like(x)
    acc, w = 0.0, 0.0
    for i, v in enumerate(x):
        acc = a * v + (1 - a) * acc
        w = a + (1 - a) * w
        out[i] = acc / w
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


def mark_tasks(ax, upto=NUM_STEPS):
    for x in np.arange(CHANGE, upto, CHANGE):
        ax.axvline(x, color=INK_MUTED, linewidth=0.7, linestyle="--", alpha=0.30, zorder=0)


def fmt_steps(x, _):
    return f"{x/1e6:.1f}M" if x >= 1e6 else f"{x/1e3:.0f}k"


def main():
    os.makedirs(PLOT_DIR, exist_ok=True)
    data = {}
    for key, label, color, d, prefix in METHODS:
        steps, stack = load(d, prefix)
        data[key] = (steps, stack)
        print(f"{label:<30} seeds={stack.shape[0]}  points={stack.shape[1]}  mean={stack.mean():.3f}")

    # ---------- fig1: raw curve (light smoothing w=3, keeps sawtooth) ----------
    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    for key, label, color, *_ in METHODS:
        steps, stack = data[key]
        m = moving_avg(iqm(stack, 0), 3)
        ax.plot(steps, m, color=color, linewidth=1.7, label=label, zorder=3)
    mark_tasks(ax)
    style_axes(ax)
    ax.set_ylim(-0.03, 1.03); ax.set_xlim(0, NUM_STEPS)
    ax.xaxis.set_major_formatter(FuncFormatter(fmt_steps))
    ax.set_xlabel("Environment steps (dashed = task switch)")
    ax.set_ylabel("Success rate")
    ax.set_title("Gridworld Nine-Rooms — six methods (light smoothing w=3)",
                 loc="left", color=INK_PRIMARY, fontweight="bold", pad=10)
    ax.legend(frameon=False, loc="upper center", ncol=3, fontsize=9.5,
              bbox_to_anchor=(0.5, -0.17))
    out1 = f"{PLOT_DIR}/gridworld_all_curve_raw.png"
    fig.savefig(out1); plt.close(fig); print("wrote", out1)

    # ---------- fig2: EMA smoothing, three spans ----------
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.0), sharey=True)
    for ax, span in zip(axes, (5, 16, 40)):
        for key, label, color, *_ in METHODS:
            steps, stack = data[key]
            ax.plot(steps, ema(iqm(stack, 0), span), color=color, linewidth=1.9,
                    label=label, zorder=3)
        mark_tasks(ax)
        style_axes(ax)
        ax.set_ylim(-0.03, 1.03); ax.set_xlim(0, NUM_STEPS)
        ax.xaxis.set_major_formatter(FuncFormatter(fmt_steps))
        note = {5: "~half a task", 16: "~2 tasks", 40: "~5 tasks"}[span]
        ax.set_title(f"EMA span={span} ({note})", loc="left",
                     color=INK_SECONDARY, fontsize=10.5, pad=8)
        ax.set_xlabel("Environment steps")
    axes[0].set_ylabel("Success rate")
    fig.suptitle("Gridworld — EMA smoothing comparison (larger span = flatter, but erases task structure)",
                 x=0.005, ha="left", color=INK_PRIMARY, fontweight="bold", fontsize=12.5)
    axes[1].legend(frameon=False, loc="upper center", ncol=3, fontsize=9.5,
                   bbox_to_anchor=(0.5, -0.19))
    out2 = f"{PLOT_DIR}/gridworld_all_ema.png"
    fig.savefig(out2); plt.close(fig); print("wrote", out2)

    # ---------- fig3: per-task trend (one point per task, drop boundary) ----------
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    summary = {}
    for key, label, color, *_ in METHODS:
        steps, stack = data[key]
        n_task = stack.shape[1] // PTS_PER_TASK
        per_task = []
        for t in range(n_task):
            seg = stack[:, t * PTS_PER_TASK:(t + 1) * PTS_PER_TASK]
            per_task.append(seg[:, :-1].mean(axis=1))   # drop last point (boundary artifact)
        per_task = np.stack(per_task, axis=1)           # (seeds, tasks)
        m = iqm(per_task, 0)
        lo = np.percentile(per_task, 25, axis=0)
        hi = np.percentile(per_task, 75, axis=0)
        xs = np.arange(1, n_task + 1)
        ax.plot(xs, m, color=color, linewidth=2.1, marker="o", markersize=3.6,
                label=label, zorder=3)
        ax.fill_between(xs, lo, hi, color=color, alpha=0.12, linewidth=0, zorder=2)
        summary[label] = m
    style_axes(ax)
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlim(0.5, 20.5); ax.set_xticks(range(1, 21, 1))
    ax.set_xlabel("Task index (40k steps per task)")
    ax.set_ylabel("Mean success rate")
    ax.set_title("Gridworld — per-task trend (plasticity loss at a glance)",
                 loc="left", color=INK_PRIMARY, fontweight="bold", pad=10)
    ax.legend(frameon=False, loc="upper center", ncol=3, fontsize=9.5,
              bbox_to_anchor=(0.5, -0.17))
    out3 = f"{PLOT_DIR}/gridworld_per_task_trend.png"
    fig.savefig(out3); plt.close(fig); print("wrote", out3)

    # ---------- per-task numbers, for the record ----------
    print("\nPer-task success (mean of tasks 1-5 vs 16-20, boundary dropped):")
    print(f"{'Method':<30} {'tasks 1-5':>10} {'tasks 16-20':>12} {'decay':>8}")
    print("-" * 64)
    for label, m in summary.items():
        a, b = m[:5].mean(), m[-5:].mean()
        print(f"{label:<30} {a:>10.3f} {b:>12.3f} {b-a:>+8.3f}")


if __name__ == "__main__":
    main()
