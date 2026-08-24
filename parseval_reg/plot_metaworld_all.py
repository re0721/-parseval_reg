"""MetaWorld 10-task comparison plots (English, same style as gridworld).

Three figures: (1) success curve, (2) per-task trend, (3) actor stable rank.
Boundary point: the last eval of each task (step % 1e6 == 999999) measures the
brand-new task, so it is dropped from per-task/stable-rank aggregation.
"""
import os
import re
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
CF, SF, NUM_STEPS = 1_000_000, 25_000, 10_000_000
SEEDS = range(6)

import sys
_DATA = sys.argv[1] if len(sys.argv) > 1 else "mw10"   # "mw10" 或 "mw10_shuffle"
if _DATA == "mw10_shuffle":
    LOGS_DIR = "logs/mw10_shuffle"
    PKL_ENV = "metaworld_sequence_shuffle"
    PKL_SUFFIX = "shuffle"
    OUT_PREFIX = "shuffle_"
else:
    LOGS_DIR = "logs/mw10"
    PKL_ENV = "metaworld_sequence_set0"
    PKL_SUFFIX = "10task"
    OUT_PREFIX = ""

METHODS = [
    ("pion",       "Pion (spectrum-preserving)",  "#2a78d6"),
    ("poet_exact", "POET-exact",                  "#7b4fd6"),
    ("parseval",   "Parseval (soft reg.)",        "#1baf7a"),
    ("base",       "base (no reg.)",              "#8b8a85"),
]
PAT = re.compile(r'^(\d+) success ([\d.]+)')


def load_logs(prefix):
    curves = []
    for s in SEEDS:
        pts = []
        for line in open(f"{LOGS_DIR}/{prefix}_{s}.log", encoding="utf-8", errors="ignore"):
            m = PAT.match(line.strip())
            if m and int(m.group(1)) < NUM_STEPS:
                pts.append((int(m.group(1)), float(m.group(2))))
        curves.append(pts)
    L = min(len(c) for c in curves)
    steps = np.array([c[L - 1][0] for c in curves])  # placeholder
    steps = np.array([p[0] for p in curves[0][:L]])
    return steps, np.stack([[v for _, v in c[:L]] for c in curves])


def is_boundary(step):
    return step % CF == CF - 1


def load_stable_rank(prefix, layer=1):
    arrs = []
    for s in SEEDS:
        with open(f"results/data_{PKL_ENV}_{prefix}_{s}_{PKL_SUFFIX}.pkl", "rb") as f:
            d = pickle.load(f)
        arrs.append(np.asarray(d["actor_matrix_stable_rank"], dtype=float)[:, layer])
    return np.stack(arrs)  # (seeds, points)


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


def mark_tasks(ax):
    for x in np.arange(CF, NUM_STEPS, CF):
        ax.axvline(x, color=INK_MUTED, linewidth=0.7, linestyle="--", alpha=0.30, zorder=0)


def fmt_steps(x, _):
    return f"{x/1e6:.0f}M"


def main():
    os.makedirs(PLOT_DIR, exist_ok=True)
    data = {m[0]: load_logs(m[0]) for m in METHODS}

    # ---- fig1: success curve (light smoothing) ----
    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    for key, label, color in METHODS:
        steps, stack = data[key]
        ax.plot(steps, moving_avg(iqm(stack, 0), 5), color=color, linewidth=1.7,
                label=label, zorder=3)
    mark_tasks(ax)
    style_axes(ax)
    ax.set_ylim(-0.03, 1.03); ax.set_xlim(0, NUM_STEPS)
    ax.xaxis.set_major_formatter(FuncFormatter(fmt_steps))
    ax.set_xlabel("Environment steps (dashed = task switch)")
    ax.set_ylabel("Success rate")
    ax.set_title("MetaWorld — four methods (10 tasks, light smoothing w=5)",
                 loc="left", color=INK_PRIMARY, fontweight="bold", pad=10)
    ax.legend(frameon=False, loc="upper center", ncol=4, fontsize=9.5,
              bbox_to_anchor=(0.5, -0.17))
    fig.savefig(f"{PLOT_DIR}/{OUT_PREFIX}metaworld_all_curve.png"); plt.close(fig)

    # ---- fig2: EMA smoothing, three spans ----
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.0), sharey=True)
    for ax, span in zip(axes, (5, 40, 80)):
        for key, label, color in METHODS:
            steps, stack = data[key]
            ax.plot(steps, ema(iqm(stack, 0), span), color=color, linewidth=1.9,
                    label=label, zorder=3)
        mark_tasks(ax)
        style_axes(ax)
        ax.set_ylim(-0.03, 1.03); ax.set_xlim(0, NUM_STEPS)
        ax.xaxis.set_major_formatter(FuncFormatter(fmt_steps))
        note = {5: "~1/8 task", 40: "~1 task", 80: "~2 tasks"}[span]
        ax.set_title(f"EMA span={span} ({note})", loc="left",
                     color=INK_SECONDARY, fontsize=10.5, pad=8)
        ax.set_xlabel("Environment steps")
    axes[0].set_ylabel("Success rate")
    fig.suptitle("MetaWorld — EMA smoothing comparison", x=0.005, ha="left",
                 color=INK_PRIMARY, fontweight="bold", fontsize=12.5)
    axes[1].legend(frameon=False, loc="upper center", ncol=4, fontsize=9.5,
                   bbox_to_anchor=(0.5, -0.19))
    fig.savefig(f"{PLOT_DIR}/{OUT_PREFIX}metaworld_all_ema.png"); plt.close(fig)

    # ---- fig2: per-task trend ----
    fig, ax = plt.subplots(figsize=(7.8, 4.6))
    for key, label, color in METHODS:
        _, stack = data[key]
        n_task = stack.shape[1] // (CF // SF)
        per_task = []
        for t in range(n_task):
            seg = stack[:, t * (CF // SF):(t + 1) * (CF // SF)]
            # drop last point of each task (boundary)
            per_task.append(seg[:, :-1].mean(axis=1))
        arr = np.stack(per_task, axis=1)  # (seeds, tasks)
        xs = np.arange(1, n_task + 1)
        ax.plot(xs, iqm(arr, 0), color=color, linewidth=2.1, marker="o", markersize=3.6,
                label=label, zorder=3)
        ax.fill_between(xs, np.percentile(arr, 25, 0), np.percentile(arr, 75, 0),
                        color=color, alpha=0.12, linewidth=0, zorder=2)
    style_axes(ax)
    ax.set_ylim(-0.03, 1.03); ax.set_xlim(0.5, 10.5); ax.set_xticks(range(1, 11))
    ax.set_xlabel("Task index (1M steps per task)")
    ax.set_ylabel("Mean success rate")
    ax.set_title("MetaWorld — per-task trend", loc="left", color=INK_PRIMARY,
                 fontweight="bold", pad=10)
    ax.legend(frameon=False, loc="upper center", ncol=4, fontsize=9.5,
              bbox_to_anchor=(0.5, -0.17))
    fig.savefig(f"{PLOT_DIR}/{OUT_PREFIX}metaworld_per_task.png"); plt.close(fig)

    # ---- fig3: stable rank (hidden layer 64x64) ----
    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    for key, label, color in METHODS:
        v = load_stable_rank(key)              # (seeds, points)
        xs = SF * np.arange(v.shape[1])
        ax.plot(xs, moving_avg(iqm(v, 0), 5), color=color, linewidth=2.0, label=label, zorder=3)
    mark_tasks(ax)
    style_axes(ax)
    ax.set_xlim(0, NUM_STEPS)
    ax.xaxis.set_major_formatter(FuncFormatter(fmt_steps))
    ax.set_ylabel("Stable rank (hidden layer 64x64)")
    ax.set_xlabel("Environment steps")
    ax.set_title("MetaWorld — actor stable rank", loc="left", color=INK_PRIMARY,
                 fontweight="bold", pad=10)
    ax.legend(frameon=False, loc="upper right", fontsize=9.5)
    fig.savefig(f"{PLOT_DIR}/{OUT_PREFIX}metaworld_stable_rank.png"); plt.close(fig)

    # ---- summary print ----
    print("wrote 3 plots")
    print("\nStable rank (hidden layer): first vs last task:")
    for key, label, color in METHODS:
        v = load_stable_rank(key)
        pts_per_task = CF // SF
        a = v[:, :pts_per_task][:, :-1].mean()
        b = v[:, -pts_per_task:][:, :-1].mean()
        print(f"  {label:<28} {a:>6.2f} -> {b:>6.2f}")


if __name__ == "__main__":
    main()
