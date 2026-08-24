"""Publication-quality figures for the Parseval reproduction.

Loads the saved pkl results and produces clean learning-curve and
performance-profile PNGs (light mode, colorblind-safe categorical palette).

Metrics: gridworld/metaworld use success rate; CARL envs use return.

Usage:
    python make_figures.py
"""

import os
import pickle

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import scipy.stats

# ---- palette (documented reference palette, light mode) ----
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
AXIS = "#c3c2b7"

COLORS = {"parseval": "#2a78d6", "base": "#eb6834"}
LABELS = {"parseval": "Parseval", "base": "Base"}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "DejaVu Sans"],
    "font.size": 11,
    "axes.edgecolor": AXIS,
    "axes.linewidth": 0.8,
    "axes.labelcolor": INK_SECONDARY,
    "xtick.color": INK_MUTED,
    "ytick.color": INK_MUTED,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
})

PLOT_DIR = "plots"
RESULTS = "results"

# env -> config
ENVS = {
    "gridworld": dict(
        name="gridworld_ninerooms", metric="mean_eval_success",
        save_freq=5000, change_freq=40000, num_steps=800000,
        title="Gridworld", ylabel="Success rate",
    ),
    "metaworld": dict(
        name="metaworld_sequence_set0", metric="mean_eval_success",
        save_freq=25000, change_freq=1000000, num_steps=5000000,
        title="MetaWorld", ylabel="Success rate",
    ),
    "lunarlander": dict(
        name="carl_sequence_lunarlander_0", metric="mean_eval_return",
        save_freq=25000, change_freq=500000, num_steps=1000000,
        title="CARL-LunarLander", ylabel="Average return",
    ),
}


# ---------- data loading ----------
def load_metric(env, alg, seeds):
    cfg = ENVS[env]
    curves = []
    for s in seeds:
        path = f"{RESULTS}/data_{cfg['name']}_{alg}_{s}.pkl"
        with open(path, "rb") as f:
            d = pickle.load(f)
        curves.append(np.asarray(d[cfg["metric"]], dtype=float))
    return np.stack(curves)


def iqm(x, axis=0):
    return scipy.stats.trim_mean(x, 0.25, axis)


def iqm_ci(x, axis=0):
    lo, hi = scipy.stats.mstats.trimmed_mean_ci(x, limits=(0.25, 0.25), axis=axis)
    return lo, hi


def smooth(x, w):
    x = np.asarray(x, dtype=float)
    if w <= 1:
        return x
    out = np.empty_like(x)
    csum = np.cumsum(np.insert(x, 0, 0))
    for i in range(len(x)):
        lo = max(0, i - w + 1)
        out[i] = (csum[i + 1] - csum[lo]) / (i + 1 - lo)
    return out


def task_averages(curve, num_tasks):
    n = len(curve)
    seg = n // num_tasks
    out = []
    for t in range(num_tasks):
        chunk = curve[t * seg:(t + 1) * seg]
        out.append(np.mean(chunk[len(chunk) // 2:]))
    return np.asarray(out)


# ---------- styling helpers ----------
def style_axes(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
    ax.grid(True, color=GRIDLINE, linewidth=0.7, alpha=0.9)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def mark_task_changes(ax, change_freq, num_steps, color=INK_MUTED):
    for x in np.arange(change_freq, num_steps, change_freq):
        ax.axvline(x, color=color, linewidth=0.7, linestyle="--", alpha=0.35, zorder=0)


def fmt_millions(x, _):
    return f"{x / 1e6:.0f}M" if x >= 1e6 else f"{x / 1e3:.0f}k"


# ---------- learning curve ----------
def plot_learning_curve(env, algs, seeds, smooth_w=5):
    cfg = ENVS[env]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for alg in algs:
        curves = load_metric(env, alg, seeds)
        m = smooth(iqm(curves, axis=0), smooth_w)
        lo = smooth(iqm_ci(curves, axis=0)[0], smooth_w)
        hi = smooth(iqm_ci(curves, axis=0)[1], smooth_w)
        xs = cfg["save_freq"] * np.arange(len(m))
        ax.plot(xs, m, color=COLORS[alg], linewidth=2.0, label=LABELS[alg], zorder=3)
        ax.fill_between(xs, lo, hi, color=COLORS[alg], alpha=0.14, linewidth=0, zorder=2)

    mark_task_changes(ax, cfg["change_freq"], cfg["num_steps"])
    style_axes(ax)
    if cfg["metric"] == "mean_eval_success":
        ax.set_ylim(-0.03, 1.03)
    ax.set_xlim(0, cfg["num_steps"])
    ax.xaxis.set_major_formatter(FuncFormatter(fmt_millions))
    ax.set_xlabel("Environment steps")
    ax.set_ylabel(cfg["ylabel"])
    ax.set_title(cfg["title"], loc="left", color=INK_PRIMARY, fontweight="bold", pad=10)
    ax.legend(frameon=False, loc="lower right", ncol=2)
    out = f"{PLOT_DIR}/{env}_learning_curve.png"
    fig.savefig(out)
    plt.close(fig)
    print("wrote", out)


# ---------- performance profile ----------
def plot_performance_profile(env, algs, seeds):
    cfg = ENVS[env]
    num_tasks = int(cfg["num_steps"] // cfg["change_freq"])
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for alg in algs:
        curves = load_metric(env, alg, seeds)
        vals = np.concatenate([task_averages(c, num_tasks) for c in curves])
        x = np.sort(vals)
        y = 1.0 - np.arange(len(x)) / float(len(x))
        ax.plot(x, y, color=COLORS[alg], linewidth=2.0, label=LABELS[alg], zorder=3)
        err = np.sqrt(np.log(2 / 0.1) / (2 * len(x)))
        ax.fill_between(x, np.clip(y - err, 0, 1), np.clip(y + err, 0, 1),
                        color=COLORS[alg], alpha=0.14, linewidth=0, zorder=2)

    style_axes(ax)
    if cfg["metric"] == "mean_eval_success":
        ax.set_xlim(-0.03, 1.03)
        ax.set_xlabel("Average success rate")
        ax.set_ylabel("Pr(Success rate > x)")
    else:
        ax.set_xlabel("Average return")
        ax.set_ylabel("Pr(Return > x)")
    ax.set_ylim(-0.03, 1.03)
    ax.set_title(cfg["title"], loc="left", color=INK_PRIMARY, fontweight="bold", pad=10)
    ax.legend(frameon=False, loc="upper right", ncol=1)
    out = f"{PLOT_DIR}/{env}_performance_profile.png"
    fig.savefig(out)
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    os.makedirs(PLOT_DIR, exist_ok=True)
    seeds = list(range(6))
    for env in ("gridworld", "metaworld", "lunarlander"):
        plot_learning_curve(env, ["base", "parseval"], seeds)
        plot_performance_profile(env, ["base", "parseval"], seeds)
