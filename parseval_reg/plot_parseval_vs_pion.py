"""Parseval vs Pion on gridworld: learning curve + performance profile.

Parseval is read from its intact pkl; Pion is reconstructed from the
complete original logs (logs/gw_pion_*.log), since the pion pkl was
overwritten by the later killed v3 run.
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

# ---- palette (match plot_lie_comparison.py) ----
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
AXIS = "#c3c2b7"

COLORS = {"parseval": "#1baf7a", "pion": "#2a78d6"}
LABELS = {"parseval": "Parseval", "pion": "Pion"}

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
SAVE, CHANGE, NUM_STEPS = 5000, 40000, 800000
SEEDS = list(range(6))


def load_parseval():
    curves = []
    for s in SEEDS:
        with open(f"results/data_gridworld_ninerooms_parseval_{s}.pkl", "rb") as f:
            d = pickle.load(f)
        curves.append(np.asarray(d["mean_eval_success"], dtype=float))
    return curves


def load_pion_from_logs():
    # 使用 8/19 的 "c" 版 (lr=1e-3, xavier init) —— 即 gridworld_lie_comparison.png 里用的好 pion。
    # 原始 gw_pion (lr=2.5e-4, orthogonal) 只有 ~12% 成功率，是失败的参数，不用。
    curves = []
    pat = re.compile(r"\d+\s+success\s+([\d.]+)")
    for s in SEEDS:
        with open(f"logs/gw_pion_c_{s}.log", "r", encoding="utf-8", errors="ignore") as f:
            vals = [float(m.group(1)) for line in f if (m := pat.search(line))]
        curves.append(np.asarray(vals, dtype=float))
    return curves


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


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
    ax.grid(True, color=GRIDLINE, linewidth=0.7, alpha=0.9)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def mark_task_changes(ax):
    for x in np.arange(CHANGE, NUM_STEPS, CHANGE):
        ax.axvline(x, color=INK_MUTED, linewidth=0.7, linestyle="--", alpha=0.35, zorder=0)


def fmt_millions(x, _):
    return f"{x / 1e6:.0f}M" if x >= 1e6 else f"{x / 1e3:.0f}k"


def align(curves):
    L = min(len(c) for c in curves)
    return [c[:L] for c in curves], L


def learning_curve(curves, smooth_w=5):
    curves, L = align(curves)
    stack = np.stack(curves)
    m = smooth(iqm(stack, axis=0), smooth_w)
    lo = smooth(iqm_ci(stack, axis=0)[0], smooth_w)
    hi = smooth(iqm_ci(stack, axis=0)[1], smooth_w)
    xs = SAVE * np.arange(L)
    return xs, m, lo, hi, L


def main():
    os.makedirs(PLOT_DIR, exist_ok=True)
    data = {
        "parseval": load_parseval(),
        "pion": load_pion_from_logs(),
    }
    for k, v in data.items():
        lens = [len(c) for c in v]
        print(f"{k}: seeds lens = {lens}  (mean_succ={np.mean([c.mean() for c in v]):.3f})")

    # ---- learning curve (success vs steps) ----
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for k in ("parseval", "pion"):
        xs, m, lo, hi, L = learning_curve(data[k])
        ax.plot(xs, m, color=COLORS[k], linewidth=2.0, label=LABELS[k], zorder=3)
        ax.fill_between(xs, lo, hi, color=COLORS[k], alpha=0.14, linewidth=0, zorder=2)
    mark_task_changes(ax)
    style_axes(ax)
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlim(0, NUM_STEPS)
    ax.xaxis.set_major_formatter(FuncFormatter(fmt_millions))
    ax.set_xlabel("Environment steps")
    ax.set_ylabel("Success rate")
    ax.set_title("Gridworld — Parseval vs Pion", loc="left",
                 color=INK_PRIMARY, fontweight="bold", pad=10)
    ax.legend(frameon=False, loc="lower right")
    out = f"{PLOT_DIR}/gridworld_parseval_vs_pion_learning_curve.png"
    fig.savefig(out)
    plt.close(fig)
    print("wrote", out)

    # ---- performance profile ----
    PTS = CHANGE // SAVE
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for k in ("parseval", "pion"):
        curves, L = align(data[k])
        nt = L // PTS
        vals = np.concatenate([
            np.asarray([c[t * PTS:(t + 1) * PTS][PTS // 2:].mean() for t in range(nt)])
            for c in curves
        ])
        x = np.sort(vals)
        y = 1.0 - np.arange(len(x)) / float(len(x))
        ax.plot(x, y, color=COLORS[k], linewidth=2.0, label=LABELS[k], zorder=3)
        err = np.sqrt(np.log(2 / 0.1) / (2 * len(x)))
        ax.fill_between(x, np.clip(y - err, 0, 1), np.clip(y + err, 0, 1),
                        color=COLORS[k], alpha=0.14, linewidth=0, zorder=2)
    style_axes(ax)
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("Average success rate")
    ax.set_ylabel("Pr(Success rate > x)")
    ax.set_title("Gridworld — Parseval vs Pion", loc="left",
                 color=INK_PRIMARY, fontweight="bold", pad=10)
    ax.legend(frameon=False, loc="upper right")
    out = f"{PLOT_DIR}/gridworld_parseval_vs_pion_profile.png"
    fig.savefig(out)
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    main()
