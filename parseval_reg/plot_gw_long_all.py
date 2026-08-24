"""Gridworld 80k/task (long training): success curve, per-task trend, stable rank.

Plain research style (default matplotlib). Success from logs, stable rank from pkl.
Boundary eval point of each task (step % 80000 == 79999) is dropped in per-task/stable-rank.
"""
import re
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scipy.stats

PAT = re.compile(r'^(\d+) success ([\d.]+)')
CF, SF, NUM_STEPS = 80000, 5000, 1600000
PTS_PER_TASK = CF // SF  # 16
SEEDS = range(6)
METHODS = [
    ("pion",       "Pion",       "#2a78d6"),
    ("poet_exact", "POET-exact", "#7b4fd6"),
    ("parseval",   "Parseval",   "#1baf7a"),
    ("base",       "base",       "#8b8a85"),
]
LOGS_DIR = "logs/gw_long"
PKL_ENV = "gridworld_ninerooms"
PKL_SUFFIX = "long80k"


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


def load_success(prefix):
    curves, steps = [], []
    for s in SEEDS:
        ss, vv = [], []
        for line in open(f"{LOGS_DIR}/{prefix}_{s}.log", encoding="utf-8", errors="ignore"):
            m = PAT.match(line.strip())
            if m and int(m.group(1)) < NUM_STEPS:
                ss.append(int(m.group(1)))
                vv.append(float(m.group(2)))
        curves.append(vv)
        if not steps:
            steps = ss
    L = min(len(c) for c in curves)
    return np.array(steps[:L]), np.stack([c[:L] for c in curves])  # (points), (seeds, points)


def load_stable_rank(prefix):
    arrs = []
    for s in SEEDS:
        with open(f"results/data_{PKL_ENV}_{prefix}_{s}_{PKL_SUFFIX}.pkl", "rb") as f:
            d = pickle.load(f)
        arrs.append(np.asarray(d["actor_matrix_stable_rank"], dtype=float))
    return np.stack(arrs)  # (seeds, points, layers)


def mark_tasks(ax):
    for x in np.arange(CF, NUM_STEPS, CF):
        ax.axvline(x, color="#c3c2b7", linewidth=0.7, linestyle="--", alpha=0.3, zorder=0)


def style(ax):
    ax.grid(True, alpha=0.3)
    ax.set_axisbelow(True)


data = {m[0]: load_success(m[0]) for m in METHODS}

# ---- fig1: success curve ----
fig, ax = plt.subplots(figsize=(9, 4.6))
for key, label, color in METHODS:
    steps, stack = data[key]
    ax.plot(steps, moving_avg(iqm(stack, 0), 3), color=color, linewidth=1.7, label=label)
mark_tasks(ax); style(ax)
ax.set_ylim(-0.03, 1.03); ax.set_xlim(0, NUM_STEPS)
ax.set_xlabel("Environment steps (dashed = task switch)")
ax.set_ylabel("Success rate")
ax.set_title("Gridworld — success rate (80k steps/task, light smoothing w=3)", loc="left", fontweight="bold")
ax.legend(frameon=False, loc="upper center", ncol=4, bbox_to_anchor=(0.5, -0.14))
fig.tight_layout()
fig.savefig("plots/gw_long_curve.png", dpi=200); plt.close(fig)
print("wrote plots/gw_long_curve.png")

# ---- fig2: per-task trend ----
fig, ax = plt.subplots(figsize=(8.4, 4.6))
for key, label, color in METHODS:
    _, stack = data[key]
    n_task = stack.shape[1] // PTS_PER_TASK
    per_task = []
    for t in range(n_task):
        seg = stack[:, t * PTS_PER_TASK:(t + 1) * PTS_PER_TASK]
        per_task.append(seg[:, :-1].mean(axis=1))  # drop boundary
    arr = np.stack(per_task, axis=1)  # (seeds, tasks)
    xs = np.arange(1, n_task + 1)
    ax.plot(xs, iqm(arr, 0), color=color, linewidth=2.1, marker="o", markersize=3.6, label=label)
    ax.fill_between(xs, np.percentile(arr, 25, 0), np.percentile(arr, 75, 0), color=color, alpha=0.12, linewidth=0)
style(ax)
ax.set_ylim(-0.03, 1.03); ax.set_xlim(0.5, 20.5); ax.set_xticks(range(1, 21))
ax.set_xlabel("Task index (80k steps per task)")
ax.set_ylabel("Mean success rate")
ax.set_title("Gridworld — per-task trend (80k steps/task)", loc="left", fontweight="bold")
ax.legend(frameon=False, loc="upper center", ncol=4, bbox_to_anchor=(0.5, -0.14))
fig.tight_layout()
fig.savefig("plots/gw_long_per_task.png", dpi=200); plt.close(fig)
print("wrote plots/gw_long_per_task.png")

# ---- fig3: stable rank (two layers: input 225->64, hidden 64->64) ----
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=False)
for li, (ax, title) in enumerate(zip(axes, ["input layer 225x64", "hidden layer 64x64"])):
    for key, label, color in METHODS:
        v = load_stable_rank(key)[:, :, li]  # (seeds, points)
        xs = SF * np.arange(v.shape[1])
        ax.plot(xs, moving_avg(iqm(v, 0), 3), color=color, linewidth=1.8, label=label)
    mark_tasks(ax); style(ax)
    ax.set_xlim(0, NUM_STEPS)
    ax.set_xlabel("Environment steps")
    ax.set_ylabel("Stable rank")
    ax.set_title(title, loc="left", fontsize=11)
axes[1].legend(frameon=False, loc="upper right", fontsize=9)
fig.suptitle("Gridworld — actor stable rank (80k steps/task)", x=0.01, ha="left", fontweight="bold", fontsize=12)
fig.tight_layout()
fig.savefig("plots/gw_long_stable_rank.png", dpi=200); plt.close(fig)
print("wrote plots/gw_long_stable_rank.png")
