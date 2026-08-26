"""初始化消融的 3 张图：成功率 / 奇异值演化 / 每层 stable rank。

数据来源：
  成功率       -> logs/mw_init_ppo/pion_{init}_{s}.log（success 行）
  奇异值       -> results/*.pkl 的 actor_param_singular_values（每层完整谱）
  每层 stable rank -> results/*.pkl 的 actor_matrix_stable_rank（3 层）

6 seed 用 IQM（trim_mean 0.25）聚合，成功率/stable rank 做轻平滑。
"""
import re
import pickle
import numpy as np
import scipy.stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "font.size": 11, "savefig.dpi": 200, "savefig.bbox": "tight",
})

CF, SF, NUM_STEPS = 1_000_000, 25_000, 10_000_000
SEEDS = range(6)
PKL_ENV = "metaworld_sequence_set0"
INITS = [
    ("xavier", "xavier", "#1f77b4"),
    ("orthogonal", "orthogonal", "#ff7f0e"),
    ("standard", "standard", "#2ca02c"),
    ("identity", "identity", "#d62728"),
]
LAYER_NAMES = ["input 39->64", "hidden 64->64", "output 64->4"]
PAT = re.compile(r'^(\d+) success ([\d.]+)')


def iqm(x, axis=0):
    return scipy.stats.trim_mean(x, 0.25, axis)


def ma(x, w):
    x = np.asarray(x, float)
    out = np.empty_like(x)
    c = np.cumsum(np.insert(x, 0, 0))
    for i in range(len(x)):
        lo = max(0, i - w + 1)
        out[i] = (c[i + 1] - c[lo]) / (i + 1 - lo)
    return out


def load_success(init):
    curves = []
    for s in SEEDS:
        pts = []
        for line in open(f"logs/mw_init_ppo/pion_{init}_{s}.log", encoding="utf-8", errors="ignore"):
            m = PAT.match(line.strip())
            if m and int(m.group(1)) < NUM_STEPS:
                pts.append((int(m.group(1)), float(m.group(2))))
        curves.append([v for _, v in pts])
    L = min(len(c) for c in curves)
    steps = [p[0] for p in pts[:L]]
    return np.array(steps), np.stack([c[:L] for c in curves])


def load_singular_values(init, layer):
    # 返回 (seeds, points, n_sv)，n_sv 是该层奇异值个数（各 seed 一致）
    per_seed = []
    for s in SEEDS:
        with open(f"results/data_{PKL_ENV}_pion_{s}_init_{init}_ppo.pkl", "rb") as f:
            d = pickle.load(f)
        sv_list = d["actor_param_singular_values"]  # list of 400, each list of 3 layers
        arr = np.stack([np.asarray(pt[layer], dtype=float) for pt in sv_list])  # (points, n_sv)
        per_seed.append(arr)
    L = min(a.shape[0] for a in per_seed)
    return np.stack([a[:L] for a in per_seed])  # (seeds, points, n_sv)


def load_stable_rank(init):
    per_seed = []
    for s in SEEDS:
        with open(f"results/data_{PKL_ENV}_pion_{s}_init_{init}_ppo.pkl", "rb") as f:
            d = pickle.load(f)
        per_seed.append(np.asarray(d["actor_matrix_stable_rank"], dtype=float))
    L = min(a.shape[0] for a in per_seed)
    return np.stack([a[:L] for a in per_seed])  # (seeds, points, 3 layers)


def mark_tasks(ax):
    for x in np.arange(CF, NUM_STEPS, CF):
        ax.axvline(x, color="0.75", linewidth=0.7, linestyle="--", alpha=0.6, zorder=0)


def fmt(x, _):
    return f"{x/1e6:.0f}M"


# ---- fig1: success curve ----
fig, ax = plt.subplots(figsize=(8.5, 4.6))
for init, label, color in INITS:
    steps, stack = load_success(init)
    ax.plot(steps, ma(iqm(stack), 5), color=color, linewidth=2.0, label=label, zorder=3)
mark_tasks(ax)
ax.set_ylim(0, 1.03); ax.set_xlim(0, NUM_STEPS)
ax.xaxis.set_major_formatter(FuncFormatter(fmt))
ax.set_xlabel("Environment steps (dashed = task switch)")
ax.set_ylabel("Success rate (IQM, w=5)")
ax.set_title("MetaWorld (PPO) — Pion, 4 initializations", loc="left", fontweight="bold", pad=10)
ax.legend(frameon=False, loc="lower right")
ax.grid(True, color="0.9", alpha=0.9); ax.set_axisbelow(True)
fig.savefig("plots/init_spectrum_success.png"); plt.close(fig)
print("wrote plots/init_spectrum_success.png")

# ---- fig2: singular values over time (top-5 per layer) ----
fig, axes = plt.subplots(1, 3, figsize=(15, 4.0))
for li, ax in enumerate(axes):
    for init, label, color in INITS:
        sv = load_singular_values(init, li)  # (seeds, points, n_sv)
        top5 = iqm(sv, 0)                    # (points, n_sv) IQM across seeds
        n_sv = top5.shape[1]
        for k in range(min(5, n_sv)):
            ax.plot(SF * np.arange(top5.shape[0]), top5[:, k],
                    color=color, linewidth=1.6 if k == 0 else 0.9,
                    alpha=1.0 if k == 0 else 0.45)
    mark_tasks(ax)
    ax.set_xlim(0, NUM_STEPS); ax.set_ylim(bottom=0)
    ax.xaxis.set_major_formatter(FuncFormatter(fmt))
    ax.set_xlabel("Environment steps")
    ax.set_title(f"{LAYER_NAMES[li]} — top-5 singular values", loc="left", fontsize=10.5)
axes[0].set_ylabel("Singular value")
axes[1].legend([l for _, l, _ in INITS], frameon=False, loc="upper right", fontsize=8.5)
fig.suptitle("MetaWorld (PPO) — Pion singular values (solid=σ1, faded=σ2..σ5)", x=0.01, ha="left", fontweight="bold", fontsize=12)
fig.savefig("plots/init_spectrum_singular_values.png"); plt.close(fig)
print("wrote plots/init_spectrum_singular_values.png")

# ---- fig3: per-layer stable rank ----
fig, axes = plt.subplots(1, 3, figsize=(15, 4.0))
for li, ax in enumerate(axes):
    for init, label, color in INITS:
        sr = load_stable_rank(init)  # (seeds, points, 3)
        ax.plot(SF * np.arange(sr.shape[1]), ma(iqm(sr[:, :, li], 0), 3),
                color=color, linewidth=2.0, label=label)
    mark_tasks(ax)
    ax.set_xlim(0, NUM_STEPS)
    ax.xaxis.set_major_formatter(FuncFormatter(fmt))
    ax.set_xlabel("Environment steps")
    ax.set_title(f"{LAYER_NAMES[li]}", loc="left", fontsize=10.5)
    ax.set_ylim(bottom=0)
axes[0].set_ylabel("Stable rank")
axes[1].legend(frameon=False, loc="upper right", fontsize=8.5)
fig.suptitle("MetaWorld (PPO) — Pion per-layer stable rank", x=0.01, ha="left", fontweight="bold", fontsize=12)
fig.savefig("plots/init_spectrum_stable_rank.png"); plt.close(fig)
print("wrote plots/init_spectrum_stable_rank.png")
