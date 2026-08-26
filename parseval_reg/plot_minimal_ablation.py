"""最简 Pion 消融：① 标准 Pion vs 最简 Pion 成功率对比（柱状图）；
② 最简 Pion 下 4 初始化的 stable rank 演化（隐藏层）。
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
INITS = ["xavier", "orthogonal", "standard", "identity"]
PAT = re.compile(r'^(\d+) success ([\d.]+)')


def success(logfmt, seeds=range(6)):
    cleans = []
    for s in seeds:
        pts = []
        for line in open(logfmt.format(s=s), encoding='utf-8', errors='ignore'):
            m = PAT.match(line.strip())
            if m and int(m.group(1)) < NUM_STEPS:
                pts.append((int(m.group(1)), float(m.group(2))))
        clean = [v for st, v in pts if st % CF != CF - 1]
        cleans.append(np.mean(clean))
    return np.array(cleans)


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


def load_stable_rank(init):
    arrs = []
    for s in range(6):
        with open(f"results/data_metaworld_sequence_set0_pion_minimal_{s}_minimal_init_{init}.pkl", "rb") as f:
            d = pickle.load(f)
        arrs.append(np.asarray(d["actor_matrix_stable_rank"], dtype=float)[:, 1])  # 隐藏层
    L = min(len(a) for a in arrs)
    return np.stack([a[:L] for a in arrs])


# ---- 图 1：成功率对比柱状图 ----
std_means = [success(f"logs/mw_init_ppo/pion_{i}_" + "{s}.log").mean() for i in INITS]
std_std = [success(f"logs/mw_init_ppo/pion_{i}_" + "{s}.log").std() for i in INITS]
min_means = [success(f"logs/mw_pion_minimal_init/pion_minimal_{i}_" + "{s}.log").mean() for i in INITS]
min_std = [success(f"logs/mw_pion_minimal_init/pion_minimal_{i}_" + "{s}.log").std() for i in INITS]

fig, ax = plt.subplots(figsize=(7.6, 4.6))
x = np.arange(len(INITS))
w = 0.36
ax.bar(x - w/2, std_means, w, yerr=std_std, capsize=3, color="#1f77b4", label="standard Pion (additive)")
ax.bar(x + w/2, min_means, w, yerr=min_std, capsize=3, color="#ff7f0e", label="minimal Pion (multiplicative)")
ax.axhline(0.618, color="0.4", linewidth=1.2, linestyle="--")
ax.text(3.45, 0.628, "base = 0.618", color="0.35", fontsize=9)
ax.set_xticks(x); ax.set_xticklabels(INITS)
ax.set_ylabel("Success rate (boundary-removed)")
ax.set_ylim(0, 1.0)
ax.set_title("MetaWorld (PPO) — standard vs minimal Pion", loc="left", fontweight="bold", pad=10)
ax.legend(frameon=False, loc="lower right")
ax.grid(axis="y", color="0.9", alpha=0.9); ax.set_axisbelow(True)
fig.savefig("plots/minimal_ablation_success.png"); plt.close(fig)
print("wrote plots/minimal_ablation_success.png")

# ---- 图 2：最简 Pion 4 初始化 stable rank（隐藏层）----
fig, ax = plt.subplots(figsize=(8.5, 4.6))
colors = {"xavier": "#1f77b4", "orthogonal": "#ff7f0e", "standard": "#2ca02c", "identity": "#d62728"}
for init in INITS:
    sr = load_stable_rank(init)
    ax.plot(SF * np.arange(sr.shape[1]), ma(iqm(sr, 0), 3), color=colors[init], linewidth=2.0, label=init)
for x0 in np.arange(CF, NUM_STEPS, CF):
    ax.axvline(x0, color="0.75", linewidth=0.7, linestyle="--", alpha=0.6, zorder=0)
ax.set_xlim(0, NUM_STEPS)
ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x/1e6:.0f}M"))
ax.set_xlabel("Environment steps (dashed = task switch)")
ax.set_ylabel("Stable rank (hidden 64x64)")
ax.set_title("Minimal Pion — per-init stable rank (multiplicative, spectrum-preserving)",
             loc="left", fontweight="bold", pad=10)
ax.legend(frameon=False, loc="upper right")
ax.grid(True, color="0.9", alpha=0.9); ax.set_axisbelow(True)
fig.savefig("plots/minimal_ablation_stable_rank.png"); plt.close(fig)
print("wrote plots/minimal_ablation_stable_rank.png")
