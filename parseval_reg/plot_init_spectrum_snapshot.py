"""单方法奇异值谱快照：Pion（xavier init，PPO）。

在几个时间点，把每层所有奇异值从大到小排序，画成一条下降的谱曲线。
看谱随训练怎么变：
  塌缩 = 曲线变陡（σ1 翘起、其余贴地，即 1 个主导方向）
  健康 = 曲线平缓（多个均匀方向）
"""
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "font.size": 11, "savefig.dpi": 200, "savefig.bbox": "tight",
})

SEEDS = range(6)
PKL = "results/data_metaworld_sequence_set0_pion_{s}_init_xavier_ppo.pkl"
LAYERS = ["input 39->64", "hidden 64->64", "output 64->4"]
# (eval 点索引, 步数标签)  每个任务 40 个 eval 点（100万步/2.5万步）
TIMEPOINTS = [
    (0,   "task 1 start (~25k steps)"),
    (200, "task 5 (~5M steps)"),
    (399, "task 10 end (~10M steps)"),
]


def load_snapshot(t, layer):
    vals = []
    for s in SEEDS:
        with open(PKL.format(s=s), "rb") as f:
            d = pickle.load(f)
        vals.append(np.asarray(d["actor_param_singular_values"][t][layer], dtype=float))
    return np.mean(vals, axis=0)  # 6 seed 均值


fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
for li, ax in enumerate(axes):
    for t, label in TIMEPOINTS:
        spec = load_snapshot(t, li)
        ax.plot(np.arange(1, len(spec) + 1), spec, marker="o", markersize=2.2, label=label)
    ax.set_xlabel("singular value index (sorted descending)")
    ax.set_ylim(bottom=0)
    ax.set_title(LAYERS[li], loc="left", fontsize=11)
    ax.grid(True, color="0.9", alpha=0.8); ax.set_axisbelow(True)
axes[0].set_ylabel("singular value")
axes[1].legend(frameon=False, loc="upper right", fontsize=9)
fig.suptitle("Pion (xavier init, PPO) — singular value spectrum snapshot",
             x=0.01, ha="left", fontweight="bold", fontsize=12)
fig.tight_layout()
fig.savefig("plots/init_spectrum_snapshot.png")
plt.close(fig)
print("wrote plots/init_spectrum_snapshot.png")
