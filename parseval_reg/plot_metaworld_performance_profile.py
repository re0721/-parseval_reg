"""MetaWorld 10-task: performance profile (success-rate fraction).

Follows the paper's Figure 14 style: x-axis is a success-rate threshold, y-axis
is the fraction of (seed, task) runs whose success rate exceeds that threshold.
A large drop near x=0 means many runs got stuck at ~0 success.
"""
import re
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

CF, SF, NUM_STEPS = 1_000_000, 25_000, 10_000_000
SEEDS = range(6)
N_TASK = NUM_STEPS // CF          # 10
PTS_PER_TASK = CF // SF           # 40
METHODS = [
    ("base", "base", "#8b8a85"),
    ("parseval", "Parseval", "#1baf7a"),
    ("pion", "Pion", "#2a78d6"),
    ("poet_exact", "POET-exact", "#7b4fd6"),
]
PAT = re.compile(r'^(\d+) success ([\d.]+)')


def load_task_success(prefix):
    """Collect per-(seed, task) mean success, dropping each task's last eval point."""
    vals = []
    for s in SEEDS:
        pts = []
        for line in open(f"logs/mw10/{prefix}_{s}.log", encoding="utf-8", errors="ignore"):
            m = PAT.match(line.strip())
            if m and int(m.group(1)) < NUM_STEPS:
                pts.append((int(m.group(1)), float(m.group(2))))
        for t in range(N_TASK):
            seg = [v for st, v in pts if t * CF <= st < (t + 1) * CF]
            if seg:
                vals.append(np.mean(seg[:-1]))  # drop boundary point
    return np.array(vals)  # (seeds * tasks,)


fig, ax = plt.subplots(figsize=(7.2, 4.6))
for prefix, label, color in METHODS:
    a = load_task_success(prefix)
    x = np.sort(a)
    y = 1 - np.arange(len(x)) / len(x)
    ax.plot(x, y, color=color, linewidth=2.0, label=label, zorder=3)
    # DKW 90% 置信带（和论文 Figure 14 一致）
    err = np.sqrt(1 / (2 * len(a)) * np.log(2 / 0.1))
    ax.fill_between(x, np.clip(y - err, 0, 1), np.clip(y + err, 0, 1),
                    color=color, alpha=0.18, zorder=2)
ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
ax.set_xlabel("Success rate threshold")
ax.set_ylabel("Pr(Success > threshold)")
ax.set_title("MetaWorld (PPO, tanh) — performance profile", loc="left", fontweight="bold", pad=10)
ax.legend(frameon=False, loc="lower left")
ax.grid(True, color="0.9", alpha=0.9); ax.set_axisbelow(True)
fig.savefig("plots/metaworld_performance_profile.png")
plt.close(fig)
print("wrote plots/metaworld_performance_profile.png")

# 打印每个方法卡在 0 的 run 比例
for prefix, label, _ in METHODS:
    a = load_task_success(prefix)
    frac_zero = (a < 0.05).mean()
    print(f"{label:<12} 成功率<0.05 的 run 占比 = {frac_zero:.3f}")
