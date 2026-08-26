"""最简 Pion 4 初始化的成功率曲线（横轴步数）。"""
import re
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

CF, NUM_STEPS = 1_000_000, 10_000_000
INITS = [
    ("xavier", "xavier", "#1f77b4"),
    ("orthogonal", "orthogonal", "#ff7f0e"),
    ("standard", "standard", "#2ca02c"),
    ("identity", "identity", "#d62728"),
]
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


def load(init):
    curves = []
    for s in range(6):
        pts = []
        for line in open(f"logs/mw_pion_minimal_init/pion_minimal_{init}_{s}.log", encoding="utf-8", errors="ignore"):
            m = PAT.match(line.strip())
            if m and int(m.group(1)) < NUM_STEPS:
                pts.append((int(m.group(1)), float(m.group(2))))
        curves.append([v for _, v in pts])
    L = min(len(c) for c in curves)
    steps = [p[0] for p in pts[:L]]
    return np.array(steps), np.stack([c[:L] for c in curves])


fig, ax = plt.subplots(figsize=(9, 4.6))
for init, label, color in INITS:
    steps, stack = load(init)
    ax.plot(steps, ma(iqm(stack, 0), 5), color=color, linewidth=2.0, label=label, zorder=3)
for x0 in np.arange(CF, NUM_STEPS, CF):
    ax.axvline(x0, color="0.75", linewidth=0.7, linestyle="--", alpha=0.6, zorder=0)
ax.set_ylim(0, 1.03); ax.set_xlim(0, NUM_STEPS)
ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x/1e6:.0f}M"))
ax.set_xlabel("Environment steps (dashed = task switch)")
ax.set_ylabel("Success rate (IQM, w=5)")
ax.set_title("Minimal Pion — 4 initializations success curve", loc="left", fontweight="bold", pad=10)
ax.legend(frameon=False, loc="lower right")
ax.grid(True, color="0.9", alpha=0.9); ax.set_axisbelow(True)
fig.savefig("plots/minimal_success_curve.png")
plt.close(fig)
print("wrote plots/minimal_success_curve.png")
