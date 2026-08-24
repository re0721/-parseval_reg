"""MetaWorld: Pion with orthogonal vs xavier initialization (success curve).

Cross-environment check of the Gridworld finding that a flat full-rank spectrum
(orthogonal) does NOT collapse Pion — it ties with a shaped spectrum (xavier).
Plain matplotlib style.
"""
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
pat = re.compile(r'^(\d+) success ([\d.]+)')


def load(logfmt, seeds=range(6)):
    curves, steps = [], None
    for s in seeds:
        pts = []
        for line in open(logfmt.format(s=s), encoding='utf-8', errors='ignore'):
            m = pat.match(line.strip())
            if m and int(m.group(1)) < NUM_STEPS:
                pts.append((int(m.group(1)), float(m.group(2))))
        curves.append(np.array([v for _, v in pts]))
        if steps is None:
            steps = np.array([p for p, _ in pts])
    L = min(len(c) for c in curves)
    return steps[:L], np.stack([c[:L] for c in curves])


def iqm(x):
    return scipy.stats.trim_mean(x, 0.25, axis=0)


def ma(x, w):
    x = np.asarray(x, float)
    out = np.empty_like(x)
    c = np.cumsum(np.insert(x, 0, 0))
    for i in range(len(x)):
        lo = max(0, i - w + 1)
        out[i] = (c[i + 1] - c[lo]) / (i + 1 - lo)
    return out


def main():
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    for name, logfmt, color in [
        ("orthogonal init (flat, rank 64)", "logs/mw_orthogonal_10task/pion_orthogonal_{s}.log", "#ff7f0e"),
        ("xavier init (shaped, rank ~17.7)", "logs/mw10/pion_{s}.log", "#1f77b4"),
    ]:
        steps, stack = load(logfmt)
        ax.plot(steps, ma(iqm(stack), 5), color=color, linewidth=2.0, label=name, zorder=3)
    for x in np.arange(CF, NUM_STEPS, CF):
        ax.axvline(x, color="0.75", linewidth=0.7, linestyle="--", alpha=0.6, zorder=0)
    ax.set_ylim(0.0, 1.03); ax.set_xlim(0, NUM_STEPS)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x/1e6:.0f}M"))
    ax.set_xlabel("Environment steps (dashed = task switch)")
    ax.set_ylabel("Success rate (IQM, w=5)")
    ax.set_title("MetaWorld — Pion: orthogonal vs xavier init (tie)",
                 loc="left", fontweight="bold", pad=10)
    ax.legend(frameon=False, loc="lower right")
    ax.grid(True, color="0.9", linewidth=0.7, alpha=0.9); ax.set_axisbelow(True)
    fig.savefig("plots/init_comparison_metaworld.png")
    plt.close(fig)
    print("wrote plots/init_comparison_metaworld.png")


if __name__ == "__main__":
    main()
