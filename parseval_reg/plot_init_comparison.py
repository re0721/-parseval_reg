"""Initialization comparison: orthogonal vs xavier under Pion (Gridworld).

Two non-degenerate shapes of the *initial* spectrum that a spectrum-preserving
optimizer can preserve:

  orthogonal  -> flat full-rank (all singular values equal, stable rank 64)
  xavier      -> shaped Marchenko-Pastur (stable rank ~17.7)

Result: the two tie (raw 0.781 vs 0.759), i.e. Pion is robust to whether the
initial spectrum is flat or shaped, as long as it is not degenerate. The flat
spectrum does NOT collapse to hard orthogonality under Pion's *additive* update
(the update is only approximately spectrum-preserving, so the spectrum can still
evolve).

Plain matplotlib style (white bg, default palette) per reviewer feedback.
"""
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "font.size": 11,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
})

pat = re.compile(r'^(\d+) success ([\d.]+)')
CF, NUM_STEPS = 40000, 160000


def load(init):
    curves = []
    for s in (0, 1):
        vals, steps = [], []
        for line in open(f"logs/init_sweep/pion_{init}_{s}.log", encoding="utf-8", errors="ignore"):
            m = pat.match(line.strip())
            if m:
                steps.append(int(m.group(1)))
                vals.append(float(m.group(2)))
        curves.append(vals)
    L = min(len(c) for c in curves)
    return np.array(steps[:L]), np.mean([c[:L] for c in curves], axis=0)


def main():
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    for init, label, color in [
        ("xavier",     "xavier  (shaped spectrum, rank ~17.7)", "#1f77b4"),
        ("orthogonal", "orthogonal  (flat full-rank, rank 64)", "#ff7f0e"),
    ]:
        steps, m = load(init)
        ax.plot(steps, m, color=color, linewidth=2.0, label=label, zorder=3)
    for x in np.arange(CF, NUM_STEPS, CF):
        ax.axvline(x, color="0.75", linewidth=0.7, linestyle="--", alpha=0.6, zorder=0)
    ax.set_ylim(0.0, 1.03)
    ax.set_xlim(0, NUM_STEPS)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x/1e3:.0f}k"))
    ax.set_xlabel("Environment steps (dashed = task switch)")
    ax.set_ylabel("Success rate")
    ax.set_title("Pion — flat vs shaped initial spectrum: tie",
                 loc="left", fontweight="bold", pad=10)
    ax.legend(frameon=False, loc="lower right")
    ax.grid(True, color="0.9", linewidth=0.7, alpha=0.9)
    ax.set_axisbelow(True)
    fig.savefig("plots/init_comparison.png")
    plt.close(fig)
    print("wrote plots/init_comparison.png")


if __name__ == "__main__":
    main()
