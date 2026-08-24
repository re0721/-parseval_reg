"""Compare MetaWorld success: original order (set0) vs shuffled order.

Plain research style (default matplotlib, not a "design system") so the figure
reads as field-standard. Data = real logs; last (boundary) eval point of each
task is dropped (it measures the brand-new task, ~always 0).
"""
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PAT = re.compile(r'^(\d+) success ([\d.]+)')
CF, NUM_STEPS = 1_000_000, 10_000_000
METHODS = ['base', 'parseval', 'pion', 'poet_exact']
LABELS = ['base', 'Parseval', 'Pion', 'POET-exact']


def per_seed_mean(method, d):
    """Mean success per seed, dropping the last (boundary) eval of each task."""
    means = []
    for s in range(6):
        vals = []
        for line in open(f'logs/{d}/{method}_{s}.log', encoding='utf-8', errors='ignore'):
            m = PAT.match(line.strip())
            if m and int(m.group(1)) < NUM_STEPS:
                st, v = int(m.group(1)), float(m.group(2))
                if st % CF != CF - 1:
                    vals.append(v)
        means.append(np.mean(vals))
    return np.array(means)  # (6 seeds,)


orig = {m: per_seed_mean(m, 'mw10') for m in METHODS}
shuf = {m: per_seed_mean(m, 'mw10_shuffle') for m in METHODS}

fig, ax = plt.subplots(figsize=(7.5, 4.5))
x = np.arange(len(METHODS))
w = 0.38
for offset, d, color, label in [(-w / 2, orig, '#9e9e9e', 'original order'),
                                 (w / 2, shuf, '#1f77b4', 'shuffled order')]:
    means = [d[m].mean() for m in METHODS]
    stds = [d[m].std() for m in METHODS]
    ax.bar(x + offset, means, w, yerr=stds, capsize=3, color=color, label=label)

ax.set_xticks(x)
ax.set_xticklabels(LABELS)
ax.set_ylabel('Success rate (boundary-removed)')
ax.set_ylim(0, 1)
ax.legend()
ax.grid(axis='y', alpha=0.3)
fig.tight_layout()
fig.savefig('plots/metaworld_order_comparison.png', dpi=200)
print('wrote plots/metaworld_order_comparison.png')

print('\nmethod        original   shuffled   delta')
for m, lab in zip(METHODS, LABELS):
    a, b = orig[m].mean(), shuf[m].mean()
    print(f'{lab:<12} {a:.3f}      {b:.3f}      {b-a:+.3f}')
