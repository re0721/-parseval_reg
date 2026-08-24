"""Gridworld: original 40k/task vs doubled 80k/task success rate.

Plain research style (default matplotlib). Data = real logs; last (boundary)
eval point of each task is dropped.
"""
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PAT = re.compile(r'^(\d+) success ([\d.]+)')
METHODS = ['base', 'parseval', 'pion', 'poet_exact']
LABELS = ['base', 'Parseval', 'Pion', 'POET-exact']

# original 40k/task runs (800k steps, change_freq=40000)
ORIG_PATHS = {
    'base':       'logs/ref/gw_base_{s}.log',
    'parseval':   'logs/ref/gw_parseval_{s}.log',
    'pion':       'logs/pion_full/pion_{s}.log',
    'poet_exact': 'logs/full/poet_exact_lr5e4_{s}.log',
}
# doubled 80k/task runs (1.6M steps, change_freq=80000)
LONG_PATH = 'logs/gw_long/{m}_{s}.log'


def per_seed_mean(path_template, cf, num_steps):
    means = []
    for s in range(6):
        vals = []
        for line in open(path_template.format(s=s), encoding='utf-8', errors='ignore'):
            m = PAT.match(line.strip())
            if m and int(m.group(1)) < num_steps:
                st, v = int(m.group(1)), float(m.group(2))
                if st % cf != cf - 1:
                    vals.append(v)
        means.append(np.mean(vals))
    return np.array(means)  # (6 seeds,)


orig = {m: per_seed_mean(ORIG_PATHS[m], 40000, 800000) for m in METHODS}
long = {m: per_seed_mean(LONG_PATH.format(m=m, s='{s}'), 80000, 1600000) for m in METHODS}

fig, ax = plt.subplots(figsize=(7.5, 4.5))
x = np.arange(len(METHODS))
w = 0.38
for offset, d, color, label in [(-w / 2, orig, '#9e9e9e', '40k steps / task'),
                                 (w / 2, long, '#1f77b4', '80k steps / task (2x)')]:
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
fig.savefig('plots/gw_40k_vs_80k.png', dpi=200)
print('wrote plots/gw_40k_vs_80k.png')

print('\nmethod       40k      80k      delta')
for m, lab in zip(METHODS, LABELS):
    a, b = orig[m].mean(), long[m].mean()
    print(f'{lab:<12} {a:.3f}    {b:.3f}    {b-a:+.3f}')
