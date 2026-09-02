"""POET 降 lr 前后对照：lr=2e-4（重跑） vs lr=5e-4（之前 official）。

左/右各 4 条初始化曲线，横轴训练步数，纵轴成功率，6 seed 平均 + 滑动平均。
朴素风、短标题、标签不带括号。
"""
import pickle, numpy as np, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RESULTS = 'results'
OUT = r'C:\Users\杨斯杰\Desktop\第六周'
SEEDS = range(6)

INITS = [
    ('identity',   '#2ca02c'),
    ('standard',   '#1f77b4'),
    ('xavier',     '#d62728'),
    ('orthogonal', '#ff7f0e'),
]

plt.rcParams.update({
    'axes.labelsize': 15,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
})


def moving_avg(x, w=5):
    if len(x) < w:
        return x
    return np.convolve(x, np.ones(w) / w, mode='same')


def load_success(template):
    succ = []
    for seed in SEEDS:
        p = os.path.join(RESULTS, template.format(seed=seed))
        if not os.path.exists(p):
            continue
        with open(p, 'rb') as f:
            d = pickle.load(f)
        succ.append(np.asarray(d['mean_eval_success'], dtype=float))
    if not succ:
        return None
    return np.mean(np.asarray(succ), axis=0)


fig, axes = plt.subplots(1, 2, figsize=(15, 5.8), sharey=True)

panels = [
    ('lr 2e-4', 'lr2e4_init_{init}'),
    ('lr 5e-4', 'official_init_{init}'),
]

for ax, (tag, suffix) in zip(axes, panels):
    for init, color in INITS:
        suc = load_success(f'data_metaworld_sequence_set0_poet_{{seed}}_{suffix.format(init=init)}.pkl')
        if suc is None:
            print(f'  缺失: {tag} {init}')
            continue
        steps = np.arange(len(suc)) * 25000 / 1e6
        ax.plot(steps, moving_avg(suc, 5), color=color, lw=1.8, label=init)
    ax.set_title(tag, fontsize=14)
    ax.set_xlabel('training steps (M)')
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.25)

axes[0].set_ylabel('success rate')
axes[1].legend(frameon=False, loc='lower left')

fig.tight_layout()
fig.savefig(os.path.join(OUT, 'poet_lr_compare_success.png'), dpi=160, bbox_inches='tight')
plt.close(fig)
print('已画 poet_lr_compare_success.png')
