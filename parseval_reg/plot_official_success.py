"""official 版 POET（带 SPO）四种初始化的成功率曲线。

横轴 = 训练步数（M），纵轴 = 成功率，401 点滑动平均，6 seed 平均。
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
    'axes.labelsize': 16,
    'xtick.labelsize': 13,
    'ytick.labelsize': 13,
    'legend.fontsize': 13,
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


fig, ax = plt.subplots(figsize=(9, 5.5))
for init, color in INITS:
    suc = load_success(f'data_metaworld_sequence_set0_poet_{{seed}}_official_init_{init}.pkl')
    if suc is None:
        print(f'  缺失: {init}')
        continue
    steps = np.arange(len(suc)) * 25000 / 1e6
    ax.plot(steps, moving_avg(suc, 5), color=color, lw=1.8, label=init)

ax.set_xlabel('training steps (M)')
ax.set_ylabel('success rate')
ax.set_ylim(0, 1.05)
ax.legend(frameon=False)
ax.grid(alpha=0.25)
fig.savefig(os.path.join(OUT, 'official_success_curve.png'), dpi=160, bbox_inches='tight')
plt.close(fig)
print('已画 official_success_curve.png')
