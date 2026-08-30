"""第二个任务：4 个正则化方法 × 3 强度的可塑性曲线（成功率曲线），一张图。

横轴 = 任务 index（1-10），纵轴 = 每任务成功率（剔除每任务最后一个 off-by-one eval 点）。
同一方法同一色系，不同强度不同深浅。
"""
import pickle, numpy as np, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

RESULTS = 'results'
OUT = r'C:\Users\杨斯杰\Desktop\第六周'
SEEDS = range(6)

METHODS = [
    ('Parseval', 'parseval_{s}',        ['0.0001', '0.001', '0.01'], '#d62728'),
    ('Spectral', 'spectral_{s}',        ['0.0001', '0.001', '0.01'], '#1f77b4'),
    ('ISO',      'iso_{s}',             ['0.0001', '0.001', '0.01'], '#2ca02c'),
    ('L2-ER',    'l2er_b0.001_wd{s}',   ['0.0001', '0.001', '0.01'], '#9467bd'),
]

STR_AMOUNT = {'0.0001': 0.55, '0.001': 0.25, '0.01': 0.0}  # 弱=浅, 强=深


def lighten(color, amount):
    rgb = np.array(mcolors.to_rgb(color))
    return tuple(rgb + (np.array([1.0, 1.0, 1.0]) - rgb) * amount)


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
    return np.mean(np.asarray(succ), axis=0)   # [400]


fig, ax = plt.subplots(figsize=(8, 5))
for name, suffix, strengths, color in METHODS:
    for s in strengths:
        suc = load_success(f'data_metaworld_sequence_set0_base_{{seed}}_{suffix.format(s=s)}.pkl')
        if suc is None:
            print(f'  缺失: {name} {s}')
            continue
        # 10 任务 × 每任务 40 个 eval 点；剔除每任务最后 1 点(off-by-one)后取均值
        per_task = [suc[t * 40:(t + 1) * 40][:-1].mean() for t in range(10)]
        col = lighten(color, STR_AMOUNT[s])
        ax.plot(range(1, 11), per_task, color=col, lw=1.8, marker='o', ms=3,
                label=f'{name} {s}')

ax.set_xlabel('task')
ax.set_ylabel('success rate')
ax.set_xticks(range(1, 11))
ax.set_ylim(0, 1.05)
ax.legend(fontsize=8, ncol=2, frameon=False)
ax.grid(alpha=0.25)
fig.savefig(os.path.join(OUT, 'plasticity_curve.png'), dpi=160, bbox_inches='tight')
plt.close(fig)
print('已画 plasticity_curve.png')
