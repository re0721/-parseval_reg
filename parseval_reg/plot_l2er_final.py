"""L2-ER 解耦版（lecun+ReLU+lr=3e-4+beta=1e-2）四挡 weight_decay 成功率曲线。朴素风。"""
import pickle, numpy as np, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RESULTS = 'results'
OUT = r'C:\Users\杨斯杰\Desktop\第六周'
SEEDS = range(6)

WDS = [
    ('1e-4', '0.0001', '#1f77b4'),
    ('1e-3', '0.001',  '#d62728'),
    ('1e-2', '0.01',   '#2ca02c'),
    ('0.1',  '0.1',    '#ff7f0e'),
]

plt.rcParams.update({'axes.labelsize': 16, 'xtick.labelsize': 13, 'ytick.labelsize': 13, 'legend.fontsize': 13})


def moving_avg(x, w=5):
    return x if len(x) < w else np.convolve(x, np.ones(w) / w, mode='same')


def load_success(template):
    succ = []
    for seed in SEEDS:
        p = os.path.join(RESULTS, template.format(seed=seed))
        if not os.path.exists(p):
            continue
        with open(p, 'rb') as f:
            d = pickle.load(f)
        succ.append(np.asarray(d['mean_eval_success'], dtype=float))
    return None if not succ else np.mean(np.asarray(succ), axis=0)


fig, ax = plt.subplots(figsize=(9, 5.5))
for label, wd, color in WDS:
    suc = load_success(f'data_metaworld_sequence_set0_base_{{seed}}_l2er_final_wd{wd}.pkl')
    if suc is None:
        print(f'  缺失: {label}')
        continue
    steps = np.arange(len(suc)) * 25000 / 1e6
    ax.plot(steps, moving_avg(suc, 5), color=color, lw=1.8, label=f'weight decay {label}')

ax.set_xlabel('training steps (M)')
ax.set_ylabel('success rate')
ax.set_ylim(0, 1.05)
ax.legend(frameon=False)
ax.grid(alpha=0.25)
fig.savefig(os.path.join(OUT, 'l2er_final_success.png'), dpi=160, bbox_inches='tight')
plt.close(fig)
print('已画 l2er_final_success.png')
