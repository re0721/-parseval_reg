"""Parseval seed=1（真正的 seed 1）四挡正则强度成功率曲线，统一配色。"""
import pickle, numpy as np, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RESULTS = 'results'
OUT = r'C:\Users\杨斯杰\Desktop\第七周'

plt.rcParams.update({
    'axes.labelsize': 15,
    'xtick.labelsize': 13,
    'ytick.labelsize': 13,
    'legend.fontsize': 12,
    'legend.title_fontsize': 12,
    'axes.unicode_minus': False,
})

# 统一配色（最弱到最强，最强者红）
STRENGTHS = [
    ('0.0001', '#7f7f7f'),
    ('0.001',  '#1f77b4'),
    ('0.01',   '#2ca02c'),
    ('0.1',    '#d62728'),
]


def moving_avg(x, w=5):
    return x if len(x) < w else np.convolve(x, np.ones(w) / w, mode='same')


fig, ax = plt.subplots(figsize=(7, 4.5))
for label, color in STRENGTHS:
    p = os.path.join(RESULTS, f'data_metaworld_sequence_set0_base_0_parseval_diag_{label}_seed1.pkl')
    with open(p, 'rb') as f:
        d = pickle.load(f)
    suc = np.asarray(d['mean_eval_success'], dtype=float)
    steps = np.arange(len(suc)) * 25000 / 1e6
    ax.plot(steps, moving_avg(suc, 5), color=color, lw=1.8, label=label)

ax.set_xlabel('training steps (M)')
ax.set_ylabel('success rate')
ax.set_ylim(0, 1.05)
ax.grid(alpha=0.25)
ax.legend(frameon=False, title='Parseval seed 1', loc='upper left')
fig.savefig(os.path.join(OUT, 'success_Parseval_seed1.png'), dpi=160, bbox_inches='tight')
plt.close(fig)
print('已画 success_Parseval_seed1.png')
