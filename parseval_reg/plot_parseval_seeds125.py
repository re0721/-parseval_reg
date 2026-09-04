"""Parseval 用指定 seed 平均后，四挡正则强度画在一张图（统一配色）。用法: python plot_parseval_seeds125.py 0,1,2"""
import pickle, numpy as np, os, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RESULTS = 'results'
OUT = r'C:\Users\杨斯杰\Desktop\第七周'
SEEDS = [int(x) for x in (sys.argv[1].split(',') if len(sys.argv) > 1 else '1,2,5')]
SEEDTAG = ''.join(map(str, SEEDS))

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


def boundary_excluded(arr):
    mask = np.ones(len(arr), dtype=bool)
    mask[39::40] = False
    return arr[mask]


def load_mean(strength):
    succ = []
    for seed in SEEDS:
        p = os.path.join(RESULTS, f'data_metaworld_sequence_set0_base_{seed}_parseval_diag_{strength}.pkl')
        with open(p, 'rb') as f:
            d = pickle.load(f)
        succ.append(np.asarray(d['mean_eval_success'], dtype=float))
    return np.mean(np.asarray(succ), axis=0)


fig, ax = plt.subplots(figsize=(7, 4.5))
for label, color in STRENGTHS:
    mean = load_mean(label)
    steps = np.arange(len(mean)) * 25000 / 1e6
    ax.plot(steps, moving_avg(mean, 5), color=color, lw=1.8, label=label)

ax.set_xlabel('training steps (M)')
ax.set_ylabel('success rate')
ax.set_ylim(0, 1.05)
ax.grid(alpha=0.25)
ax.legend(frameon=False, title='Parseval', loc='upper left')
fig.savefig(os.path.join(OUT, f'success_Parseval_seeds{SEEDTAG}.png'), dpi=160, bbox_inches='tight')
plt.close(fig)
print(f'已画 success_Parseval_seeds{SEEDTAG}.png')

print(f'\n=== Parseval seed {"/".join(map(str, SEEDS))} 各强度成功率 ===')
for label, _ in STRENGTHS:
    mean = load_mean(label)
    raw = float(np.mean(mean))
    be = float(np.mean(boundary_excluded(mean)))
    print(f'  {label}: raw={raw:.3f}  剔除边界={be:.3f}')
