"""Parseval λ=0.01 的 6 个 seed 各自成功率曲线（不平均，逐 seed 画）。"""
import pickle, numpy as np, os, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RESULTS = 'results'
OUT = r'C:\Users\杨斯杰\Desktop\第七周'
SEEDS = range(6)
STR = sys.argv[1] if len(sys.argv) > 1 else '0.01'

plt.rcParams.update({
    'axes.labelsize': 15,
    'xtick.labelsize': 13,
    'ytick.labelsize': 13,
    'legend.fontsize': 12,
    'axes.unicode_minus': False,
})

COLORS = plt.cm.tab10.colors  # 6 个 seed 用不同色


def moving_avg(x, w=5):
    return x if len(x) < w else np.convolve(x, np.ones(w) / w, mode='same')


def boundary_excluded(arr):
    mask = np.ones(len(arr), dtype=bool)
    mask[39::40] = False
    return arr[mask]


fig, ax = plt.subplots(figsize=(7, 4.5))
for i, seed in enumerate(SEEDS):
    p = os.path.join(RESULTS, f'data_metaworld_sequence_set0_base_{seed}_parseval_diag_{STR}.pkl')
    if not os.path.exists(p):
        print(f'  缺失 seed {seed}')
        continue
    with open(p, 'rb') as f:
        d = pickle.load(f)
    suc = np.asarray(d['mean_eval_success'], dtype=float)
    steps = np.arange(len(suc)) * 25000 / 1e6
    ax.plot(steps, moving_avg(suc, 5), color=COLORS[i], lw=1.5, label=f'seed {seed}')

ax.set_xlabel('training steps (M)')
ax.set_ylabel('success rate')
ax.set_ylim(0, 1.05)
ax.grid(alpha=0.25)
ax.legend(frameon=False, loc='upper left', ncol=2)
fig.savefig(os.path.join(OUT, f'success_Parseval_{STR}_seeds.png'), dpi=160, bbox_inches='tight')
plt.close(fig)
print(f'已画 success_Parseval_{STR}_seeds.png')

# 打印每个 seed 的成功率
print(f'\n=== Parseval λ={STR} 各 seed 成功率 ===')
for seed in SEEDS:
    p = os.path.join(RESULTS, f'data_metaworld_sequence_set0_base_{seed}_parseval_diag_{STR}.pkl')
    with open(p, 'rb') as f:
        d = pickle.load(f)
    suc = np.asarray(d['mean_eval_success'], dtype=float)
    raw = float(np.mean(suc))
    be = float(np.mean(boundary_excluded(suc)))
    print(f'  seed {seed}: raw={raw:.3f}  剔除边界={be:.3f}')
