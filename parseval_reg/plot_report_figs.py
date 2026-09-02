"""第六周汇报用补图：Pion/POET 初始化成功率 + 四正则化稳定秩演化。朴素风、无标题。"""
import pickle, numpy as np, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

RESULTS = 'results'
OUT = r'C:\Users\杨斯杰\Desktop\第六周'
SEEDS = range(6)

INITS = [('identity', '#2ca02c'), ('standard', '#1f77b4'), ('xavier', '#d62728'), ('orthogonal', '#ff7f0e')]
REG = [
    ('Parseval', 'parseval_{s}',        ['0.0001', '0.001', '0.01'], '#d62728'),
    ('Spectral', 'spectral_{s}',        ['0.0001', '0.001', '0.01'], '#1f77b4'),
    ('ISO',      'iso_{s}',             ['0.0001', '0.001', '0.01'], '#2ca02c'),
    ('L2-ER',    'l2er_b0.001_wd{s}',   ['0.0001', '0.001', '0.01'], '#9467bd'),
]
STR_AMOUNT = {'0.0001': 0.55, '0.001': 0.25, '0.01': 0.0}

plt.rcParams.update({'axes.labelsize': 16, 'xtick.labelsize': 13, 'ytick.labelsize': 13, 'legend.fontsize': 13})


def lighten(color, amount):
    rgb = np.array(mcolors.to_rgb(color))
    return tuple(rgb + (np.array([1.0, 1.0, 1.0]) - rgb) * amount)


def moving_avg(x, w=5):
    return x if len(x) < w else np.convolve(x, np.ones(w) / w, mode='same')


def load(template, field):
    out = []
    for seed in SEEDS:
        p = os.path.join(RESULTS, template.format(seed=seed))
        if not os.path.exists(p):
            continue
        with open(p, 'rb') as f:
            d = pickle.load(f)
        if field == 'suc':
            out.append(np.asarray(d['mean_eval_success'], dtype=float))
        elif field == 'sr':
            sr = d['actor_matrix_stable_rank']
            out.append([t[1] for t in sr])   # 隐藏层 stable rank
    return None if not out else np.mean(np.asarray(out), axis=0)


# ---- 1. Pion 四种初始化成功率曲线 ----
fig, ax = plt.subplots(figsize=(9, 5.5))
for init, color in INITS:
    suc = load(f'data_metaworld_sequence_set0_pion_minimal_rms_{{seed}}_rms_init_{init}.pkl', 'suc')
    if suc is None:
        print(f'  缺 Pion {init}'); continue
    ax.plot(np.arange(len(suc)) * 25000 / 1e6, moving_avg(suc), color=color, lw=1.8, label=init)
ax.set_xlabel('training steps (M)'); ax.set_ylabel('success rate'); ax.set_ylim(0, 1.05)
ax.legend(frameon=False); ax.grid(alpha=0.25)
fig.savefig(os.path.join(OUT, 'pion_init_success.png'), dpi=160, bbox_inches='tight'); plt.close(fig)
print('已画 pion_init_success.png')

# ---- 2. POET pure 四种初始化成功率曲线 ----
fig, ax = plt.subplots(figsize=(9, 5.5))
for init, color in INITS:
    suc = load(f'data_metaworld_sequence_set0_poet_{{seed}}_pure_init_{init}.pkl', 'suc')
    if suc is None:
        print(f'  缺 POET pure {init}'); continue
    ax.plot(np.arange(len(suc)) * 25000 / 1e6, moving_avg(suc), color=color, lw=1.8, label=init)
ax.set_xlabel('training steps (M)'); ax.set_ylabel('success rate'); ax.set_ylim(0, 1.05)
ax.legend(frameon=False); ax.grid(alpha=0.25)
fig.savefig(os.path.join(OUT, 'poet_pure_success.png'), dpi=160, bbox_inches='tight'); plt.close(fig)
print('已画 poet_pure_success.png')

# ---- 3. 四正则化稳定秩演化（每方法一张） ----
for name, suffix, strengths, color in REG:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for s in strengths:
        sr = load(f'data_metaworld_sequence_set0_base_{{seed}}_{suffix.format(s=s)}.pkl', 'sr')
        if sr is None:
            print(f'  缺 {name} {s}'); continue
        ax.plot(np.arange(len(sr)) * 25000 / 1e6, moving_avg(sr), color=lighten(color, STR_AMOUNT[s]), lw=1.6, label=s)
    ax.set_xlabel('training steps (M)'); ax.set_ylabel('stable rank')
    ax.legend(frameon=False, title=name); ax.grid(alpha=0.25)
    fig.savefig(os.path.join(OUT, f'stable_rank_{name}.png'), dpi=160, bbox_inches='tight'); plt.close(fig)
    print(f'已画 stable_rank_{name}.png')

print('全部完成')
