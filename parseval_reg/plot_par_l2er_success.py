"""Parseval 与 L2-ER 的成功率曲线（各 4 档正则强度，统一配色，拆成两张独立图）。

- 配色方案（四个算法 Parseval/Spectral/ISO/L2-ER 共用）：
    最强 0.1    -> 红
    0.01        -> 绿
    0.001       -> 蓝
    最弱 0.0001  -> 灰
- 两张图 figsize、字体、图例位置完全一致，叠放在一起时上下对齐。
"""
import pickle, numpy as np, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RESULTS = 'results'
OUT = r'C:\Users\杨斯杰\Desktop\第七周'
SEEDS = range(6)

# 统一字体（两张图一致；xtick == ytick）
plt.rcParams.update({
    'axes.labelsize': 15,
    'xtick.labelsize': 13,
    'ytick.labelsize': 13,
    'legend.fontsize': 12,
    'legend.title_fontsize': 12,
    'axes.unicode_minus': False,
})

# 统一配色（值 -> 颜色），从最弱到最强，最强者为红
STRENGTHS = [
    ('0.0001', '#7f7f7f'),
    ('0.001',  '#1f77b4'),
    ('0.01',   '#2ca02c'),
    ('0.1',    '#d62728'),
]

METHODS = [
    ('Parseval', 'data_metaworld_sequence_set0_base_{seed}_parseval_diag_{s}.pkl'),
    ('L2-ER',    'data_metaworld_sequence_set0_base_{seed}_l2er_final_wd{s}.pkl'),
]


def moving_avg(x, w=5):
    return x if len(x) < w else np.convolve(x, np.ones(w) / w, mode='same')


def load_success(base_tmpl, s_label):
    """base_tmpl 含 {seed} 与 {s}，返回 6 seed 平均的 success 数组（缺失 seed 跳过）。"""
    succ = []
    for seed in SEEDS:
        p = os.path.join(RESULTS, base_tmpl.format(seed=seed, s=s_label))
        if not os.path.exists(p):
            continue
        with open(p, 'rb') as f:
            d = pickle.load(f)
        succ.append(np.asarray(d['mean_eval_success'], dtype=float))
    return None if not succ else np.mean(np.asarray(succ), axis=0)


def boundary_excluded(arr):
    """剔除每任务最后一个 eval 点（每 40 个点废 1 个：index 39,79,...）。"""
    mask = np.ones(len(arr), dtype=bool)
    mask[39::40] = False
    return arr[mask]


for name, tmpl in METHODS:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for label, color in STRENGTHS:
        suc = load_success(tmpl, label)
        if suc is None:
            print(f'  缺失: {name} {label}')
            continue
        steps = np.arange(len(suc)) * 25000 / 1e6
        ax.plot(steps, moving_avg(suc, 5), color=color, lw=1.8, label=label)

    ax.set_xlabel('training steps (M)')
    ax.set_ylabel('success rate')
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, title=name, loc='upper left')

    fig.savefig(os.path.join(OUT, f'success_{name}.png'), dpi=160, bbox_inches='tight')
    plt.close(fig)
    print(f'已画 success_{name}.png')

# 打印各强度的成功率（raw 与剔除边界），便于记录
print('\n=== 成功率汇总（6 seed 均值） ===')
for name, tmpl in METHODS:
    print(f'[{name}]')
    for label, _ in STRENGTHS:
        suc = load_success(tmpl, label)
        if suc is None:
            print(f'  {label}: 缺失')
            continue
        raw = float(np.mean(suc))
        be = float(np.mean(boundary_excluded(suc)))
        print(f'  {label}: raw={raw:.3f}  剔除边界={be:.3f}')
