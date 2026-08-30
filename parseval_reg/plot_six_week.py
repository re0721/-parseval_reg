"""第六周：5 算法（Parseval/Spectral/ISO/L2-ER/Pion）的谱 + 有效秩，6 张图。

- 谱 = 隐藏层(层1)权重矩阵的奇异值谱（降序）
- 有效秩 = 隐藏层 stable rank
- 3 个时间点（开始/中间/快结束）× 2 指标 = 6 张图
- 同一方法同一色系，不同正则化强度不同深浅
"""
import pickle, numpy as np, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

RESULTS = 'results'
OUT = r'C:\Users\杨斯杰\Desktop\第六周'
SEEDS = range(6)

# 方法: (名称, 强度列表, 主色)。文件模板 base_{seed}_{name}_{s}.pkl
METHODS = [
    ('Parseval', 'parseval_{s}',        ['0.0001', '0.001', '0.01'], '#d62728'),
    ('Spectral', 'spectral_{s}',        ['0.0001', '0.001', '0.01'], '#1f77b4'),
    ('ISO',      'iso_{s}',             ['0.0001', '0.001', '0.01'], '#2ca02c'),
    ('L2-ER',    'l2er_b0.001_wd{s}',   ['0.0001', '0.001', '0.01'], '#9467bd'),
]
PION_FILE = 'data_metaworld_sequence_set0_pion_minimal_rms_{seed}_rms_init_orthogonal.pkl'
PION_COLOR = '#ff7f0e'


def lighten(color, amount):
    """amount 0=原色(深), 1=白色(浅)"""
    rgb = np.array(mcolors.to_rgb(color))
    return tuple(rgb + (np.array([1.0, 1.0, 1.0]) - rgb) * amount)


def load_layer1(template):
    """template 含 {seed}。返回 (奇异值谱[time,64], stable_rank[time])，6 seed 平均。"""
    svs, srs = [], []
    for seed in SEEDS:
        p = os.path.join(RESULTS, template.format(seed=seed))
        if not os.path.exists(p):
            continue
        with open(p, 'rb') as f:
            d = pickle.load(f)
        sv = d['actor_param_singular_values']   # [time][layer][singvals]
        sr = d['actor_matrix_stable_rank']      # [time][layer]
        svs.append([np.sort(np.asarray(t[1]))[::-1] for t in sv])  # 层1 奇异值降序
        srs.append([t[1] for t in sr])                              # 层1 stable rank
    if not svs:
        return None, None
    return np.mean(np.asarray(svs), axis=0), np.mean(np.asarray(srs), axis=0)


# 收集所有曲线（顺序：4 正则化 × 3 强度，再 Pion）
curves = []
for name, suffix, strengths, color in METHODS:
    for s in strengths:
        sv, sr = load_layer1(f'data_metaworld_sequence_set0_base_{{seed}}_{suffix.format(s=s)}.pkl')
        if sv is None:
            print(f'  缺失: {name} {s}')
            continue
        curves.append(dict(name=name, strength=s, color=color, sv=sv, sr=sr))

sv, sr = load_layer1(PION_FILE)
if sv is None:
    print('  Pion 缺失')
else:
    curves.append(dict(name='Pion', strength=None, color=PION_COLOR, sv=sv, sr=sr))

TIMES = [('start', 0), ('middle', 200), ('end', 399)]
TIME_TITLE = {'start': '0 steps', 'middle': '5M steps', 'end': '10M steps'}
STR_AMOUNT = {'0.0001': 0.55, '0.001': 0.25, '0.01': 0.0}  # 弱=浅, 强=深

os.makedirs(OUT, exist_ok=True)

# ============ 谱图（3 张） ============
for key, ti in TIMES:
    fig, ax = plt.subplots(figsize=(7, 5))
    for c in curves:
        if c['strength'] is None:
            col, lw, label = c['color'], 2.2, c['name']
        else:
            col = lighten(c['color'], STR_AMOUNT[c['strength']])
            lw, label = 1.6, f"{c['name']} {c['strength']}"
        ax.plot(c['sv'][ti], color=col, lw=lw, label=label)
    ax.set_xlabel('singular value index')
    ax.set_ylabel('singular value')
    ax.set_title(TIME_TITLE[key])
    ax.legend(fontsize=7, ncol=2, frameon=False)
    ax.grid(alpha=0.25)
    fig.savefig(os.path.join(OUT, f'spectrum_{key}.png'), dpi=160, bbox_inches='tight')
    plt.close(fig)
    print(f'已画 spectrum_{key}.png')

# ============ 有效秩图（3 张） ============
for key, ti in TIMES:
    fig, ax = plt.subplots(figsize=(9, 4.5))
    vals = [c['sr'][ti] for c in curves]
    cols = [(lighten(c['color'], STR_AMOUNT[c['strength']]) if c['strength'] else c['color']) for c in curves]
    labels = [f"{c['name']}\n{c['strength'] if c['strength'] else ''}" for c in curves]
    ax.bar(range(len(vals)), vals, color=cols, edgecolor='white', linewidth=0.4)
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels(labels, fontsize=6.5)
    ax.set_ylabel('stable rank (hidden layer)')
    ax.set_title(TIME_TITLE[key])
    ax.grid(alpha=0.25, axis='y')
    fig.savefig(os.path.join(OUT, f'stable_rank_{key}.png'), dpi=160, bbox_inches='tight')
    plt.close(fig)
    print(f'已画 stable_rank_{key}.png')

print('全部完成')
