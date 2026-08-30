"""第六周最终版：4 个正则化方法，谱图和成功率曲线各一张/方法。

- 谱图：只要 10M 步（训练结束），去标题，去 Pion，每方法一张（3 强度同色系深浅）
- 成功率曲线：每方法一张（3 强度同色系深浅）
"""
import pickle, numpy as np, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# 调大字体（老师反馈：坐标轴标签和图例太小）
plt.rcParams.update({
    'axes.labelsize': 16,
    'xtick.labelsize': 13,
    'ytick.labelsize': 13,
    'legend.fontsize': 13,
    'legend.title_fontsize': 14,
})

RESULTS = 'results'
OUT = r'C:\Users\杨斯杰\Desktop\第六周'
SEEDS = range(6)
END = 399  # 最后一个 eval 点 = 10M 步

METHODS = [
    ('Parseval', 'parseval_{s}',        ['0.0001', '0.001', '0.01'], '#d62728'),
    ('Spectral', 'spectral_{s}',        ['0.0001', '0.001', '0.01'], '#1f77b4'),
    ('ISO',      'iso_{s}',             ['0.0001', '0.001', '0.01'], '#2ca02c'),
    ('L2-ER',    'l2er_b0.001_wd{s}',   ['0.0001', '0.001', '0.01'], '#9467bd'),
]
STR_AMOUNT = {'0.0001': 0.55, '0.001': 0.25, '0.01': 0.0}


def lighten(color, amount):
    rgb = np.array(mcolors.to_rgb(color))
    return tuple(rgb + (np.array([1.0, 1.0, 1.0]) - rgb) * amount)


def moving_avg(x, w=5):
    if len(x) < w:
        return x
    return np.convolve(x, np.ones(w) / w, mode='same')


def load(template, field):
    out = []
    for seed in SEEDS:
        p = os.path.join(RESULTS, template.format(seed=seed))
        if not os.path.exists(p):
            continue
        with open(p, 'rb') as f:
            d = pickle.load(f)
        if field == 'sv':      # 隐藏层奇异值谱
            sv = d['actor_param_singular_values']
            out.append([np.sort(np.asarray(t[1]))[::-1] for t in sv])
        elif field == 'sr':    # 隐藏层 stable rank
            sr = d['actor_matrix_stable_rank']
            out.append([t[1] for t in sr])
        elif field == 'suc':   # 成功率
            out.append(np.asarray(d['mean_eval_success'], dtype=float))
    if not out:
        return None
    return np.mean(np.asarray(out), axis=0)


# ============ 谱图：4 张，每方法一张，10M 步，无标题，无 Pion ============
for name, suffix, strengths, color in METHODS:
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for s in strengths:
        sv = load(f'data_metaworld_sequence_set0_base_{{seed}}_{suffix.format(s=s)}.pkl', 'sv')
        if sv is None:
            print(f'  谱缺失: {name} {s}')
            continue
        ax.plot(sv[END], color=lighten(color, STR_AMOUNT[s]), lw=1.6, label=s)
    ax.set_xlabel('singular value index')
    ax.set_ylabel('singular value')
    ax.legend(frameon=False, title=name)
    ax.grid(alpha=0.25)
    fig.savefig(os.path.join(OUT, f'spectrum_{name}.png'), dpi=160, bbox_inches='tight')
    plt.close(fig)
    print(f'已画 spectrum_{name}.png')

# ============ 成功率曲线：4 张，每方法一张 ============
for name, suffix, strengths, color in METHODS:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for s in strengths:
        suc = load(f'data_metaworld_sequence_set0_base_{{seed}}_{suffix.format(s=s)}.pkl', 'suc')
        if suc is None:
            print(f'  成功率缺失: {name} {s}')
            continue
        steps = np.arange(len(suc)) * 25000 / 1e6
        ax.plot(steps, moving_avg(suc, 5), color=lighten(color, STR_AMOUNT[s]), lw=1.5, label=s)
    ax.set_xlabel('training steps (M)')
    ax.set_ylabel('success rate')
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, title=name)
    ax.grid(alpha=0.25)
    fig.savefig(os.path.join(OUT, f'success_{name}.png'), dpi=160, bbox_inches='tight')
    plt.close(fig)
    print(f'已画 success_{name}.png')

print('全部完成')
