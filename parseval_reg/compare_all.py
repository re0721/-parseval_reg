"""跨方法汇总对比：把所有 run 截断到同一步数上限，按配置聚合 seed。

用法：python compare_all.py [max_steps]
不给 max_steps 就用各 run 的全部数据（此时被杀的 run 会因为少跑后段而占便宜）。

口径：raw = 所有 eval 点原始均值（与 experiment-plot-log.md 一致，
可复现 Parseval 0.617 / Pion 0.726）。
"""
import re
import sys
from pathlib import Path
from collections import defaultdict

CHANGE_FREQ = 40000
MAX_STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else None

# 每个配置：显示名 -> (日志目录, 文件名前缀)
CONFIGS = [
    ("base",                      'logs/ref',     'gw_base'),
    ("Parseval",                  'logs/ref',     'gw_parseval'),
    ("Pion (官方加法版)",          'logs/pion_full', 'pion'),
    ("Pion-mult (乘法版)",         'logs/pion_full', 'pion_mult'),
    ("Pion (c 版, 旧)",            'logs/ref',     'gw_pion_c'),
    ("OFT  lr3e-3",               'logs/full',    'oft_lr3e3'),
    ("POET Neumann lr5e-4",       'logs/full',    'poet_lr5e4'),
    ("POET-exact  lr5e-4",        'logs/full',    'poet_exact_lr5e4'),
    ("POET-exact  lr1e-2",        'logs/full',    'poet_exact_lr1e2'),
    ("POET Neumann lr1e-2 (旧)",  'logs/oldv3',   'gw_poet_v3'),
]

pat = re.compile(r'^(\d+) success ([\d.]+)')


def run_stats(path, cap):
    pts = []
    for line in path.read_text(errors='ignore').splitlines():
        m = pat.match(line.strip())
        if m:
            step = int(m.group(1))
            if cap is None or step < cap:
                pts.append((step, float(m.group(2))))
    if not pts:
        return None
    raw = sum(v for _, v in pts) / len(pts)
    return raw, max(s for s, _ in pts)


rows = []
for name, d, prefix in CONFIGS:
    files = sorted(Path(d).glob(f'{prefix}_[0-5].log'))
    stats = [run_stats(f, MAX_STEPS) for f in files]
    stats = [s for s in stats if s]
    if not stats:
        continue
    raws = [s[0] for s in stats]
    reached = min(s[1] for s in stats)
    mean = sum(raws) / len(raws)
    spread = (max(raws) - min(raws)) / 2
    rows.append((mean, name, spread, len(raws), reached))

cap_txt = f"截断到 {MAX_STEPS} 步（{MAX_STEPS // CHANGE_FREQ} 个任务）" if MAX_STEPS else "全部数据（步数不齐，慎用）"
print(f"\n=== Gridworld 方法对比 · {cap_txt} ===\n")
print(f"{'方法':<26} {'raw 均值':>9} {'±半幅':>7} {'seeds':>6} {'最短 run 到达':>12}")
print("-" * 68)
for mean, name, spread, n, reached in sorted(rows, reverse=True):
    print(f"{name:<26} {mean:>9.3f} {spread:>7.3f} {n:>6} {reached:>12}")
print()
