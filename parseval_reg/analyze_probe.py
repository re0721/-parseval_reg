"""解析 lr 探针日志，报告每个配置是否崩溃 + 成功率。

两种口径都给（别混用，见 experiment-plot-log.md 的教训）：
  raw   = 所有 eval 点的原始均值
  steady = 每个任务后半段（换目标后 20k-40k 步）的均值，即"稳态"口径
崩溃判据：最后一个任务全程 success == 0（POET v3 的签名）。
"""
import re
import sys
from pathlib import Path
from collections import defaultdict

CHANGE_FREQ = 40000
LOG_DIR = Path(sys.argv[1] if len(sys.argv) > 1 else 'logs/probe')

pat = re.compile(r'^(\d+) success ([\d.]+) \+/- ([\d.]+)')


def parse(path):
    pts = []
    for line in path.read_text(errors='ignore').splitlines():
        m = pat.match(line.strip())
        if m:
            pts.append((int(m.group(1)), float(m.group(2))))
    return pts


def summarize(pts):
    if not pts:
        return None
    raw = sum(v for _, v in pts) / len(pts)
    by_task = defaultdict(list)
    for step, v in pts:
        by_task[step // CHANGE_FREQ].append((step % CHANGE_FREQ, v))
    steady_vals, task_means = [], []
    for t in sorted(by_task):
        half = [v for off, v in by_task[t] if off >= CHANGE_FREQ // 2]
        if half:
            steady_vals.extend(half)
            task_means.append(sum(half) / len(half))
    steady = sum(steady_vals) / len(steady_vals) if steady_vals else float('nan')
    # 崩溃判据只看「已跑完的」任务：刚换目标时成功率天然是 0，
    # 拿未完成的末任务判崩溃会假阳性。
    complete = [t for t in sorted(by_task)
                if max(off for off, _ in by_task[t]) >= CHANGE_FREQ - 10000]
    collapsed = bool(complete) and all(v == 0.0 for _, v in by_task[complete[-1]])
    return raw, steady, task_means, collapsed, len(pts), max(s for s, _ in pts)


groups = defaultdict(list)
for f in sorted(LOG_DIR.glob('*.log')):
    if f.name.startswith('_'):
        continue
    cfg = f.stem.rsplit('_', 1)[0]     # 去掉 seed 索引
    groups[cfg].append(f)

print(f"{'配置':<26} {'seed':>4} {'步数':>8} {'raw':>7} {'steady':>7}  {'各任务稳态':<34} 状态")
print("-" * 108)
for cfg in sorted(groups):
    for f in sorted(groups[cfg]):
        r = summarize(parse(f))
        if r is None:
            print(f"{cfg:<26} {f.stem[-1]:>4}  (无数据)")
            continue
        raw, steady, tms, collapsed, n, last = r
        tstr = ' '.join(f'{m:.2f}' for m in tms[:8])
        status = '崩溃' if collapsed else 'ok'
        print(f"{cfg:<26} {f.stem[-1]:>4} {last:>8} {raw:>7.3f} {steady:>7.3f}  {tstr:<34} {status}")
    print()
