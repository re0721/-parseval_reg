#!/usr/bin/env bash
# 初始化扫描：Pion 在不同初始谱形状下的表现。
#
# 动机（老师要求"多试几种初始化，比如全 1"）：
#   保谱方法保的是"初始化时的谱"。所以初始化的谱形状直接决定命运。
#   - ones（全 1）= 秩 1 退化谱（stable rank = 1），保谱会把这个退化谱锁死，
#     网络永远只剩 1 个有效方向，学不动 —— 预期失败（负对照）。
#   - xavier = 有形状的健康谱（Marchenko–Pastur），保谱保持健康谱 —— 好（对照）。
#
# 用法：等 MetaWorld 跑完（内存空出来）再执行。跑完用
#   python analyze_probe.py logs/init_sweep
# 分析。
set -u
PY="D:/anaconda/envs/parseval/python.exe"
STEPS=160000          # 4 任务短探针（只判断"学得动 vs 学不动"，不做最终评价）
SEEDS="0 1"
mkdir -p logs/init_sweep
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

for cfg in "xavier" "ones"; do
  for i in $SEEDS; do
    echo "  -> pion init=$cfg seed_idx=$i"
    "$PY" main.py --env gridworld_ninerooms --algorithm pion \
        --repeat_idx "$i" --learning_rate 0.001 --num_steps $STEPS \
        --weight_init "$cfg" --save_suffix "init_$cfg" \
        > "logs/init_sweep/pion_${cfg}_${i}.log" 2>&1 &
  done
done

echo ""
echo "已启动 $(jobs -r | wc -l) 个进程（16 万步 / 4 任务）"
wait
echo "=== 初始化扫描跑完 ==="
