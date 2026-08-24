#!/usr/bin/env bash
# Pion / Pion-mult 全量重跑到 80 万步（20 任务）、6 seed。
#
# 为什么是重跑而不是"续跑"：save_model_freq 默认 -1，之前没存过任何 model
# checkpoint，也没有 --resume 入口，所以中断的 run 无法接续。
#
# 关键：**必须限制 BLAS 线程**。之前 v3 那批不限线程跑 12 进程 → 12×32=384 线程
# 抢 32 核 → SPS 掉到 16（跑满要 14 小时，结果被杀在 13 任务）。
# 实测单进程 SPS≈490。这里给每进程 2 线程，12 进程共 24 线程 < 32 核。
# 配置与 v3 完全一致（lr=1e-3、xavier 初始化、β2 由代码按变体设定：
# 官方加法版 0.999 / 乘法版 0.95），只是这次跑到头。

set -u
PY="D:/anaconda/envs/parseval/python.exe"
SEEDS="0 1 2 3 4 5"
mkdir -p logs/pion_full
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2

for algo in pion pion_mult; do
  for i in $SEEDS; do
    echo "  -> $algo seed_idx=$i"
    "$PY" main.py --env gridworld_ninerooms --algorithm "$algo" \
        --repeat_idx "$i" --learning_rate 0.001 \
        --save_suffix full \
        > "logs/pion_full/${algo}_${i}.log" 2>&1 &
  done
done

echo ""
echo "已启动 $(jobs -r | wc -l) 个进程"
wait
echo "=== Pion 全量跑完 ==="
