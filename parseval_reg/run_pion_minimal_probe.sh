#!/usr/bin/env bash
# 最简 Pion（无动量/RMS）的 lr 扫描。Gridworld 16 万步 / 4 任务 / 2 seed。
# 没 RMS 缩放后 lr 尺度会大变（Gin 范数小，lr 可能要很大），先扫一个宽范围。
set -u
PY="D:/anaconda/envs/parseval/python.exe"
SEEDS="0 1"
STEPS=160000
mkdir -p logs/pion_minimal_probe
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

for lr in 0.01 0.1 1.0 10.0; do
  for i in $SEEDS; do
    echo "  -> pion_minimal lr=$lr seed_idx=$i"
    "$PY" main.py --env gridworld_ninerooms --algorithm pion_minimal \
        --repeat_idx "$i" --learning_rate "$lr" --num_steps $STEPS \
        --weight_init xavier --save_suffix "minimal_lr${lr}" \
        > "logs/pion_minimal_probe/pion_minimal_lr${lr}_${i}.log" 2>&1 &
  done
done

echo ""
echo "已启动 $(jobs -r | wc -l) 个进程（16 万步 / 4 任务，4 lr × 2 seed）"
wait
echo "=== 最简 Pion lr 扫描跑完 ==="
