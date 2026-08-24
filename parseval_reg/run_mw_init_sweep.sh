#!/usr/bin/env bash
# MetaWorld 缩短版（500 万步 / 5 任务）初始谱消融：orthogonal vs xavier（Pion）。
# 把 gridworld 短探针的发现（orthogonal ≈ xavier，不退化）在论文主环境坐实。
#
# env = metaworld_sequence_set0（metaworld_env.py:103 是 RPO10_SEQ[env_set_id-1]，
#       所以 set0 实际取第 20 条序列 —— 沿用旧实验以便对照）
# lr = 0.002（MetaWorld 的 pion lr，见 run_mw10.sh）
# 必须限制 BLAS 线程（6 进程并行，OMP/MKL=1）
set -u
PY="D:/anaconda/envs/parseval/python.exe"
SEEDS="0 1 2 3 4 5"
ENV="metaworld_sequence_set0"
mkdir -p logs/mw_init_sweep
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

launch () {  # $1=weight_init
  for i in $SEEDS; do
    echo "  -> pion init=$1 seed_idx=$i"
    "$PY" main.py --env "$ENV" --algorithm pion \
        --repeat_idx "$i" --learning_rate 0.002 --num_steps 5000000 \
        --weight_init "$1" --save_suffix "init_$1" \
        > "logs/mw_init_sweep/pion_${1}_${i}.log" 2>&1 &
  done
}

launch orthogonal
launch xavier

echo ""
echo "已启动 $(jobs -r | wc -l) 个进程（500 万步 / 5 任务，预计 2-3 小时）"
wait
echo "=== MetaWorld 初始谱消融（orthogonal vs xavier）跑完 ==="
