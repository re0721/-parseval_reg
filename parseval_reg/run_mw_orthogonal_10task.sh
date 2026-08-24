#!/usr/bin/env bash
# MetaWorld 完整 10 任务：orthogonal 初始化（Pion）。
# 对齐已有的 xavier 10 任务结果（Pion 0.841，logs/mw10/pion_*.log），
# 只跑 orthogonal，不重跑 xavier（10 任务口径下 xavier 已有权威数据）。
#
# env = metaworld_sequence_set0（metaworld_env.py:103 是 RPO10_SEQ[env_set_id-1]，
#       所以 set0 实际取第 20 条序列 —— 沿用旧实验以便对照）
# lr = 0.002（MetaWorld 的 pion lr，见 run_mw10.sh）
set -u
PY="D:/anaconda/envs/parseval/python.exe"
SEEDS="0 1 2 3 4 5"
ENV="metaworld_sequence_set0"
mkdir -p logs/mw_orthogonal_10task
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

for i in $SEEDS; do
  echo "  -> pion init=orthogonal seed_idx=$i"
  "$PY" main.py --env "$ENV" --algorithm pion \
      --repeat_idx "$i" --learning_rate 0.002 --num_steps 10000000 \
      --weight_init orthogonal --save_suffix "init_orthogonal_10task" \
      > "logs/mw_orthogonal_10task/pion_orthogonal_${i}.log" 2>&1 &
done

echo ""
echo "已启动 $(jobs -r | wc -l) 个进程（1000 万步 / 10 任务，预计 6-10 小时）"
wait
echo "=== MetaWorld 10 任务 orthogonal（Pion）跑完 ==="
