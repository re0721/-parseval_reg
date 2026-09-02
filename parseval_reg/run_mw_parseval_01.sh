#!/usr/bin/env bash
# Parseval 正则化强度 0.1 补跑（补老师加的 0.1 档）。
# 与现有 {1e-4,1e-3,1e-2} 完全同设置：--algorithm base --parseval_reg 0.1，
# orthogonal + tanh + lr=3e-4，add_diag_layer=False。
# 1000 万步 / 10 任务，6 seed。
set -u
PY="D:/anaconda/envs/parseval/python.exe"
SEEDS="0 1 2 3 4 5"
ENV="metaworld_sequence_set0"
mkdir -p logs/reg_full
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

for i in $SEEDS; do
  echo "  -> parseval 0.1 seed=$i"
  "$PY" main.py --env "$ENV" --algorithm base --repeat_idx "$i" \
      --learning_rate 0.0003 --num_steps 10000000 \
      --parseval_reg 0.1 \
      --save_suffix "parseval_0.1" \
      > "logs/reg_full/parseval_0.1_${i}.log" 2>&1 &
done

echo ""
echo "已启动 $(jobs -r | wc -l) 个进程（1 强度 × 6 seed = 6）"
wait
echo "=== Parseval 0.1 补跑完成 ==="
