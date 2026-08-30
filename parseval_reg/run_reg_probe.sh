#!/usr/bin/env bash
# 正则化方法短探针：Parseval / Spectral / ISO / L2-ER × 3 强度 × 1 seed。
# Gridworld 16 万步 / 4 任务，统一正交初始化。lr 用 gridworld 默认 0.00025。
set -u
PY="D:/anaconda/envs/parseval/python.exe"
ENV="gridworld_ninerooms"
STEPS=160000
mkdir -p logs/reg_probe
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

for strength in 0.0001 0.001 0.01; do
  "$PY" main.py --env "$ENV" --algorithm base --repeat_idx 0 --num_steps "$STEPS" \
      --weight_init orthogonal --parseval_reg "$strength" --save_suffix "parseval_${strength}" \
      > "logs/reg_probe/parseval_${strength}.log" 2>&1 &

  "$PY" main.py --env "$ENV" --algorithm base --repeat_idx 0 --num_steps "$STEPS" \
      --weight_init orthogonal --spectral_reg "$strength" --save_suffix "spectral_${strength}" \
      > "logs/reg_probe/spectral_${strength}.log" 2>&1 &

  "$PY" main.py --env "$ENV" --algorithm base --repeat_idx 0 --num_steps "$STEPS" \
      --weight_init orthogonal --iso_reg "$strength" --save_suffix "iso_${strength}" \
      > "logs/reg_probe/iso_${strength}.log" 2>&1 &

  "$PY" main.py --env "$ENV" --algorithm base --repeat_idx 0 --num_steps "$STEPS" \
      --weight_init orthogonal --l2_er_weight_decay "$strength" --l2_er_beta 0.01 --save_suffix "l2er_${strength}" \
      > "logs/reg_probe/l2er_${strength}.log" 2>&1 &
done

echo "已启动 $(jobs -r | wc -l) 个进程（4 方法 × 3 强度）"
wait
echo "=== 正则化短探针跑完 ==="
