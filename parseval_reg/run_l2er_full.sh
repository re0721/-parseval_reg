#!/usr/bin/env bash
# L2-ER 完整版：weight_decay {1e-4,1e-3,1e-2} x 6 seed x 1000万步/10任务。
# beta 固定 0.001（探针定），统一正交初始化，PPO。
set -u
PY="D:/anaconda/envs/parseval/python.exe"
ENV="metaworld_sequence_set0"
STEPS=10000000
SEEDS="0 1 2 3 4 5"
mkdir -p logs/reg_full
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

for wd in 0.0001 0.001 0.01; do
  for i in $SEEDS; do
    "$PY" main.py --env "$ENV" --algorithm base --repeat_idx "$i" --num_steps "$STEPS" \
        --weight_init orthogonal --l2_er_weight_decay "$wd" --l2_er_beta 0.001 \
        --save_suffix "l2er_b0.001_wd${wd}" \
        > "logs/reg_full/l2er_b0.001_wd${wd}_${i}.log" 2>&1 &
  done
done

echo "已启动 18 个 L2-ER 完整版 run"
wait
echo "=== L2-ER 完整版全部跑完 ==="
