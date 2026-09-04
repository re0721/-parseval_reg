#!/usr/bin/env bash
# Spectral（谱正则）对齐版：ReLU + 真正的 seed 1/2/3，四档强度。
# 论文激活是 ReLU（之前的 spectral 是 tanh，这里对齐论文）。
# 初始化：论文未明示，这里沿用 orthogonal（满足动态等距、奇异值≈1，与论文出发点一致）。
# lr=3e-4，add_diag_layer=False。1000 万步 / 10 任务，4 挡 × 3 seed = 12 run。
set -u
PY="D:/anaconda/envs/parseval/python.exe"
SEEDS="1 2 3"
ENV="metaworld_sequence_set0"
mkdir -p logs/reg_full
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

for s in 0.0001 0.001 0.01 0.1; do
  for seed in $SEEDS; do
    echo "  -> spectral relu strength=$s seed=$seed"
    "$PY" main.py --env "$ENV" --algorithm base --repeat_idx 0 --seed "$seed" \
        --learning_rate 0.0003 --num_steps 10000000 \
        --spectral_reg "$s" --net_activation relu \
        --wandb --wandb_online \
        --save_suffix "spectral_relu_${s}_seed${seed}" \
        > "logs/reg_full/spectral_relu_${s}_seed${seed}.log" 2>&1 &
  done
done

echo ""
echo "已启动 $(jobs -r | wc -l) 个进程（4 挡 × 3 seed = 12）"
wait
echo "=== Spectral 对齐版（ReLU + seed 1/2/3）四档强度跑完 ==="
