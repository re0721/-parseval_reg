#!/usr/bin/env bash
# ISO 对齐版：激活 ReLU（从 tanh 改），iso_reg 四档强度。
# orthogonal √2 + 输出 0.01 + lr=3e-4，add_diag_layer=False（ISO 论文不用对角层）。
# 与之前 ISO(tanh) 唯一区别：net_activation relu。
# 1000 万步 / 10 任务，4 挡 × 6 seed = 24 run。
set -u
PY="D:/anaconda/envs/parseval/python.exe"
SEEDS="0 1 2 3 4 5"
ENV="metaworld_sequence_set0"
mkdir -p logs/reg_full
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

for s in 0.0001 0.001 0.01 0.1; do
  for i in $SEEDS; do
    echo "  -> iso relu strength=$s seed=$i"
    "$PY" main.py --env "$ENV" --algorithm base --repeat_idx "$i" \
        --learning_rate 0.0003 --num_steps 10000000 \
        --iso_reg "$s" --net_activation relu \
        --save_model_freq 5000000 \
        --save_suffix "iso_relu_${s}" \
        > "logs/reg_full/iso_relu_${s}_${i}.log" 2>&1 &
  done
done

echo ""
echo "已启动 $(jobs -r | wc -l) 个进程（4 挡 × 6 seed = 24）"
wait
echo "=== ISO 对齐版（ReLU）四档强度跑完 ==="
