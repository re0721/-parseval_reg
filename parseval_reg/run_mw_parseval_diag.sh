#!/usr/bin/env bash
# Parseval 按论文补对角层重跑：add_diag_layer=True，四档强度 {1e-4,1e-3,1e-2,1e-1}。
# 之前三档是 --algorithm base --parseval_reg {强度}（add_diag_layer=False，漏了对角层）；
# 论文原版 --algorithm parseval 会 add_diag_layer=True。这里补回来。
# orthogonal + tanh + lr=3e-4。save_suffix 用 parseval_diag_{强度} 与旧版区分。
# 1000 万步 / 10 任务，4 挡 × 6 seed = 24 run。
set -u
PY="D:/anaconda/envs/parseval/python.exe"
SEEDS="0 1 2 3 4 5"
ENV="metaworld_sequence_set0"
mkdir -p logs/reg_full
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

for strength in 0.0001 0.001 0.01 0.1; do
  for i in $SEEDS; do
    echo "  -> parseval diag strength=$strength seed=$i"
    "$PY" main.py --env "$ENV" --algorithm base --repeat_idx "$i" \
        --learning_rate 0.0003 --num_steps 10000000 \
        --parseval_reg "$strength" --add_diag_layer True \
        --save_suffix "parseval_diag_${strength}" \
        > "logs/reg_full/parseval_diag_${strength}_${i}.log" 2>&1 &
  done
done

echo ""
echo "已启动 $(jobs -r | wc -l) 个进程（4 挡 × 6 seed = 24）"
wait
echo "=== Parseval（加对角层）四强度跑完 ==="
