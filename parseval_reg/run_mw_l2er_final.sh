#!/usr/bin/env bash
# L2-ER 解耦版全量：lr=3e-4（MetaWorld 默认）、beta=1e-2（探针最优）、
# weight_decay 扫四挡 {1e-4, 1e-3, 1e-2, 1e-1}（原三强度 + 老师加的 0.1）。
# lecun init + ReLU + 解耦（L2 走 AdamW、ER 走单独梯度上升）。
# 1000 万步 / 10 任务，4 挡 × 6 seed = 24 run。
set -u
PY="D:/anaconda/envs/parseval/python.exe"
SEEDS="0 1 2 3 4 5"
ENV="metaworld_sequence_set0"
mkdir -p logs/mw_l2er_final
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

for wd in 0.0001 0.001 0.01 0.1; do
  for i in $SEEDS; do
    echo "  -> l2_er lr=3e-4 beta=1e-2 wd=$wd seed=$i"
    "$PY" main.py --env "$ENV" --algorithm base --repeat_idx "$i" \
        --learning_rate 0.0003 --num_steps 10000000 \
        --l2_er_weight_decay "$wd" --l2_er_beta 0.01 \
        --weight_init lecun --net_activation relu \
        --save_suffix "l2er_final_wd${wd}" \
        > "logs/mw_l2er_final/l2er_final_wd${wd}_${i}.log" 2>&1 &
  done
done

echo ""
echo "已启动 $(jobs -r | wc -l) 个进程（4 挡 × 6 seed = 24）"
wait
echo "=== L2-ER 解耦版全量（lr=3e-4, beta=1e-2, wd 四强度）跑完 ==="
