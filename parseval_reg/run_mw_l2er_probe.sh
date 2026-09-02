#!/usr/bin/env bash
# L2-ER 解耦版 lr/beta 快速探针：lr=3e-4 固定（MetaWorld 默认），扫 beta。
# 3M 步 / 3 任务，2 seed。用于判断 beta 哪个量级能学得动、task3 崩不崩。
set -u
PY="D:/anaconda/envs/parseval/python.exe"
ENV="metaworld_sequence_set0"
mkdir -p logs/mw_l2er_probe
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

for beta in 0.0001 0.001 0.01; do
  for i in 0 1; do
    echo "  -> l2_er decoupled lr=3e-4 beta=$beta wd=1e-3 seed=$i"
    "$PY" main.py --env "$ENV" --algorithm base --repeat_idx "$i" \
        --learning_rate 0.0003 --num_steps 3000000 \
        --l2_er_weight_decay 0.001 --l2_er_beta "$beta" \
        --weight_init lecun --net_activation relu \
        --save_suffix "probe_beta${beta}" \
        > "logs/mw_l2er_probe/probe_beta${beta}_${i}.log" 2>&1 &
  done
done

echo ""
echo "已启动 $(jobs -r | wc -l) 个探针进程（3 beta × 2 seed = 6）"
wait
echo "=== L2-ER 解耦版 lr/beta 探针跑完 ==="
