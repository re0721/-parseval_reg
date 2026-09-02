#!/usr/bin/env bash
# L2-ER 解耦版 + lecun init + tanh 激活（把激活从 ReLU 改回 tanh）。
# 与 run_mw_l2er_final.sh 唯一区别：net_activation tanh。
# 用于对照"lecun 配 ReLU vs lecun 配 tanh"哪个在 MetaWorld 上更好。
# lr=3e-4、beta=1e-2、weight_decay 四挡 {1e-4,1e-3,1e-2,1e-1}。
# 1000 万步 / 10 任务，4 挡 × 6 seed = 24 run。
set -u
PY="D:/anaconda/envs/parseval/python.exe"
SEEDS="0 1 2 3 4 5"
ENV="metaworld_sequence_set0"
mkdir -p logs/mw_l2er_tanh
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

for wd in 0.0001 0.001 0.01 0.1; do
  for i in $SEEDS; do
    echo "  -> l2_er lecun+tanh lr=3e-4 beta=1e-2 wd=$wd seed=$i"
    "$PY" main.py --env "$ENV" --algorithm base --repeat_idx "$i" \
        --learning_rate 0.0003 --num_steps 10000000 \
        --l2_er_weight_decay "$wd" --l2_er_beta 0.01 \
        --weight_init lecun --net_activation tanh \
        --save_suffix "l2er_tanh_wd${wd}" \
        > "logs/mw_l2er_tanh/l2er_tanh_wd${wd}_${i}.log" 2>&1 &
  done
done

echo ""
echo "已启动 $(jobs -r | wc -l) 个进程（4 挡 × 6 seed = 24）"
wait
echo "=== L2-ER 解耦版（lecun + tanh）四强度跑完 ==="
