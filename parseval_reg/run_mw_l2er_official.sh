#!/usr/bin/env bash
# L2-ER 核心配方按官方 RL（Spectral Collapse, Table 7 best）：
#   - 初始化 lecun_uniform（所有层）
#   - 激活 ReLU
#   - lr = 1e-4（官方）
#   - beta（有效秩 er_lr）= 1e-6（官方 best）
#   - weight_decay 扫三挡 {1e-4, 1e-3(官方 best), 1e-2}（沿用原三挡，便于和旧版对比）
# 网络保留 MetaWorld 64/2 隐藏层（任务设置，不搬官方 256/1）。
# 1000 万步 / 10 任务，3 挡 × 6 seed = 18 run。
set -u
PY="D:/anaconda/envs/parseval/python.exe"
SEEDS="0 1 2 3 4 5"
ENV="metaworld_sequence_set0"
mkdir -p logs/mw_l2er_official
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

for wd in 0.0001 0.001 0.01; do
  for i in $SEEDS; do
    echo "  -> l2_er lecun+relu beta=1e-6 lr=1e-4 wd=$wd seed=$i"
    "$PY" main.py --env "$ENV" --algorithm base --repeat_idx "$i" \
        --learning_rate 0.0001 --num_steps 10000000 \
        --l2_er_weight_decay "$wd" --l2_er_beta 0.000001 \
        --weight_init lecun --net_activation relu \
        --save_suffix "l2er_official_wd${wd}" \
        > "logs/mw_l2er_official/l2er_official_wd${wd}_${i}.log" 2>&1 &
  done
done

echo ""
echo "已启动 $(jobs -r | wc -l) 个进程（3 挡 × 6 seed = 18）"
wait
echo "=== L2-ER 核心配方（lecun+relu+beta=1e-6+lr=1e-4）× 3 挡 weight_decay 跑完 ==="
