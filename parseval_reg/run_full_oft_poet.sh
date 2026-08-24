#!/usr/bin/env bash
# 全量重跑：Gridworld 80 万步 / 20 任务 / 6 seed(123-128)。
# lr 由 16 万步探针选出（见 analyze_probe.py 输出）：
#   POET(Neumann): 5e-4 最好(steady 0.71/0.91)，1e-3 不稳，3e-3 退化，1e-2 全死
#   OFT(精确Cayley): 3e-3 好(0.63/0.79)，1e-2 退化
# poet_exact_lr1e2 是关键对照：与已跑完的 gw_poet_v3(Neumann, lr=1e-2, 全 6 seed 恒为 0)
# 唯一差别就是 Cayley 精确求逆 vs 4 阶 Neumann 截断。

set -u
PY="D:/anaconda/envs/parseval/python.exe"
SEEDS="0 1 2 3 4 5"
mkdir -p logs/full
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

launch () {  # $1=algorithm $2=lr $3=tag $4=extra
  for i in $SEEDS; do
    echo "  -> $1 lr=$2 seed_idx=$i"
    "$PY" main.py --env gridworld_ninerooms --algorithm "$1" \
        --repeat_idx "$i" --learning_rate "$2" \
        --save_suffix "full_$3" $4 \
        > "logs/full/${1}_${3}_${i}.log" 2>&1 &
  done
}

launch poet       0.0005 lr5e4 ""                      # 探针最优
launch oft        0.003  lr3e3 "--oet_num_blocks 9"    # 探针最优
launch poet_exact 0.01   lr1e2 ""                      # 数值对照（对上 v3 的全零）
launch poet_exact 0.0005 lr5e4 ""                      # 好 lr 下 exact 是否也更强

echo ""
echo "已启动 $(jobs -r | wc -l) 个进程（80 万步，预计 1-1.5 小时）"
wait
echo "=== 全量实验跑完 ==="
