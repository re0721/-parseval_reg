#!/usr/bin/env bash
# lr 探针：Gridworld 16 万步（4 个任务）× 2 seed，找出 POET/OFT 不炸的 lr。
#
# 背景：gw_poet_v3 用 lr=0.01 全 6 seed 死在 success=0.0（eval return -16.96 ± 0.0，
# 方差为零 = 确定性退化策略）。根因是 poet_official.py 的 4 阶 Neumann 截断 Cayley
# 在 ||Q|| ≳ 1 后失去正交性，sigma_max 炸到 1e2~1e6（见 diag_poet_neumann.py）。
# OFT 用精确求逆 Cayley，任意 Q 都严格正交，所以没炸、只是平庸（~0.36）。
#
# poet_exact 是对照组：只把 Neumann 换成精确求逆，其余（双侧分块/SPO 置换/W0 冻结）不变。
# 如果它在 lr=0.01 下活着而 poet 死了 -> 零分是数值问题，不是方法失败。

set -u
PY="D:/anaconda/envs/parseval/python.exe"
STEPS=160000
SEEDS="0 1"
mkdir -p logs/probe
# 网络很小（64 宽），BLAS 多线程只会让 12 个进程互相抢核
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

launch () {  # $1=algorithm  $2=lr  $3=tag  $4=extra args
  for i in $SEEDS; do
    echo "  -> $1 lr=$2 seed_idx=$i  (logs/probe/${1}_${3}_${i}.log)"
    "$PY" main.py --env gridworld_ninerooms --algorithm "$1" \
        --repeat_idx "$i" --learning_rate "$2" --num_steps $STEPS \
        --save_suffix "probe_$3" $4 \
        > "logs/probe/${1}_${3}_${i}.log" 2>&1 &
  done
}

echo "=== POET (Neumann 截断) lr 扫描 ==="
launch poet       0.0005 lr5e4  ""
launch poet       0.001  lr1e3  ""
launch poet       0.003  lr3e3  ""

echo "=== OFT (精确 Cayley) lr 扫描 ==="
launch oft        0.003  lr3e3  "--oet_num_blocks 9"
launch oft        0.01   lr1e2  "--oet_num_blocks 9"

echo "=== POET-exact 对照组（大 lr 下是否存活）==="
launch poet_exact 0.01   lr1e2  ""

echo ""
echo "已启动 $(jobs -r | wc -l) 个进程，等待全部完成..."
wait
echo "=== 探针全部跑完 ==="
