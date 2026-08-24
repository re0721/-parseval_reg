#!/usr/bin/env bash
# MetaWorld 完整 10 任务序列（论文正式设置）：1000 万步、每 100 万步换任务。
# 之前只跑了 500 万步 = 5 任务，是截短版。
#
# 序列用 set0。注意 metaworld_env.py:103 是 RPO10_SEQ[env_set_id-1]，
# 所以 set0 实际取的是第 20 条序列（不是第一条）——沿用旧实验以便对照。
#
# lr 来源：base/parseval 0.0003、pion 0.002 均沿用旧 run；
# POET-exact 的 MetaWorld lr 未知，另由 run_mw10_poet_probe.sh 先扫。
#
# 必须限制 BLAS 线程：实测单进程 SPS 1009，旧 run 因线程超订只有 291。

set -u
PY="D:/anaconda/envs/parseval/python.exe"
SEEDS="0 1 2 3 4 5"
ENV="metaworld_sequence_set0"
mkdir -p logs/mw10
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

launch () {  # $1=algorithm $2=lr
  for i in $SEEDS; do
    echo "  -> $1 lr=$2 seed_idx=$i"
    "$PY" main.py --env "$ENV" --algorithm "$1" \
        --repeat_idx "$i" --learning_rate "$2" \
        --save_suffix 10task \
        > "logs/mw10/${1}_${i}.log" 2>&1 &
  done
}

launch base     0.0003
launch parseval 0.0003
launch pion     0.002

echo ""
echo "已启动 $(jobs -r | wc -l) 个进程（1000 万步 / 10 任务，预计 5-7 小时）"
wait
echo "=== MetaWorld 10 任务（base/parseval/pion）跑完 ==="
