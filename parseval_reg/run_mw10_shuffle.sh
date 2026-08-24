#!/usr/bin/env bash
# MetaWorld 10-task, SHUFFLED order (task-order robustness check).
# Same 10 tasks as set0, order shuffled by numpy.default_rng(42).permutation(10):
#   plate-slide-back -> coffee-button -> reach -> sweep-into -> door-close ->
#   plate-slide-side -> plate-slide-back-side -> drawer-close -> button-press -> reach-wall
# lr / methods identical to run_mw10.sh + run_mw10_poet.sh, so results are directly
# comparable to the original set0 run. 1000 万步 / 10 任务 / 6 seed (123-128)。

set -u
PY="D:/anaconda/envs/parseval/python.exe"
SEEDS="0 1 2 3 4 5"
ENV="metaworld_sequence_shuffle"
mkdir -p logs/mw10_shuffle
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

launch () {  # $1=algorithm $2=lr
  for i in $SEEDS; do
    echo "  -> $1 lr=$2 seed_idx=$i"
    "$PY" main.py --env "$ENV" --algorithm "$1" \
        --repeat_idx "$i" --learning_rate "$2" \
        --save_suffix shuffle \
        > "logs/mw10_shuffle/${1}_${i}.log" 2>&1 &
  done
}

launch base       0.0003
launch parseval   0.0003
launch pion       0.002
launch poet_exact 0.0006

echo ""
echo "已启动 $(jobs -r | wc -l) 个进程（4 方法 x 6 seed = 24，1000 万步，预计 ~9-10 小时）"
wait
echo "=== MetaWorld 10-task SHUFFLED（base/parseval/pion/poet_exact）跑完 ==="
