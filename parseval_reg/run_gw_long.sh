#!/usr/bin/env bash
# Gridworld 长训练：每个任务多训 4 万步 → 80k/任务 × 20 任务 = 160 万步（原 40k/task 的 2 倍）。
# 老师要求：验证"base 差是因为塑性丧失，还是每任务训练不足"。
# 80k/task 下 Pion/POET 约 7.4 小时，四个方法全跑。
set -u
PY="D:/anaconda/envs/parseval/python.exe"
SEEDS="0 1 2 3 4 5"
ENV="gridworld_ninerooms"
CF=80000
NS=1600000
mkdir -p logs/gw_long
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

launch () {  # $1=algorithm $2=lr
  for i in $SEEDS; do
    echo "  -> $1 lr=$2 seed_idx=$i"
    "$PY" main.py --env "$ENV" --algorithm "$1" \
        --repeat_idx "$i" --learning_rate "$2" \
        --change_freq $CF --num_steps $NS \
        --save_suffix long80k \
        > "logs/gw_long/${1}_${i}.log" 2>&1 &
  done
}

launch base       0.00025
launch parseval   0.00025
launch pion       0.001
launch poet_exact 0.0005

echo ""
echo "已启动 $(jobs -r | wc -l) 个进程（4 方法，160 万步，预计 ~7.4 小时）"
wait
echo "=== Gridworld 长训练（80k/任务）跑完 ==="
