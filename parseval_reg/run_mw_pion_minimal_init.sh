#!/usr/bin/env bash
# 最简 Pion（乘法版，删动量/RMS）在 MetaWorld PPO 下跑 4 种初始化。
# lr=0.1（lr 扫描确定的最优点）。1000 万步 / 10 任务 / 6 seed。
set -u
PY="D:/anaconda/envs/parseval/python.exe"
SEEDS="0 1 2 3 4 5"
ENV="metaworld_sequence_set0"
mkdir -p logs/mw_pion_minimal_init
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

for init in xavier orthogonal standard identity; do
  for i in $SEEDS; do
    echo "  -> pion_minimal lr=0.1 init=$init seed_idx=$i"
    "$PY" main.py --env "$ENV" --algorithm pion_minimal \
        --repeat_idx "$i" --learning_rate 0.1 --num_steps 10000000 \
        --weight_init "$init" --save_suffix "minimal_init_${init}" \
        > "logs/mw_pion_minimal_init/pion_minimal_${init}_${i}.log" 2>&1 &
  done
done

echo ""
echo "已启动 $(jobs -r | wc -l) 个进程（1000 万步 / 10 任务，4 init × 6 seed）"
wait
echo "=== 最简 Pion MetaWorld 4 初始化跑完 ==="
