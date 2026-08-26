#!/usr/bin/env bash
# 重启 standard / identity 两组（之前因 gain 参数 bug 崩溃）。
set -u
PY="D:/anaconda/envs/parseval/python.exe"
SEEDS="0 1 2 3 4 5"
ENV="metaworld_sequence_set0"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

for init in standard identity; do
  for i in $SEEDS; do
    echo "  -> pion init=$init seed_idx=$i (restart)"
    "$PY" main.py --env "$ENV" --algorithm pion \
        --repeat_idx "$i" --learning_rate 0.002 --num_steps 10000000 \
        --weight_init "$init" --save_suffix "init_${init}_ppo" \
        > "logs/mw_init_ppo/pion_${init}_${i}.log" 2>&1 &
  done
done

echo "已重启 $(jobs -r | wc -l) 个进程（standard + identity 各 6 seed）"
wait
echo "=== standard/identity 重启跑完 ==="
