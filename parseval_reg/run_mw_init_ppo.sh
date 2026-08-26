#!/usr/bin/env bash
# MetaWorld（PPO，rpo_alpha=0）4 种初始化消融：xavier / orthogonal / standard / identity。
# 固定 Pion，扫初始谱形状。老师要求：MetaWorld 也用 PPO（不是 RPO）。
# lr = 0.002（沿用之前 RPO 下的 pion lr，先不重扫）。
set -u
PY="D:/anaconda/envs/parseval/python.exe"
SEEDS="0 1 2 3 4 5"
ENV="metaworld_sequence_set0"
mkdir -p logs/mw_init_ppo
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

for init in xavier orthogonal standard identity; do
  for i in $SEEDS; do
    echo "  -> pion init=$init seed_idx=$i"
    "$PY" main.py --env "$ENV" --algorithm pion \
        --repeat_idx "$i" --learning_rate 0.002 --num_steps 10000000 \
        --weight_init "$init" --save_suffix "init_${init}_ppo" \
        > "logs/mw_init_ppo/pion_${init}_${i}.log" 2>&1 &
  done
done

echo ""
echo "已启动 $(jobs -r | wc -l) 个进程（1000 万步 / 10 任务，4 init × 6 seed）"
wait
echo "=== MetaWorld PPO 4 初始化消融跑完 ==="
