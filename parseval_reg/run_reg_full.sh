#!/usr/bin/env bash
# 正则化完整版：Parseval/Spectral/ISO/L2-ER × 3 强度 × 6 seed = 72 run。
# MetaWorld 1000 万步 / 10 任务，统一正交初始化，lr 用默认 0.0003。
# 分批：每个方法 18 个进程并行（< 32 核），方法之间顺序。
set -u
PY="D:/anaconda/envs/parseval/python.exe"
ENV="metaworld_sequence_set0"
STEPS=10000000
SEEDS="0 1 2 3 4 5"
mkdir -p logs/reg_full
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

launch () {
  local name=$1 reg_flag=$2; shift 2
  local extra="$*"
  for strength in 0.0001 0.001 0.01; do
    for i in $SEEDS; do
      "$PY" main.py --env "$ENV" --algorithm base --repeat_idx "$i" --num_steps "$STEPS" \
          --weight_init orthogonal "$reg_flag" "$strength" $extra --save_suffix "${name}_${strength}" \
          > "logs/reg_full/${name}_${strength}_${i}.log" 2>&1 &
    done
  done
  echo "已启动 $(jobs -r | wc -l) 个 ${name} 进程"
  wait
  echo "=== ${name} 跑完 ==="
}

launch parseval --parseval_reg
launch spectral --spectral_reg
launch iso --iso_reg
launch l2er --l2_er_weight_decay "--l2_er_beta 0.01"

echo "=== 全部 72 run 跑完 ==="
