#!/usr/bin/env bash
# pion_minimal_rms (lr=0.001) + 纯正交 POET (lr=0.0005)，4 初始化 × 6 seed。
# 1000 万步 / 10 任务。lr 由短探针扫描确定。
set -u
PY="D:/anaconda/envs/parseval/python.exe"
SEEDS="0 1 2 3 4 5"
ENV="metaworld_sequence_set0"
mkdir -p logs/mw_pion_minimal_rms logs/mw_poet_pure
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

for init in xavier orthogonal standard identity; do
  for i in $SEEDS; do
    echo "  -> pion_minimal_rms lr=0.001 init=$init seed=$i"
    "$PY" main.py --env "$ENV" --algorithm pion_minimal_rms --repeat_idx "$i" \
        --learning_rate 0.001 --num_steps 10000000 --weight_init "$init" \
        --save_suffix "rms_init_${init}" \
        > "logs/mw_pion_minimal_rms/pion_minimal_rms_${init}_${i}.log" 2>&1 &
    echo "  -> poet(pure) lr=0.0005 init=$init seed=$i"
    "$PY" main.py --env "$ENV" --algorithm poet --repeat_idx "$i" \
        --learning_rate 0.0005 --num_steps 10000000 --weight_init "$init" \
        --save_suffix "pure_init_${init}" \
        > "logs/mw_poet_pure/poet_${init}_${i}.log" 2>&1 &
  done
done

echo ""
echo "已启动 $(jobs -r | wc -l) 个进程（2 方法 × 4 init × 6 seed = 48）"
wait
echo "=== pion_minimal_rms + 纯正交 POET 4 初始化完整版跑完 ==="
