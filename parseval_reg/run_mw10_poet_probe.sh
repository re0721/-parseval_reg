#!/usr/bin/env bash
# POET-exact 在 MetaWorld 上的 lr 探针：只跑 2 个任务（200 万步）就够判断会不会数值爆炸。
#
# 为什么必须探：Gridworld 上 POET 对 lr 极敏感（1e-2 全死、5e-4 最优），
# 因为 Cayley 的 Q 单向漂移，lr 越大越早越过 ||Q||~1。MetaWorld 的 base lr 是 3e-4，
# 按 Gridworld 的 2x 比例外推应为 6e-4，但这只是外推，不能拿 5 小时去赌。
#
# 注意：短探针只用于「判断炸不炸 / 选 lr」，不能用于评价最终性能
# —— Gridworld 的教训是 OFT 在 4 任务探针里 0.63-0.79，全量 20 任务只有 0.41。

set -u
PY="D:/anaconda/envs/parseval/python.exe"
ENV="metaworld_sequence_set0"
mkdir -p logs/mw10_probe
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

for cfg in "0.0003 lr3e4" "0.0006 lr6e4" "0.0015 lr15e4"; do
  set -- $cfg
  for i in 0 1; do
    echo "  -> poet_exact lr=$1 seed_idx=$i"
    "$PY" main.py --env "$ENV" --algorithm poet_exact \
        --repeat_idx "$i" --learning_rate "$1" --num_steps 2000000 \
        --save_suffix "probe_$2" \
        > "logs/mw10_probe/poet_exact_${2}_${i}.log" 2>&1 &
  done
done

echo ""
echo "已启动 $(jobs -r | wc -l) 个探针进程（200 万步 / 2 任务）"
wait
echo "=== POET-exact lr 探针跑完 ==="
