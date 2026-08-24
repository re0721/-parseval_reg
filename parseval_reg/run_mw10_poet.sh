#!/usr/bin/env bash
# POET-exact 的 MetaWorld 10 任务全量 run，与 run_mw10.sh 并行跑。
#
# lr=6e-4 的依据：Gridworld 上 base=2.5e-4 时 POET 最优 5e-4（2x base），
# 且安全窗口很窄——1e-3 已有 seed 崩溃、3e-3 基本退化、1e-2 全死。
# MetaWorld base=3e-4，按同样 2x 取 6e-4。这是外推，所以要盯第一个任务：
# 若 Neumann/Cayley 的 Q 漂移过大，成功率会在任务 1 内就塌到 0 并再不恢复。
#
# 前提：poet_official.py 已加 R 缓存（rollout 阶段不再每步重算 R，实测整体快 ~3.8x）。

set -u
PY="D:/anaconda/envs/parseval/python.exe"
ENV="metaworld_sequence_set0"
mkdir -p logs/mw10
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

for i in 0 1 2 3 4 5; do
  echo "  -> poet_exact lr=0.0006 seed_idx=$i"
  "$PY" main.py --env "$ENV" --algorithm poet_exact \
      --repeat_idx "$i" --learning_rate 0.0006 \
      --save_suffix 10task \
      > "logs/mw10/poet_exact_${i}.log" 2>&1 &
done

echo ""
echo "已启动 $(jobs -r | wc -l) 个 POET-exact 进程"
wait
echo "=== POET-exact 10 任务跑完 ==="
