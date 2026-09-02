#!/usr/bin/env bash
# POET (Neumann + SPO) 降 lr 重跑：lr=2e-4，4 初始化 × 6 seed，1000 万步 / 10 任务。
#
# 背景：官方/pure 两版都用 4 阶 Neumann 截断 Cayley，lr=5e-4 下 Q 单向漂移越过
# ‖Q‖~1 时 R 数值爆炸（见 diag_poet_neumann.py：lr=5e-4、5000 步漂移即 σ_max=4.5），
# 谱根本没被锁住 → tanh 饱和 → 非 identity 初始化崩到 0.2-0.4。
# 降到 2e-4 让 ‖Q‖ 全程远低于 1（漂移模型下 10M 步 ≈ 0.98，随机游走则更小）。
#
# 与之前 official（lr=5e-4）直接可比，save_suffix 换新以区分。
set -u
PY="D:/anaconda/envs/parseval/python.exe"
SEEDS="0 1 2 3 4 5"
ENV="metaworld_sequence_set0"
mkdir -p logs/mw_poet_neumann_lowlr
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

for init in xavier orthogonal standard identity; do
  for i in $SEEDS; do
    echo "  -> poet(Neumann+SPO) lr=0.0002 init=$init seed=$i"
    "$PY" main.py --env "$ENV" --algorithm poet --repeat_idx "$i" \
        --learning_rate 0.0002 --num_steps 10000000 --weight_init "$init" \
        --save_suffix "lr2e4_init_${init}" \
        > "logs/mw_poet_neumann_lowlr/poet_${init}_${i}.log" 2>&1 &
  done
done

echo ""
echo "已启动 $(jobs -r | wc -l) 个进程（1 方法 × 4 init × 6 seed = 24）"
wait
echo "=== POET(Neumann) 降 lr 4 初始化完整版跑完 ==="
