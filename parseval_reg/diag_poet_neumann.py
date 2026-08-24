"""诊断：POET 的 Neumann 截断 Cayley 在大 lr 下是否失去正交性并爆炸。

对比 OFT（精确求逆 Cayley，任意 Q 都严格正交）与 POET（4 阶 Neumann 截断，
只在 ||Q|| 小时才近似正交）。Adam 每步幅度 ~ lr，累积到 T 步后参数尺度介于
  随机游走 lr*sqrt(T)  ~  单向漂移 lr*T
之间。测量两种 Cayley 的：
  - ||R^T R - I||_F   正交性偏差（0 = 严格正交）
  - sigma_max(R)      谱范数（>1 = 放大信号 -> tanh 饱和 -> 策略死掉）
"""
import torch
from poet_official import _skew_symmetric, _cayley_neumann

torch.manual_seed(0)

BS = 25                      # POET 在 225 维输入上自动选的 block_size
n_elem = BS * (BS - 1) // 2
rows, cols = torch.triu_indices(BS, BS, 1)


def cayley_exact(Q):
    I = torch.eye(Q.shape[-1]).unsqueeze(0)
    return (I + Q) @ torch.inverse(I - Q)


def metrics(R):
    I = torch.eye(R.shape[-1]).unsqueeze(0)
    orth = torch.linalg.matrix_norm(R.transpose(-2, -1) @ R - I).max().item()
    smax = torch.linalg.matrix_norm(R, ord=2).max().item()
    return orth, smax


print(f"{'lr':>7} {'steps':>6} {'模式':>6} {'||Q||':>8} | "
      f"{'POET orth_err':>14} {'POET smax':>11} | {'OFT orth_err':>13} {'OFT smax':>9}")
print("-" * 92)

for lr in [0.0005, 0.001, 0.003, 0.01]:
    for steps in [500, 5000]:
        for mode, scale in [("游走", lr * steps ** 0.5), ("漂移", lr * steps)]:
            vec = torch.randn(1, n_elem)
            vec = vec / vec.norm() * scale          # 参数向量总范数 = scale
            Q = _skew_symmetric(vec, BS, rows, cols)
            qn = torch.linalg.matrix_norm(Q, ord=2).max().item()
            e_p, s_p = metrics(_cayley_neumann(Q))
            e_o, s_o = metrics(cayley_exact(Q))
            flag = "  <-- 爆炸" if s_p > 2 else ""
            print(f"{lr:>7} {steps:>6} {mode:>6} {qn:>8.3f} | "
                  f"{e_p:>14.3e} {s_p:>11.3f} | {e_o:>13.3e} {s_o:>9.3f}{flag}")
    print()
