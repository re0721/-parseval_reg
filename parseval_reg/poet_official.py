"""Faithful re-implementation of the official POET layer.

Copied from Sphere-AI-Lab/poet (poet_torch/core/ops.py and
poet_torch/layers/poet_linear.py), using the PyTorch fallback:

  - skew-symmetric Q built from upper-triangular parameters
  - Neumann-series Cayley: R = I + 2Q + 2Q^2 + 2Q^3 + 2Q^4
  - block-diagonal R_in and R_out (applied to input / output features)
  - random permutation (SPO block-stochastic)

W_eff = R_out^T @ W_0 @ R_in^T, with W_0 frozen.

NOTE: the official code requires a SINGLE block_size dividing both in_features and
out_features. The continual-RL MLP (e.g. 225->64) has no common divisor, so we
allow separate block sizes for the input and output sides (a minor, necessary
deviation). The merge-then-reinitialize step is omitted here (W_0 is frozen
Gaussian, and Q stays small in this setting).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _skew_symmetric(vec, block_size, rows, cols):
    """Build skew-symmetric matrices from upper-triangular vector params."""
    batch = vec.shape[0]
    Q = vec.new_zeros(batch, block_size, block_size)
    Q[:, rows, cols] = vec
    Q = Q - Q.transpose(-2, -1)
    return Q


def _cayley_neumann(Q):
    """Neumann-series Cayley transform: (I+Q)(I-Q)^-1 ~= I + 2Q + 2Q^2 + 2Q^3 + 2Q^4.

    WARNING: the truncation is only near-orthogonal while ||Q|| << 1. Once ||Q||
    drifts past ~1 the series diverges and sigma_max(R) blows up by orders of
    magnitude (see diag_poet_neumann.py), which saturates tanh and kills the
    policy. Use _cayley_exact (--algorithm poet_exact) to rule this out.
    """
    Q2 = Q @ Q
    R = 2.0 * (Q + Q2 + Q2 @ Q) + 2.0 * Q2 @ Q2
    R.diagonal(dim1=-2, dim2=-1).add_(1.0)
    return R


def _cayley_exact(Q):
    """Exact Cayley transform R = (I+Q)(I-Q)^-1 -- orthogonal for ANY skew-symmetric Q."""
    I = torch.eye(Q.shape[-1], device=Q.device, dtype=Q.dtype).expand_as(Q)
    return (I + Q) @ torch.linalg.solve(I - Q, I)


def _block_diag_apply(x, R_blocks, block_size):
    """Apply block-diagonal R_blocks (num_blocks, b, b) on the last dim of x."""
    bdims = x.shape[:-1]
    xr = x.view(*bdims, -1, block_size)
    xr = torch.einsum("...rk,rkc->...rc", xr, R_blocks)
    return xr.contiguous().view(*bdims, -1)


def _auto_block_size(n, max_bs=32):
    """Largest divisor of n that is <= max_bs (and >= 2); fallback to n."""
    for bs in range(min(n, max_bs), 1, -1):
        if n % bs == 0:
            return bs
    return n


class PoetOfficialLinear(nn.Module):
    def __init__(self, in_features, out_features, block_size_in=None, block_size_out=None, bias=True,
                 exact_cayley=False):
        super().__init__()
        self.exact_cayley = exact_cayley
        self._R_cache = None   # (param_versions, R_out, R_in)，见 _get_R
        self._W_cache = None   # (param_versions, W_eff, b_eff)，见 _effective_weight
        block_size_in = block_size_in or _auto_block_size(in_features)
        block_size_out = block_size_out or _auto_block_size(out_features)
        assert in_features % block_size_in == 0, f"{in_features} % {block_size_in}"
        assert out_features % block_size_out == 0, f"{out_features} % {block_size_out}"
        self.in_features = in_features
        self.out_features = out_features
        self.block_size_in = block_size_in
        self.block_size_out = block_size_out
        self.num_in_blocks = in_features // block_size_in
        self.num_out_blocks = out_features // block_size_out

        # frozen base weight (Gaussian init)
        self.weight = nn.Parameter(torch.empty(out_features, in_features), requires_grad=False)
        torch.nn.init.xavier_normal_(self.weight, gain=1.0)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

        # trainable skew-symmetric params (upper-triangular elements)
        n_elem_in = block_size_in * (block_size_in - 1) // 2
        n_elem_out = block_size_out * (block_size_out - 1) // 2
        self.oft_R_in = nn.Parameter(torch.zeros(self.num_in_blocks, n_elem_in))
        self.oft_R_out = nn.Parameter(torch.zeros(self.num_out_blocks, n_elem_out))

        rows_in, cols_in = torch.triu_indices(block_size_in, block_size_in, 1)
        rows_out, cols_out = torch.triu_indices(block_size_out, block_size_out, 1)
        self.register_buffer("rows_in", rows_in)
        self.register_buffer("cols_in", cols_in)
        self.register_buffer("rows_out", rows_out)
        self.register_buffer("cols_out", cols_out)

        # random permutations (SPO)
        perm_in = torch.randperm(in_features)
        perm_out = torch.randperm(out_features)
        self.register_buffer("perm_in", perm_in)
        self.register_buffer("perm_out", perm_out)

    def _get_R(self):
        # R 只依赖 Q 参数，而 Q 只在优化器 step 时（原地）改变。rollout/eval 全程
        # no_grad，且每步都要前向一次（MetaWorld 一轮 rollout 有 2048 步），
        # 不缓存的话同一个 R 会被重算上千遍——实测占了前向 ~90% 的时间。
        # 用参数的 _version 做键：优化器原地更新会自动 bump，缓存随之失效。
        grad_on = torch.is_grad_enabled()
        if not grad_on:
            ver = (self.oft_R_in._version, self.oft_R_out._version)
            if self._R_cache is not None and self._R_cache[0] == ver:
                return self._R_cache[1], self._R_cache[2]

        Q_in = _skew_symmetric(self.oft_R_in, self.block_size_in, self.rows_in, self.cols_in)
        Q_out = _skew_symmetric(self.oft_R_out, self.block_size_out, self.rows_out, self.cols_out)
        cayley = _cayley_exact if self.exact_cayley else _cayley_neumann
        R_in = cayley(Q_in)    # (num_in_blocks, b_in, b_in)
        R_out = cayley(Q_out)  # (num_out_blocks, b_out, b_out)

        if not grad_on:
            self._R_cache = (ver, R_out, R_in)
        return R_out, R_in

    def _forward_full(self, x):
        """原始实现：置换 -> 分块 R_in -> W0 -> 分块 R_out -> 置换。"""
        R_out, R_in = self._get_R()
        x = x.index_select(-1, self.perm_in)
        x = _block_diag_apply(x, R_in, self.block_size_in)
        y = F.linear(x, self.weight, self.bias)
        y = _block_diag_apply(y, R_out, self.block_size_out)
        y = y.index_select(-1, self.perm_out)
        return y

    def _effective_weight(self):
        """把整条前向塌缩成等效的 (W_eff, b_eff)。

        整条链路（置换、分块正交、W0、bias）全是仿射的，所以在 no_grad 下
        可以一次性求出等效仿射映射，之后每步只做一次 F.linear。
        直接把现有前向作用在单位阵上求列，保证与 _forward_full 逐位一致，
        不用手推置换/分块矩阵的转置关系（那里极易出错）。
        """
        ver = (self.oft_R_in._version, self.oft_R_out._version)
        if self._W_cache is not None and self._W_cache[0] == ver:
            return self._W_cache[1], self._W_cache[2]
        eye = torch.eye(self.in_features, device=self.weight.device, dtype=self.weight.dtype)
        zero = torch.zeros(1, self.in_features, device=self.weight.device, dtype=self.weight.dtype)
        b_eff = self._forward_full(zero)[0]                 # (out,)
        W_eff = (self._forward_full(eye) - b_eff).t().contiguous()   # (out, in)
        self._W_cache = (ver, W_eff, b_eff)
        return W_eff, b_eff

    def forward(self, x):
        if torch.is_grad_enabled():
            return self._forward_full(x)                    # 训练更新：需要完整计算图
        W_eff, b_eff = self._effective_weight()             # rollout/eval：单次 F.linear
        return F.linear(x, W_eff, b_eff)
