"""Pion: spectrum-preserving optimizer (aligned with the official implementation).

Follows `megatron/core/optimizer/pion.py` (Sphere-AI-Lab/pion). For each 2-D
weight matrix W, the update is the ADDITIVE Lie-generator form:

    W <- W + W @ (exp(A_in) - I) + (exp(A_out) - I) @ W
        = W @ exp(A_in) + exp(A_out) @ W - W

where exp(A) is a truncated Taylor series of degree 2: exp(A) ≈ I + A + A^2/2.
A_in / A_out are (momentum-normalized) skew-symmetric Lie-algebra elements,
and an RMS scaling keeps the induced update magnitude consistent across layers.

1-D parameters (biases, log-std) fall back to Adam.
"""

import math

import torch
from torch.optim import Optimizer


class Pion(Optimizer):
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.999), eps=1e-8, rms_constant=0.2,
                 multiplicative=False):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0 or not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameters: {betas}")
        defaults = dict(lr=lr, betas=betas, eps=eps, rms_constant=rms_constant,
                        multiplicative=multiplicative)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            c = group["rms_constant"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]

                # Pion only on hidden-layer weight matrices (both dims > 8),
                # matching Parseval's "every layer except the last".
                if grad.dim() < 2 or min(p.shape) <= 8:
                    # ---- Adam for 1-D / output-layer params ----
                    if len(state) == 0:
                        state["step"] = 0
                        state["m"] = torch.zeros_like(p)
                        state["v"] = torch.zeros_like(p)
                    state["step"] += 1
                    m, v = state["m"], state["v"]
                    m.mul_(beta1).add_(grad, alpha=1 - beta1)
                    v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                    m_hat = m / (1 - beta1 ** state["step"])
                    v_hat = v / (1 - beta2 ** state["step"])
                    p.add_(m_hat / (v_hat.sqrt() + eps), alpha=-lr)
                else:
                    # ---- Pion (additive update, official) ----
                    W = p
                    G = grad
                    Gin = W.t() @ G - G.t() @ W     # (din, din) skew-symmetric
                    Gout = G @ W.t() - W @ G.t()    # (dout, dout) skew-symmetric

                    if len(state) == 0:
                        state["step"] = 0
                        state["m_in"] = torch.zeros_like(Gin)
                        state["m_out"] = torch.zeros_like(Gout)
                        state["v_in"] = torch.zeros_like(Gin)
                        state["v_out"] = torch.zeros_like(Gout)
                    state["step"] += 1

                    m_in = state["m_in"].mul_(beta1).add_(Gin, alpha=1 - beta1)
                    m_out = state["m_out"].mul_(beta1).add_(Gout, alpha=1 - beta1)
                    v_in = state["v_in"].mul_(beta2).addcmul_(Gin, Gin, value=1 - beta2)
                    v_out = state["v_out"].mul_(beta2).addcmul_(Gout, Gout, value=1 - beta2)

                    bc1 = 1 - beta1 ** state["step"]
                    bc2 = 1 - beta2 ** state["step"]
                    A_in = -(m_in / bc1) / (torch.sqrt(v_in / bc2) + eps)
                    A_out = -(m_out / bc1) / (torch.sqrt(v_out / bc2) + eps)

                    # RMS-controlled scale consistency (official)
                    dout, din = W.shape
                    denom = torch.norm(A_out @ W + W @ A_in) + eps
                    eta = lr * c * math.sqrt(dout * din) / denom

                    if group.get("multiplicative", False):
                        # multiplicative variant (= paper Algorithm 1): E_out @ W @ E_in
                        E_out = torch.eye(dout, device=W.device, dtype=W.dtype) + eta * A_out + 0.5 * (eta * A_out) @ (eta * A_out)
                        E_in = torch.eye(din, device=W.device, dtype=W.dtype) + eta * A_in + 0.5 * (eta * A_in) @ (eta * A_in)
                        p.copy_(E_out @ W @ E_in)
                    else:
                        # additive update (official): W + W @ exp_in + exp_out @ W
                        exp_in = eta * A_in + 0.5 * (eta * A_in) @ (eta * A_in)
                        exp_out = eta * A_out + 0.5 * (eta * A_out) @ (eta * A_out)
                        p.copy_(W + W @ exp_in + exp_out @ W)

        return loss


class PionMinimal(Optimizer):
    """Minimal Pion: keep ONLY the orthogonal-transformation core.

    Removed vs the full Pion: first/second-order momentum, bias correction,
    RMS scaling (Consistent Update), and the additive variant. Kept: the
    Lie-algebra construction and the second-order truncated exponential, in
    the multiplicative form (paper Algorithm 1).

    Update:
        Gin = W^T G - G^T W,  Gout = G W^T - W G^T
        E_in  = I - lr*Gin  + 0.5*(lr*Gin)^2
        E_out = I - lr*Gout + 0.5*(lr*Gout)^2
        W <- E_out @ W @ E_in
    """
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.999), eps=1e-8, rms_scaling=False, rms_constant=0.2):
        defaults = dict(lr=lr, betas=betas, eps=eps, rms_scaling=rms_scaling, rms_constant=rms_constant)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                G = p.grad
                if G.dim() < 2 or min(p.shape) <= 8:
                    # 1-D params / output layer: Adam (same as standard Pion)
                    state = self.state[p]
                    if len(state) == 0:
                        state["step"] = 0
                        state["m"] = torch.zeros_like(p)
                        state["v"] = torch.zeros_like(p)
                    state["step"] += 1
                    m, v = state["m"], state["v"]
                    m.mul_(beta1).add_(G, alpha=1 - beta1)
                    v.mul_(beta2).addcmul_(G, G, value=1 - beta2)
                    m_hat = m / (1 - beta1 ** state["step"])
                    v_hat = v / (1 - beta2 ** state["step"])
                    p.add_(m_hat / (v_hat.sqrt() + eps), alpha=-lr)
                else:
                    W = p
                    dout, din = W.shape
                    Gin = W.t() @ G - G.t() @ W
                    Gout = G @ W.t() - W @ G.t()
                    A_in = -Gin
                    A_out = -Gout
                    if group["rms_scaling"]:
                        # RMS-controlled scale consistency (no momentum): normalize
                        # rotation magnitude across layers of different sizes.
                        c = group["rms_constant"]
                        denom = torch.norm(A_out @ W + W @ A_in) + eps
                        eta = lr * c * math.sqrt(dout * din) / denom
                    else:
                        eta = lr
                    I_in = torch.eye(din, device=W.device, dtype=W.dtype)
                    I_out = torch.eye(dout, device=W.device, dtype=W.dtype)
                    E_in = I_in + eta * A_in + 0.5 * (eta * A_in) @ (eta * A_in)
                    E_out = I_out + eta * A_out + 0.5 * (eta * A_out) @ (eta * A_out)
                    p.copy_(E_out @ W @ E_in)
        return loss
