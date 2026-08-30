"""Verify the fixed L2-ER effective-rank term actually produces non-zero gradients.

Reproduces the exact erank + hidden_activations logic now in agent.py, then checks
that backward() flows gradient into the Linear weights.
"""
import torch
import torch.nn as nn


def erank(A):
    s = torch.linalg.svdvals(A)
    s = torch.abs(s)
    total = s.sum().clamp_min(1e-8)
    p = s / total
    H = -(p * torch.log(p + 1e-12)).sum()
    return torch.exp(H)


def hidden_activations(net, obs):
    x = obs
    acts = []
    for module in net:
        if isinstance(module, nn.Linear):
            acts.append(x)
        x = module(x)
    return acts[1:]


net = nn.Sequential(
    nn.Linear(10, 64),
    nn.Tanh(),
    nn.Linear(64, 64),
    nn.Tanh(),
    nn.Linear(64, 4),
)

obs = torch.randn(32, 10)  # batch=32, obs_dim=10

acts = hidden_activations(net, obs)
print("num hidden activations:", len(acts), "shapes:", [tuple(a.shape) for a in acts])

loss_er = sum(erank(A) for A in acts)
loss_er.backward()

grads = [p.grad.abs().sum().item() for p in net.parameters() if p.grad is not None]
print("loss_er =", loss_er.item())
print("per-layer |grad| sums:", grads)
print("ANY NON-ZERO GRAD:", any(g > 1e-8 for g in grads))
