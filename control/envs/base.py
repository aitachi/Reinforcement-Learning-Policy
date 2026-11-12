"""Vectorized batched environments for the control suite.

All environments are pure PyTorch: a batch of B independent episodes lives on
one device as tensors of shape (B, ...).  ``step`` advances every episode one
control step; finished episodes auto-reset and the boundary is reported through
``info`` so on-policy trainers can cut trajectories (no cross-episode bootstrapping).

Envs expose *raw task signals* (distances, collisions, successes, efforts...)
instead of a scalar reward.  The reward paradigm layer (rewards/paradigms.py)
maps signals -> reward, which is what lets one task be trained under sparse /
dense / potential-based / curriculum paradigms without touching physics.
"""
from __future__ import annotations

import torch


class VecEnv:
    """Minimal synchronous batched env contract."""

    obs_dim: int
    act_dim: int
    max_steps: int          # control steps per episode (truncation)
    num_envs: int
    signal_keys: tuple      # names of per-step signal tensors exposed to reward layers

    def __init__(self, num_envs: int, seed: int, device: torch.device):
        self.num_envs = num_envs
        self.device = device
        self.gen = torch.Generator(device=device)
        self.gen.manual_seed(seed)
        self._progress = 0.0  # curriculum progress in [0,1], set by trainer

    # -- lifecycle ---------------------------------------------------------
    def reset(self) -> torch.Tensor:
        raise NotImplementedError

    def step(self, action: torch.Tensor):
        """action (B, act_dim) in [-1,1] -> obs, signals(dict of (B,) tensors), done, info"""
        raise NotImplementedError

    # -- helpers -----------------------------------------------------------
    def set_progress(self, p: float):
        self._progress = float(min(max(p, 0.0), 1.0))

    def rand(self, *shape, lo=0.0, hi=1.0):
        return lo + (hi - lo) * torch.rand(*shape, generator=self.gen, device=self.device)

    def randn(self, *shape):
        return torch.randn(*shape, generator=self.gen, device=self.device)

    def randint(self, lo, hi, *shape):
        return torch.randint(lo, hi, *shape, generator=self.gen, device=self.device)

    @staticmethod
    def angle_wrap(a: torch.Tensor) -> torch.Tensor:
        return (a + torch.pi) % (2 * torch.pi) - torch.pi
