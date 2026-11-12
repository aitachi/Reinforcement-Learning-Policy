"""Networks, Gaussian policy utilities and the on-policy rollout storage.

Implementation details follow "Implementation Matters in Deep RL" (Engstrom
et al., 2020): orthogonal initialization (sqrt(2) body, 0.01 policy head),
separate value head, Adam, global grad-norm clipping.
"""
from __future__ import annotations

import math

import torch
from torch import nn

LOG2PI = math.log(2.0 * math.pi)


def mlp(in_dim, hidden, out_dim, out_gain=1.0):
    layers = []
    last = in_dim
    for h in hidden:
        lin = nn.Linear(last, h)
        nn.init.orthogonal_(lin.weight, 2 ** 0.5)
        nn.init.zeros_(lin.bias)
        layers += [lin, nn.Tanh()]
        last = h
    head = nn.Linear(last, out_dim)
    nn.init.orthogonal_(head.weight, out_gain)
    nn.init.zeros_(head.bias)
    layers.append(head)
    return nn.Sequential(*layers)


class GaussianActor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=(128, 128)):
        super().__init__()
        layers, last = [], obs_dim
        for h in hidden:
            lin = nn.Linear(last, h)
            nn.init.orthogonal_(lin.weight, 2 ** 0.5)
            nn.init.zeros_(lin.bias)
            layers += [lin, nn.Tanh()]
            last = h
        self.body = nn.Sequential(*layers)
        self.mu = nn.Linear(last, act_dim)
        nn.init.orthogonal_(self.mu.weight, 0.01)
        nn.init.zeros_(self.mu.bias)
        self.log_std = nn.Parameter(torch.full((act_dim,), -0.5))

    def dist_params(self, obs):
        h = self.body(obs)
        mu = self.mu(h)
        log_std = self.log_std.clamp(-5.0, 2.0)
        return mu, log_std.expand_as(mu)

    @staticmethod
    def _logp(pre_a, mu, log_std):
        std = log_std.exp()
        lp = (-0.5 * ((pre_a - mu) / std) ** 2 - log_std - 0.5 * LOG2PI).sum(-1)
        corr = (1 - pre_a.tanh() ** 2 + 1e-6).log().sum(-1)   # tanh squash correction
        return lp - corr

    def forward(self, obs):
        mu, log_std = self.dist_params(obs)
        pre = mu + log_std.exp() * torch.randn_like(mu)
        return pre.tanh(), self._logp(pre, mu, log_std), pre

    @torch.no_grad()
    def act(self, obs, deterministic=False):
        mu, log_std = self.dist_params(obs)
        if deterministic:
            return mu.tanh(), None, mu
        pre = mu + log_std.exp() * torch.randn_like(mu)
        return pre.tanh(), self._logp(pre, mu, log_std), pre

    def evaluate(self, obs, pre_a):
        mu, log_std = self.dist_params(obs)
        logp = self._logp(pre_a, mu, log_std)
        entropy = (log_std + 0.5 * LOG2PI + 0.5).sum(-1)
        return logp, mu, log_std, entropy


class ValueNet(nn.Module):
    def __init__(self, obs_dim, hidden=(128, 128)):
        super().__init__()
        self.net = mlp(obs_dim, hidden, 1, out_gain=1.0)

    def forward(self, obs):
        return self.net(obs).squeeze(-1)


class ActorCritic(nn.Module):
    """Shared-trunk actor-critic (used by A2C; PPO/TRPO use separate nets by default)."""

    def __init__(self, obs_dim, act_dim, hidden=(128, 128)):
        super().__init__()
        self.actor = GaussianActor(obs_dim, act_dim, hidden)
        self.critic = ValueNet(obs_dim, hidden)


def gaussian_kl(mu0, log_std0, mu1, log_std1):
    """KL( N(mu0, std0) || N(mu1, std1) ), diagonal, per-sample sum over dims."""
    var0, var1 = (2 * log_std0).exp(), (2 * log_std1).exp()
    return (log_std1 - log_std0 + (var0 + (mu1 - mu0) ** 2) / (2 * var1) - 0.5).sum(-1)


class Rollout:
    """Time-major on-policy buffer with GAE."""

    def __init__(self, T, B, obs_dim, act_dim, device):
        self.T, self.B = T, B
        self.device = device
        self.obs = torch.zeros(T, B, obs_dim, device=device)
        self.pre_a = torch.zeros(T, B, act_dim, device=device)   # pre-tanh actions
        self.a = torch.zeros(T, B, act_dim, device=device)       # tanh-squashed
        self.logp = torch.zeros(T, B, device=device)
        self.r = torch.zeros(T, B, device=device)
        self.d = torch.zeros(T, B, device=device)
        self.v = torch.zeros(T, B, device=device)
        self.t = 0

    def add(self, obs, pre_a, a, logp, r, d, v):
        i = self.t
        self.obs[i], self.pre_a[i], self.a[i] = obs, pre_a, a
        self.logp[i], self.r[i], self.d[i], self.v[i] = logp, r, d, v
        self.t += 1

    @property
    def full(self):
        return self.t >= self.T

    def gae(self, critic, last_obs, gamma=0.99, lam=0.95):
        with torch.no_grad():
            next_v = critic(last_obs)
            adv = torch.zeros_like(self.v)
            running = torch.zeros(self.B, device=self.device)
            for t in reversed(range(self.T)):
                not_done = 1.0 - self.d[t]
                delta = self.r[t] + gamma * next_v * not_done - self.v[t]
                running = delta + gamma * lam * not_done * running
                adv[t] = running
                next_v = self.v[t]
        ret = adv + self.v
        return adv, ret
