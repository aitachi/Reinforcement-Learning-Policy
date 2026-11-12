"""TD3 (Fujimoto et al., 2018) - twin delayed DDPG, off-policy baseline.

Clipped double-Q, delayed policy updates (every 2nd Q update), and target
policy smoothing (Gaussian noise on target actions).  Deterministic policy
with exploration noise; shares the GPU replay layout with SAC.
"""
from __future__ import annotations

import numpy as np
import torch

from .nets import mlp


class TD3:
    needs = "step"
    kind = "td3"

    def __init__(self, obs_dim, act_dim, device, cfg):
        self.device = device
        self.act_dim = act_dim
        hid = (256, 256)
        self.policy_net = mlp(obs_dim, hid, act_dim, out_gain=1e-2).to(device)
        self.policy_t = mlp(obs_dim, hid, act_dim, out_gain=1e-2).to(device)
        self.policy_t.load_state_dict(self.policy_net.state_dict())
        self.q1 = mlp(obs_dim + act_dim, hid, 1, out_gain=1.0).to(device)
        self.q2 = mlp(obs_dim + act_dim, hid, 1, out_gain=1.0).to(device)
        self.q1_t = mlp(obs_dim + act_dim, hid, 1, out_gain=1.0).to(device)
        self.q2_t = mlp(obs_dim + act_dim, hid, 1, out_gain=1.0).to(device)
        self.q1_t.load_state_dict(self.q1.state_dict())
        self.q2_t.load_state_dict(self.q2.state_dict())
        self.opt_pi = torch.optim.Adam(self.policy_net.parameters(), lr=cfg.get("td3_lr", 1e-3))
        self.opt_q = torch.optim.Adam(list(self.q1.parameters()) + list(self.q2.parameters()),
                                      lr=cfg.get("td3_lr", 1e-3))
        self.gamma = cfg.get("gamma", 0.99)
        self.tau = cfg.get("td3_tau", 0.005)
        self.batch = cfg.get("td3_batch", 1024)
        self.warmup = cfg.get("td3_warmup", 4000)
        self.expl_noise = cfg.get("td3_expl_noise", 0.15)
        self.policy_noise = cfg.get("td3_policy_noise", 0.2)
        self.noise_clip = cfg.get("td3_noise_clip", 0.5)
        self.policy_delay = cfg.get("td3_policy_delay", 2)
        self.updates_per_step = cfg.get("td3_utd", 16)
        self.replay_capacity = cfg.get("td3_buffer", 600_000)
        self._buf = {k: None for k in ("obs", "act", "rew", "obs2", "done")}
        self._n = 0
        self._ptr = 0
        self._steps_collected = 0
        self._it = 0

    def _alloc(self, k, shape):
        if self._buf[k] is None:
            self._buf[k] = torch.zeros((self.replay_capacity, *shape), device=self.device)

    def store(self, obs, act, rew, obs2, done):
        B = obs.shape[0]
        self._alloc("obs", (obs.shape[1],))
        self._alloc("act", (act.shape[1],))
        self._alloc("rew", ())
        self._alloc("obs2", (obs2.shape[1],))
        self._alloc("done", ())
        idx = (self._ptr + torch.arange(B, device=self.device)) % self.replay_capacity
        for k, v in (("obs", obs), ("act", act), ("rew", rew), ("obs2", obs2), ("done", done)):
            self._buf[k][idx] = v
        self._ptr = (self._ptr + B) % self.replay_capacity
        self._n = min(self._n + B, self.replay_capacity)
        self._steps_collected += B

    def act(self, obs, deterministic=False):
        with torch.no_grad():
            a = self.policy_net(obs).clamp(-1, 1)
            if deterministic:
                return a
            if self._steps_collected > self.warmup:
                a = (a + self.expl_noise * torch.randn_like(a)).clamp(-1, 1)
            else:
                a = torch.rand_like(a) * 2 - 1
        return a

    def update_if_ready(self):
        if self._steps_collected < self.warmup:
            return None
        recs = []
        for _ in range(self.updates_per_step):
            self._it += 1
            idx = torch.randint(0, self._n, (self.batch,), device=self.device)
            s, a, r, s2, d = (self._buf[k][idx] for k in ("obs", "act", "rew", "obs2", "done"))
            with torch.no_grad():
                noise = (self.policy_noise * torch.randn_like(a)).clamp(-self.noise_clip, self.noise_clip)
                a2 = (self.policy_t(s2) + noise).clamp(-1, 1)
                q_t = torch.min(self.q1_t(torch.cat([s2, a2], -1)),
                                self.q2_t(torch.cat([s2, a2], -1))).squeeze(-1)
                target = r + self.gamma * (1 - d) * q_t
            q1v = self.q1(torch.cat([s, a], -1)).squeeze(-1)
            q2v = self.q2(torch.cat([s, a], -1)).squeeze(-1)
            q_loss = ((q1v - target) ** 2).mean() + ((q2v - target) ** 2).mean()
            self.opt_q.zero_grad(set_to_none=True)
            q_loss.backward()
            self.opt_q.step()

            if self._it % self.policy_delay == 0:
                a_new = self.policy_net(s).clamp(-1, 1)
                pi_loss = -self.q1(torch.cat([s, a_new], -1)).squeeze(-1).mean()
                self.opt_pi.zero_grad(set_to_none=True)
                pi_loss.backward()
                self.opt_pi.step()
                with torch.no_grad():
                    for net, net_t in ((self.policy_net, self.policy_t),
                                       (self.q1, self.q1_t), (self.q2, self.q2_t)):
                        for p, pt in zip(net.parameters(), net_t.parameters()):
                            pt.lerp_(p, self.tau)
                recs.append({"pi_loss": pi_loss.item(), "q_loss": q_loss.item(),
                             "q1": q1v.mean().item()})
        if not recs:
            return None
        out = {k: float(np.mean([x[k] for x in recs])) for k in recs[0]}
        out["optimizer_steps"] = len(recs)
        return out
