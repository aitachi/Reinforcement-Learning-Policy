"""SAC reference baseline (Haarnoja et al., 2018) - the off-policy anchor.

Twin Q, target nets, automatic entropy tuning (target = -act_dim), Gaussian
policy with tanh squash.  Included so the paper can position the on-policy
family against the strongest standard off-policy baseline on the same tasks.
"""
from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn

from .nets import mlp, LOG2PI


class SAC:
    needs = "step"
    kind = "sac"

    def __init__(self, obs_dim, act_dim, device, cfg):
        self.device = device
        self.act_dim = act_dim
        hid = (256, 256)
        self.policy_net = mlp(obs_dim, hid, act_dim * 2, out_gain=1e-2).to(device)
        self.q1 = mlp(obs_dim + act_dim, hid, 1, out_gain=1.0).to(device)
        self.q2 = mlp(obs_dim + act_dim, hid, 1, out_gain=1.0).to(device)
        self.q1_t = mlp(obs_dim + act_dim, hid, 1, out_gain=1.0).to(device)
        self.q2_t = mlp(obs_dim + act_dim, hid, 1, out_gain=1.0).to(device)
        self.q1_t.load_state_dict(self.q1.state_dict())
        self.q2_t.load_state_dict(self.q2.state_dict())
        self.opt_pi = torch.optim.Adam(self.policy_net.parameters(), lr=cfg.get("sac_lr", 3e-4))
        self.opt_q = torch.optim.Adam(list(self.q1.parameters()) + list(self.q2.parameters()),
                                      lr=cfg.get("sac_lr", 3e-4))
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.opt_a = torch.optim.Adam([self.log_alpha], lr=cfg.get("sac_lr", 3e-4))
        self.target_entropy = -float(act_dim)
        self.tau = cfg.get("sac_tau", 0.005)
        self.gamma = cfg.get("gamma", 0.99)
        self.batch = cfg.get("sac_batch", 512)
        self.warmup = cfg.get("sac_warmup", 4000)
        self.updates_per_step = cfg.get("sac_utd", 2)
        self.replay_capacity = cfg.get("sac_buffer", 600_000)
        self._buf = {k: None for k in ("obs", "act", "rew", "obs2", "done")}
        self._n = 0
        self._ptr = 0
        self._steps_collected = 0

    # -------------------------------------------------------------- replay
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
        for k, v in (("obs", obs), ("act", act), ("rew", rew), ("obs2", obs2), ("done", done)):
            t = self._buf[k]
            idx = (self._ptr + torch.arange(B, device=self.device)) % self.replay_capacity
            t[idx] = v
        self._ptr = (self._ptr + B) % self.replay_capacity
        self._n = min(self._n + B, self.replay_capacity)
        self._steps_collected += B

    # ------------------------------------------------------------- actions
    def _pi(self, obs):
        out = self.policy_net(obs)
        mu, log_std = out.chunk(2, -1)
        log_std = log_std.clamp(-20, 2)
        std = log_std.exp()
        pre = mu + std * torch.randn_like(mu)
        a = pre.tanh()
        lp = (-0.5 * ((pre - mu) / std) ** 2 - log_std - 0.5 * math.log(2 * math.pi)).sum(-1)
        lp = lp - (1 - a ** 2 + 1e-6).log().sum(-1)
        return a, lp

    def act(self, obs):
        with torch.no_grad():
            a, _ = self._pi(obs)
        return a

    # ------------------------------------------------------------- update
    def update_if_ready(self):
        if self._steps_collected < self.warmup:
            return None
        recs = []
        for _ in range(self.updates_per_step):
            idx = torch.randint(0, self._n, (self.batch,), device=self.device)
            s, a, r, s2, d = (self._buf[k][idx] for k in ("obs", "act", "rew", "obs2", "done"))
            with torch.no_grad():
                a2, lp2 = self._pi(s2)
                q_t = torch.min(self.q1_t(torch.cat([s2, a2], -1)),
                                self.q2_t(torch.cat([s2, a2], -1))).squeeze(-1)
                target = r + self.gamma * (1 - d) * (q_t - self.log_alpha.exp() * lp2)
            q1v = self.q1(torch.cat([s, a], -1)).squeeze(-1)
            q2v = self.q2(torch.cat([s, a], -1)).squeeze(-1)
            q_loss = ((q1v - target) ** 2).mean() + ((q2v - target) ** 2).mean()
            self.opt_q.zero_grad(set_to_none=True)
            q_loss.backward()
            self.opt_q.step()

            a_new, lp_new = self._pi(s)
            q_new = torch.min(self.q1(torch.cat([s, a_new], -1)),
                              self.q2(torch.cat([s, a_new], -1))).squeeze(-1)
            pi_loss = (self.log_alpha.exp() * lp_new - q_new).mean()
            self.opt_pi.zero_grad(set_to_none=True)
            pi_loss.backward()
            self.opt_pi.step()

            alpha_loss = -(self.log_alpha.exp() * (lp_new.detach() + self.target_entropy)).mean()
            self.opt_a.zero_grad(set_to_none=True)
            alpha_loss.backward()
            self.opt_a.step()

            with torch.no_grad():
                for q, qt in ((self.q1, self.q1_t), (self.q2, self.q2_t)):
                    for p, pt in zip(q.parameters(), qt.parameters()):
                        pt.lerp_(p, self.tau)
            recs.append({"pi_loss": pi_loss.item(), "q_loss": q_loss.item(),
                         "alpha": self.log_alpha.exp().item(),
                         "entropy": -lp_new.mean().item()})
        out = {k: float(np.mean([x[k] for x in recs])) for k in recs[0]}
        out["optimizer_steps"] = len(recs)
        return out


class SACTR(SAC):
    """SAC with a KL trust region on the actor update (the off-policy
    counterpart of PPO-KL: same constraint family, different sampling).

    After each proposed Adam step on the actor, the batch KL between the
    pre-step and post-step Gaussians is measured; if it exceeds
    sac_tr_delta the step is backtracked (linear interpolation toward the
    old parameters) until the budget is met, mirroring TRPO's line search.
    """

    def __init__(self, obs_dim, act_dim, device, cfg):
        super().__init__(obs_dim, act_dim, device, cfg)
        self.tr_delta = cfg.get("sac_tr_delta", 0.01)
        self.tr_backtracks = 0
        self.tr_last_kl = 0.0

    def _kl(self, obs):
        out = self.policy_net(obs)
        mu, log_std = out.chunk(2, -1)
        var_new = (2 * log_std.clamp(-20, 2)).exp()
        return 0.5 * (((self._old_std ** 2) / var_new
                       + ((self._old_mu - mu) ** 2) / var_new
                       - 2 + (2 * (log_std.clamp(-20, 2) - self._old_logstd))).sum(-1)).mean()

    def _snapshot(self):
        with torch.no_grad():
            out = self.policy_net(self._kl_batch)
            mu, log_std = out.chunk(2, -1)
            self._old_mu, self._old_logstd = mu, log_std.clamp(-20, 2)
            self._old_std = self._old_logstd.exp()
            self._old_params = [p.detach().clone() for p in self.policy_net.parameters()]

    def _restore_interpolated(self, t):
        with torch.no_grad():
            for p, old in zip(self.policy_net.parameters(), self._old_params):
                p.copy_(old + t * (p - old))

    def update_if_ready(self):
        if self._steps_collected < self.warmup:
            return None
        recs = []
        for _ in range(self.updates_per_step):
            idx = torch.randint(0, self._n, (self.batch,), device=self.device)
            s, a, r, s2, d = (self._buf[k][idx] for k in ("obs", "act", "rew", "obs2", "done"))
            with torch.no_grad():
                a2, lp2 = self._pi(s2)
                q_t = torch.min(self.q1_t(torch.cat([s2, a2], -1)),
                                self.q2_t(torch.cat([s2, a2], -1))).squeeze(-1)
                target = r + self.gamma * (1 - d) * (q_t - self.log_alpha.exp() * lp2)
            q1v = self.q1(torch.cat([s, a], -1)).squeeze(-1)
            q2v = self.q2(torch.cat([s, a], -1)).squeeze(-1)
            q_loss = ((q1v - target) ** 2).mean() + ((q2v - target) ** 2).mean()
            self.opt_q.zero_grad(set_to_none=True)
            q_loss.backward()
            self.opt_q.step()

            # --- actor step under a KL trust region ----------------------
            self._kl_batch = s
            self._snapshot()
            a_new, lp_new = self._pi(s)
            q_new = torch.min(self.q1(torch.cat([s, a_new], -1)),
                              self.q2(torch.cat([s, a_new], -1))).squeeze(-1)
            pi_loss = (self.log_alpha.exp() * lp_new - q_new).mean()
            self.opt_pi.zero_grad(set_to_none=True)
            pi_loss.backward()
            self.opt_pi.step()
            with torch.no_grad():
                kl = self._kl(s)
                t_step = 1.0
                for _ in range(6):
                    if kl <= self.tr_delta:
                        break
                    t_step *= 0.5
                    self._restore_interpolated(t_step)
                    kl = self._kl(s)
                self.tr_last_kl = float(kl)
                self.tr_backtracks += int(t_step < 1.0)

            alpha_loss = -(self.log_alpha.exp() * (lp_new.detach() + self.target_entropy)).mean()
            self.opt_a.zero_grad(set_to_none=True)
            alpha_loss.backward()
            self.opt_a.step()

            with torch.no_grad():
                for q, qt in ((self.q1, self.q1_t), (self.q2, self.q2_t)):
                    for p, pt in zip(q.parameters(), qt.parameters()):
                        pt.lerp_(p, self.tau)
            recs.append({"pi_loss": pi_loss.item(), "q_loss": q_loss.item(),
                         "alpha": self.log_alpha.exp().item(),
                         "entropy": -lp_new.mean().item(),
                         "actor_kl": self.tr_last_kl})
        out = {k: float(np.mean([x[k] for x in recs])) for k in recs[0]}
        out["optimizer_steps"] = len(recs)
        out["tr_backtracks_total"] = float(self.tr_backtracks)
        return out
