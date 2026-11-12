"""TRPO for continuous control (Schulman et al., 2015).

Natural-gradient step: solve (F + damping I) x = g with conjugate gradient on
the exact diagonal-Gaussian KL Hessian (double backward), step length from
the quadratic model sqrt(2*delta / x^T F x), then backtracking line search on
the true surrogate with a hard KL constraint.  Critic fitted separately by
GAE regression (no policy gradient through the value net).
"""
from __future__ import annotations

import numpy as np
import torch

from .nets import GaussianActor, ValueNet, gaussian_kl
from torch.nn.utils import parameters_to_vector, vector_to_parameters


class TRPO:
    needs = "transition"
    kind = "trpo"

    def __init__(self, obs_dim, act_dim, device, cfg):
        self.device = device
        self.policy = GaussianActor(obs_dim, act_dim).to(device)
        self.critic = ValueNet(obs_dim).to(device)
        self.opt_c = torch.optim.Adam(self.critic.parameters(), lr=cfg["lr"])
        self.opt_pi = torch.optim.Adam(self.policy.parameters(), lr=cfg.get("pi_lr", 3e-5))
        self.cfg = cfg
        self.max_kl = cfg.get("trpo_max_kl", 0.01)
        self.damping = cfg.get("trpo_damping", 0.1)
        self.cg_steps = cfg.get("trpo_cg_steps", 10)
        self.backtracks = cfg.get("trpo_backtracks", 10)
        self.critic_steps = cfg.get("trpo_critic_steps", 8)
        self.mb = cfg.get("minibatch", 2048)

    def update(self, rollout, last_obs):
        adv, ret = rollout.gae(self.critic, last_obs, self.cfg["gamma"], self.cfg["lam"])
        T, B = rollout.T, rollout.B
        obs = rollout.obs.reshape(T * B, -1)
        pre_a = rollout.pre_a.reshape(T * B, -1)
        old_logp = rollout.logp.reshape(T * B)
        ret = ret.reshape(T * B)
        adv = adv.reshape(T * B)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        params = [p for p in self.policy.parameters()]
        # freeze the sampling distribution before any parameter change
        with torch.no_grad():
            old_mu, old_ls = self.policy.dist_params(obs)

        def pi_terms():
            mu, ls = self.policy.dist_params(obs)
            std = ls.exp()
            lp = (-0.5 * ((pre_a - mu) / std) ** 2 - ls - 0.5 * np.log(2 * np.pi)).sum(-1)
            corr = (1 - pre_a.tanh() ** 2 + 1e-6).log().sum(-1)
            logp = lp - corr
            sur = ((logp - old_logp).exp() * adv).mean()
            kl = gaussian_kl(old_mu, old_ls, mu, ls).mean()
            return sur, kl

        sur, kl = pi_terms()
        g = parameters_to_vector(torch.autograd.grad(sur, params, retain_graph=True)).detach()
        kl_grad = parameters_to_vector(torch.autograd.grad(kl, params, create_graph=True))

        def fvp(v):
            grad_v = (kl_grad * v).sum()
            hess = parameters_to_vector(torch.autograd.grad(grad_v, params, retain_graph=True))
            return hess.detach() + self.damping * v

        x = torch.zeros_like(g)
        r = g.clone()
        p = r.clone()
        rs = r.dot(r)
        for _ in range(self.cg_steps):
            if rs < 1e-12:
                break
            fp = fvp(p)
            curv = p.dot(fp)
            if not torch.isfinite(curv) or curv <= 0:
                break
            alpha = rs / curv
            x += alpha * p
            r -= alpha * fp
            rs_new = r.dot(r)
            p = r + rs_new / rs * p
            rs = rs_new
        try:
            curv = x.dot(fvp(x))
            step = (2 * self.max_kl / curv.clamp_min(1e-12)).sqrt() * x
        except Exception:
            step = torch.zeros_like(g)

        old_vec = parameters_to_vector(params).detach()
        accepted = False
        final_sur = sur.item()
        with torch.no_grad():
            for i in range(self.backtracks):
                coef = 0.5 ** i
                vector_to_parameters(old_vec + coef * step, params)
                s2, k2 = pi_terms()
                if k2.item() <= self.max_kl and s2.item() >= sur.item():
                    final_sur, accepted = s2.item(), True
                    break
            if not accepted:
                vector_to_parameters(old_vec, params)

        crit_losses = []
        for _ in range(self.critic_steps):
            perm = torch.randperm(T * B, device=self.device)
            for idx in perm.split(self.mb):
                v = self.critic(obs[idx])
                loss = 0.5 * (v - ret[idx]).pow(2).mean()
                self.opt_c.zero_grad(set_to_none=True)
                loss.backward()
                self.opt_c.step()
                crit_losses.append(loss.item())
        with torch.no_grad():
            _, k_after = pi_terms()
        return {"pi_loss": -final_sur, "vf_loss": float(np.mean(crit_losses)),
                "trpo_accept": float(accepted), "post_kl": k_after.item(),
                "log_std": self.policy.log_std.mean().item(),
                "optimizer_steps": self.critic_steps}
