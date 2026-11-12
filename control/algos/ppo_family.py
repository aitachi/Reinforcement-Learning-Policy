"""PPO variant family + A2C for continuous control.

Methods (all share collection, GAE, and optimizer settings):
  ppo         - clipped surrogate (Schulman et al., 2017), value clipping on
  ppo_kl      - adaptive-KL penalty variant (beta doubles/halves around target)
  ppo_rb      - PPO-Rollback (Yuan et al., 2020): outside the clip band the
                surrogate slope becomes -alpha instead of a flat plateau
  tr_ppo_rb   - trust-region PPO-RB: per-state KL trigger replaces the band
  espo        - Early-Stopped PPO (Li et al., 2023): unclipped objective,
                stop the epoch when mean |ratio-1| exceeds a threshold
  ppg         - Phasic Policy Gradient (Cobbe et al., 2021): PPO phase +
                periodic auxiliary phase (value/aux-feature distillation with
                a KL clone anchor), disjoint parameters per phase
  a2c         - synchronous advantage actor-critic, single-pass, no ratio

Continuous actions use a tanh-squashed diagonal Gaussian; log-probs include
the squash correction so ratios are exact.
"""
from __future__ import annotations

import copy

import numpy as np
import torch
from torch import nn

from .nets import GaussianActor, ValueNet, gaussian_kl


class PPO:
    needs = "transition"
    kind = "ppo"

    def __init__(self, obs_dim, act_dim, device, cfg):
        self.device = device
        self.policy = GaussianActor(obs_dim, act_dim).to(device)
        self.critic = ValueNet(obs_dim).to(device)
        self.opt = torch.optim.Adam(
            list(self.policy.parameters()) + list(self.critic.parameters()), lr=cfg["lr"])
        self.cfg = cfg
        self.method = cfg.get("method", "clip")
        self.beta = cfg.get("kl_beta", 0.1)
        self.clip = cfg.get("clip", 0.2)
        self.epochs = cfg.get("epochs", 4)
        self.mb = cfg.get("minibatch", 2048)
        self.ent_coef = cfg.get("ent_coef", 0.0)
        self.vf_coef = cfg.get("vf_coef", 0.5)
        self.max_grad = cfg.get("max_grad", 0.5)
        self.value_clip = cfg.get("value_clip", True)
        # PPO-RB / TR-PPO-RB constants (Yuan et al. 2020)
        self.rb_alpha = cfg.get("rb_alpha", 0.2)
        self.tr_kl = cfg.get("tr_kl", 6.0 * cfg.get("kl_target", 0.02))
        # ESPO
        self.espo_max_epochs = cfg.get("espo_max_epochs", 12)
        self.espo_thr = cfg.get("espo_threshold", 0.25)

    # ------------------------------------------------------------- update
    def update(self, rollout, last_obs):
        cfg = self.cfg
        adv, ret = rollout.gae(self.critic, last_obs, cfg["gamma"], cfg["lam"])
        T, B = rollout.T, rollout.B
        obs = rollout.obs.reshape(T * B, -1)
        pre_a = rollout.pre_a.reshape(T * B, -1)
        old_logp = rollout.logp.reshape(T * B)
        old_v = rollout.v.reshape(T * B)
        adv = adv.reshape(T * B)
        ret = ret.reshape(T * B)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        mu0, ls0 = self.policy.dist_params(obs)
        old_mu, old_ls = mu0.detach(), ls0.detach()

        max_epochs = self.espo_max_epochs if self.method == "espo" else self.epochs
        records = []
        epochs_done = 0
        for ep in range(max_epochs):
            epochs_done = ep + 1
            perm = torch.randperm(T * B, device=self.device)
            for idx in perm.split(self.mb):
                logp, mu, ls, ent = self.policy.evaluate(obs[idx], pre_a[idx])
                v = self.critic(obs[idx])
                ratio = (logp - old_logp[idx]).exp()
                a = adv[idx]
                if self.method in ("clip", "ppg"):
                    sur = torch.min(ratio * a, ratio.clamp(1 - self.clip, 1 + self.clip) * a)
                elif self.method == "kl":
                    kl = gaussian_kl(old_mu[idx], old_ls[idx], mu, ls).mean()
                    sur = ratio * a - self.beta * kl
                elif self.method == "ppo_rb":
                    edge = ratio.clamp(1 - self.clip, 1 + self.clip)
                    rb = edge - self.rb_alpha * (ratio - edge)
                    sur = torch.min(ratio * a, rb * a)
                elif self.method == "tr_ppo_rb":
                    with torch.no_grad():
                        skl = gaussian_kl(old_mu[idx], old_ls[idx], mu, ls)
                    rb = torch.where(skl >= self.tr_kl, -self.rb_alpha * ratio, ratio)
                    sur = torch.min(ratio * a, rb * a)
                elif self.method == "espo":
                    sur = ratio * a
                else:
                    raise ValueError(self.method)
                pi_loss = -sur.mean()
                if self.value_clip:
                    v_clip = old_v[idx] + (v - old_v[idx]).clamp(-self.clip, self.clip)
                    vf_loss = 0.5 * torch.max((v - ret[idx]) ** 2, (v_clip - ret[idx]) ** 2).mean()
                else:
                    vf_loss = 0.5 * (v - ret[idx]).pow(2).mean()
                loss = pi_loss + self.vf_coef * vf_loss - self.ent_coef * ent.mean()
                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.policy.parameters()) + list(self.critic.parameters()), self.max_grad)
                self.opt.step()
                with torch.no_grad():
                    rec = {"pi_loss": pi_loss.item(), "vf_loss": vf_loss.item(),
                           "entropy": ent.mean().item(),
                           "approx_kl": (ratio - 1 - (logp - old_logp[idx])).mean().item(),
                           "clip_frac": ((ratio - 1).abs() > self.clip).float().mean().item(),
                           "log_std": self.policy.log_std.mean().item()}
                    records.append(rec)
            if self.method == "espo":
                with torch.no_grad():
                    lp = self.policy.evaluate(obs, pre_a)[0]
                    dev = ((lp - old_logp).exp() - 1).abs().mean().item()
                if dev > self.espo_thr and ep + 1 < max_epochs:
                    records[-1]["early_stop"] = 1.0
                    break
        if self.method == "kl":
            with torch.no_grad():
                lp, mu, ls, _ = self.policy.evaluate(obs, pre_a)
                post_kl = gaussian_kl(old_mu, old_ls, mu, ls).mean().item()
            tgt = cfg.get("kl_target", 0.02)
            if post_kl > 1.5 * tgt:
                self.beta = min(self.beta * 2, 1e3)
            elif post_kl < tgt / 1.5:
                self.beta = max(self.beta / 2, 1e-4)
        out = {k: float(np.mean([r[k] for r in records])) for k in records[0] if k != "early_stop"}
        out["epochs"] = epochs_done
        out["optimizer_steps"] = len(records)
        return out


class PPG(PPO):
    """Phasic Policy Gradient: PPO phase + periodic auxiliary phase."""

    kind = "ppg"

    def __init__(self, obs_dim, act_dim, device, cfg):
        super().__init__(obs_dim, act_dim, device, cfg)
        self.method = "ppg"
        self.n_pi = cfg.get("ppg_n_pi", 8)              # policy iters per aux phase
        self.aux_epochs = cfg.get("ppg_aux_epochs", 3)
        self.clone_beta = cfg.get("ppg_clone_beta", 1.0)
        self.aux_head = torch.nn.Linear(self.policy.body[-2].out_features, 1).to(device)
        nn.init.orthogonal_(self.aux_head.weight, 1.0)
        nn.init.zeros_(self.aux_head.bias)
        self.opt = torch.optim.Adam(
            list(self.policy.body.parameters()) + list(self.policy.mu.parameters())
            + [self.policy.log_std] + list(self.aux_head.parameters()), lr=cfg["lr"])
        self.opt_v = torch.optim.Adam(self.critic.parameters(), lr=cfg["lr"])
        self._it = 0
        self._aux_buf = []

    def update(self, rollout, last_obs):
        # value/policy phases use disjoint optimizers (phasic)
        adv, ret = rollout.gae(self.critic, last_obs, self.cfg["gamma"], self.cfg["lam"])
        T, B = rollout.T, rollout.B
        obs = rollout.obs.reshape(T * B, -1)
        pre_a = rollout.pre_a.reshape(T * B, -1)
        old_logp = rollout.logp.reshape(T * B)
        adv = adv.reshape(T * B)
        ret = ret.reshape(T * B)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        # 1) value phase
        for _ in range(self.cfg.get("epochs", 4)):
            perm = torch.randperm(T * B, device=self.device)
            for idx in perm.split(self.mb):
                v = self.critic(obs[idx])
                loss = 0.5 * (v - ret[idx]).pow(2).mean()
                self.opt_v.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad)
                self.opt_v.step()
        # 2) policy phase (actor only)
        pi_records = []
        for ep in range(self.cfg.get("epochs", 4)):
            perm = torch.randperm(T * B, device=self.device)
            for idx in perm.split(self.mb):
                logp, mu, ls, ent = self.policy.evaluate(obs[idx], pre_a[idx])
                ratio = (logp - old_logp[idx]).exp()
                sur = torch.min(ratio * adv[idx],
                                ratio.clamp(1 - self.clip, 1 + self.clip) * adv[idx])
                loss = -sur.mean() - self.ent_coef * ent.mean()
                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.policy.body.parameters()) + list(self.policy.mu.parameters())
                    + [self.policy.log_std], self.max_grad)
                self.opt.step()
                pi_records.append({"pi_loss": loss.item(), "entropy": ent.mean().item(),
                                   "clip_frac": ((ratio - 1).abs() > self.clip).float().mean().item(),
                                   "approx_kl": (ratio - 1 - (logp - old_logp[idx])).mean().item(),
                                   "vf_loss": 0.0, "log_std": self.policy.log_std.mean().item()})
        self._aux_buf.append((obs.detach(), ret.detach()))
        self._it += 1
        out = {k: float(np.mean([r[k] for r in pi_records])) for k in pi_records[0]}
        out["optimizer_steps"] = len(pi_records)
        # 3) auxiliary phase every n_pi iterations: trunk learns value through a
        #    dedicated aux head while a KL clone anchor pins the policy head.
        if self._it % self.n_pi == 0:
            aobs = torch.cat([o for o, _ in self._aux_buf])
            aret = torch.cat([r for _, r in self._aux_buf])
            self._aux_buf = []
            with torch.no_grad():
                a_mu, a_ls = self.policy.dist_params(aobs)
            for _ in range(self.aux_epochs):
                perm = torch.randperm(len(aobs), device=self.device)
                for idx in perm.split(self.mb):
                    h = self.policy.body(aobs[idx])
                    mu = self.policy.mu(h)
                    ls = self.policy.log_std.clamp(-5.0, 2.0).expand_as(mu)
                    kl = gaussian_kl(a_mu[idx], a_ls[idx], mu, ls).mean()
                    aux = 0.5 * (self.aux_head(h).squeeze(-1) - aret[idx]).pow(2).mean()
                    loss = aux + self.clone_beta * kl
                    self.opt.zero_grad(set_to_none=True)
                    loss.backward()
                    self.opt.step()
            out["aux_loss"] = loss.item()
        return out


class A2C:
    needs = "transition"
    kind = "a2c"

    def __init__(self, obs_dim, act_dim, device, cfg):
        self.device = device
        self.policy = GaussianActor(obs_dim, act_dim).to(device)
        self.critic = ValueNet(obs_dim).to(device)
        self.opt = torch.optim.Adam(
            list(self.policy.parameters()) + list(self.critic.parameters()), lr=cfg.get("a2c_lr", 7e-4))
        self.cfg = cfg
        self.ent_coef = cfg.get("ent_coef", 0.01)
        self.vf_coef = cfg.get("vf_coef", 0.5)
        self.max_grad = cfg.get("max_grad", 0.5)

    def update(self, rollout, last_obs):
        adv, ret = rollout.gae(self.critic, last_obs, self.cfg["gamma"], 1.0)  # lambda=1
        T, B = rollout.T, rollout.B
        obs = rollout.obs.reshape(T * B, -1)
        pre_a = rollout.pre_a.reshape(T * B, -1)
        adv = adv.reshape(T * B)
        ret = ret.reshape(T * B)
        logp, _, _, ent = self.policy.evaluate(obs, pre_a)
        v = self.critic(obs)
        pi_loss = -(adv.detach() * logp).mean()
        vf_loss = 0.5 * (v - ret).pow(2).mean()
        loss = pi_loss + self.vf_coef * vf_loss - self.ent_coef * ent.mean()
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.policy.parameters()) + list(self.critic.parameters()), self.max_grad)
        self.opt.step()
        return {"pi_loss": pi_loss.item(), "vf_loss": vf_loss.item(),
                "entropy": ent.mean().item(), "log_std": self.policy.log_std.mean().item(),
                "optimizer_steps": 1}
